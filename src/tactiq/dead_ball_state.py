"""Conservative state layer that turns dead-ball evidence into D labels.

This module consumes algorithmic evidence only. Smart Tagging is treated as
restart context, never as frame-level possession ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LabelResult:
    labels: np.ndarray
    confidence: np.ndarray


def _vector(name: str, values, n_frames: int, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if len(array) != n_frames:
        raise ValueError(f"{name} has {len(array)} frames; expected {n_frames}")
    return array


def apply_dead_ball_state(
    possession,
    features,
    evidence,
    smart_tagging=None,
    dynamics=None,
    *,
    release_confirm_frames: int = 3,
    restart_release_speed_mps: float = 1.0,
) -> LabelResult:
    """Overlay conservative D intervals on provisional H/A/X estimates.

    Entry conditions:
      * a rising edge of an automatically tagged restart interval; or
      * a strong dead-ball candidate from ``extract_dead_ball_evidence``.

    Exit condition:
      * the ball is inside the pitch, the base estimator has H/A, and this
        active-play evidence persists for ``release_confirm_frames`` frames.
      * While a restart tag is active, a provider-reported ball speed at or
        above ``restart_release_speed_mps`` can also support release.

    The function deliberately does not infer D from a missing or stationary
    ball alone. It also does not use Smart Tagging team tags as possession
    labels. Thresholds are conservative engineering defaults, not values
    selected against competition ground truth.
    """
    if isinstance(release_confirm_frames, bool) or not isinstance(
        release_confirm_frames, (int, np.integer)
    ) or release_confirm_frames < 1:
        raise ValueError("release_confirm_frames must be a positive integer")
    if not np.isfinite(restart_release_speed_mps) or restart_release_speed_mps < 0:
        raise ValueError(
            "restart_release_speed_mps must be finite and non-negative"
        )

    raw_labels = np.asarray(possession.labels)

    if raw_labels.ndim != 1:
        raise ValueError("possession.labels must be one-dimensional")

    if not np.isin(raw_labels, ["H", "A", "X", "D"]).all():
        raise ValueError("possession.labels contains an unsupported label")

    base_labels = raw_labels.astype("<U1", copy=False)
    n_frames = len(base_labels)

    inside = _vector(
        "features.ball_inside_boundary", features.ball_inside_boundary,
        n_frames,
    )
    candidate = _vector(
        "evidence.candidate", evidence.candidate, n_frames,
    )
    if not np.issubdtype(inside.dtype, np.bool_):
        raise ValueError("features.ball_inside_boundary must contain booleans")
    if not np.issubdtype(candidate.dtype, np.bool_):
        raise ValueError("evidence.candidate must contain booleans")

    restart_context = np.zeros(n_frames, dtype=bool)
    if smart_tagging is not None:
        restart_active = np.asarray(smart_tagging.restart_active)
        if restart_active.ndim != 2 or restart_active.shape[0] != n_frames:
            raise ValueError(
                "smart_tagging.restart_active must have shape (n_frames, n_restarts)"
            )
        if not np.issubdtype(restart_active.dtype, np.bool_):
            raise ValueError("smart_tagging.restart_active must contain booleans")
        restart_context = restart_active.any(axis=1)

    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    if dynamics is not None:
        ball_speed = _vector(
            "dynamics.ball_speed", dynamics.ball_speed, n_frames,
            dtype=np.float32,
        )
        invalid_speed = np.isinf(ball_speed) | (ball_speed < 0)
        if invalid_speed.any():
            raise ValueError("dynamics.ball_speed must be finite and non-negative")

    # Use only the beginning of a restart interval, not every active frame.
    restart_started = restart_context & ~np.concatenate(
        (np.array([False]), restart_context[:-1])
    )
    candidate_started = candidate & ~np.concatenate(
        (np.array([False]), candidate[:-1])
    )
    entry = restart_started | candidate_started

    labels = base_labels.copy()
    confidence = np.asarray(
        getattr(possession, "confidence", np.zeros(n_frames)),
        dtype=np.float32,
    ).copy()
    if confidence.ndim != 1 or len(confidence) != n_frames:
        raise ValueError("possession.confidence must match the label length")

    dead = False
    release_run = 0

    for frame in range(n_frames):
        if entry[frame]:
            dead = True
            release_run = 0

        if not dead:
            continue

        has_live_team = base_labels[frame] in ("H", "A")
        ball_inside = bool(inside[frame])
        speed_confirms_release = (
            np.isfinite(ball_speed[frame])
            and ball_speed[frame] >= restart_release_speed_mps
        )
        restart_allows_release = (not restart_context[frame]) or speed_confirms_release
        active_play_evidence = (
            has_live_team and ball_inside and restart_allows_release
        )

        if active_play_evidence:
            release_run += 1
        else:
            release_run = 0

        if release_run >= release_confirm_frames:
            # The confirming frame is the first frame emitted as live again.
            dead = False
            release_run = 0
            continue

        labels[frame] = "D"
        confidence[frame] = 1.0 if candidate[frame] else 0.75

    return LabelResult(labels=labels, confidence=confidence)
