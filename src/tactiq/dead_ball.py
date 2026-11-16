"""Conservative dead-ball evidence extraction.

This module produces evidence scores and candidates only.
It does NOT generate final possession labels.

Ball speed is retained as diagnostic information but does not
contribute to the dead-ball evidence score.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DeadBallEvidence:
    """Frame-level evidence related to possible dead-ball situations."""

    boundary_outside: np.ndarray
    boundary_persistent: np.ndarray
    restart_context: np.ndarray
    ball_stationary: np.ndarray
    ball_speed_available: np.ndarray
    score: np.ndarray
    candidate: np.ndarray
    level: np.ndarray


def _as_1d(name, values, n_frames):
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")

    if len(array) != n_frames:
        raise ValueError(
            f"{name} has {len(array)} frames; expected {n_frames}"
        )

    return array


def _mark_persistent_runs(mask, min_frames):
    """Mark every frame belonging to a True run of at least min_frames."""

    result = np.zeros(len(mask), dtype=bool)

    if len(mask) == 0:
        return result

    edges = np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)
    )

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    for start, end in zip(starts, ends):
        if end - start >= min_frames:
            result[start:end] = True

    return result


def extract_dead_ball_evidence(
    features,
    smart_tagging=None,
    dynamics=None,
    outside_margin_m=0.25,
    outside_persistence_frames=2,
    stationary_speed_mps=0.5,
    candidate_threshold=0.70,
):
    """Calculate conservative dead-ball evidence for every frame.

    Evidence weights:
      - Outside boundary beyond margin: +0.50
      - Persistent outside-boundary run: +0.25
      - Smart Tagging restart context: +0.10
      - Low reported ball speed: diagnostic only, +0.00

    A candidate is emitted when the score reaches candidate_threshold.
    These scores are heuristic evidence weights, not probabilities.

    Missing ball coordinates do not automatically indicate dead ball.
    Low speed does not contribute to the candidate score.
    """

    if not np.isfinite(outside_margin_m) or outside_margin_m < 0:
        raise ValueError("outside_margin_m must be finite and non-negative")

    if isinstance(outside_persistence_frames, bool) or not isinstance(
        outside_persistence_frames, (int, np.integer)
    ) or outside_persistence_frames < 1:
        raise ValueError("outside_persistence_frames must be at least 1")

    if not np.isfinite(stationary_speed_mps) or stationary_speed_mps < 0:
        raise ValueError("stationary_speed_mps must be finite and non-negative")

    if not np.isfinite(candidate_threshold) or not 0.35 <= candidate_threshold <= 1.0:
        raise ValueError("candidate_threshold must be between 0.35 and 1.0")

    visible = np.asarray(features.ball_visible)

    if visible.ndim != 1:
        raise ValueError("features.ball_visible must be one-dimensional")
    if not np.issubdtype(visible.dtype, np.bool_):
        raise ValueError("features.ball_visible must contain boolean values")

    n_frames = len(visible)
    visible = visible.astype(bool, copy=False)

    outside_reported = _as_1d(
        "features.ball_outside_boundary",
        features.ball_outside_boundary,
        n_frames,
    )
    if not np.issubdtype(outside_reported.dtype, np.bool_):
        raise ValueError(
            "features.ball_outside_boundary must contain boolean values"
        )
    outside_reported = outside_reported.astype(bool, copy=False)

    outside_distance = _as_1d(
        "features.ball_outside_distance",
        features.ball_outside_distance,
        n_frames,
    ).astype(np.float32, copy=False)

    boundary_outside = (
        visible
        & outside_reported
        & np.isfinite(outside_distance)
        & (outside_distance > outside_margin_m)
    )

    boundary_persistent = _mark_persistent_runs(
        boundary_outside,
        outside_persistence_frames,
    )

    # Restart annotations are broad context, not exact dead-ball labels.
    restart_context = np.zeros(n_frames, dtype=bool)

    if smart_tagging is not None:
        restart_active = np.asarray(smart_tagging.restart_active)

        if restart_active.ndim != 2:
            raise ValueError(
                "smart_tagging.restart_active must be two-dimensional"
            )

        if restart_active.shape[0] != n_frames:
            raise ValueError(
                "Smart Tagging and tracking frame counts do not match"
            )

        restart_context = restart_active.astype(
            bool, copy=False
        ).any(axis=1)

    # Retain ball speed only for diagnostics.
    # It does not contribute to score because its availability is sparse
    # and its relationship to coordinate-derived speed is unresolved.
    ball_speed_available = np.zeros(n_frames, dtype=bool)
    ball_stationary = np.zeros(n_frames, dtype=bool)

    if dynamics is not None:
        ball_speed = _as_1d(
            "dynamics.ball_speed",
            dynamics.ball_speed,
            n_frames,
        ).astype(np.float32, copy=False)

        ball_speed_available = (
            visible
            & np.isfinite(ball_speed)
            & (ball_speed >= 0)
        )

        ball_stationary = (
            ball_speed_available
            & (ball_speed <= stationary_speed_mps)
        )

    # Only boundary and restart evidence affect the score.
    score = np.zeros(n_frames, dtype=np.float32)
    score += boundary_outside.astype(np.float32) * 0.50
    score += boundary_persistent.astype(np.float32) * 0.25
    score += restart_context.astype(np.float32) * 0.10
    np.clip(score, 0.0, 1.0, out=score)

    candidate = score >= candidate_threshold

    level = np.full(n_frames, "none", dtype="<U8")
    level[(score > 0) & (score < 0.35)] = "weak"
    level[(score >= 0.35) & (score < candidate_threshold)] = "moderate"
    level[score >= candidate_threshold] = "strong"

    return DeadBallEvidence(
        boundary_outside=boundary_outside,
        boundary_persistent=boundary_persistent,
        restart_context=restart_context,
        ball_stationary=ball_stationary,
        ball_speed_available=ball_speed_available,
        score=score,
        candidate=candidate,
        level=level,
    )


def print_dead_ball_summary(result):
    """Print aggregate diagnostics without generating possession labels."""

    total = len(result.score)

    print("\nTACTIQ AI - DEAD BALL EVIDENCE REPORT")
    print("=" * 50)
    print(f"Total frames                 : {total:,}")
    print(f"Outside-boundary evidence   : {result.boundary_outside.sum():,}")
    print(f"Persistent boundary evidence: {result.boundary_persistent.sum():,}")
    print(f"Restart context             : {result.restart_context.sum():,}")
    print(f"Low-speed diagnostic frames : {result.ball_stationary.sum():,}")
    print(f"Dead-ball candidates        : {result.candidate.sum():,}")

    if total:
        print(f"Mean evidence score         : {result.score.mean():.3f}")

    print("\nEvidence levels:")
    for name in ("none", "weak", "moderate", "strong"):
        count = int(np.sum(result.level == name))
        print(f"  {name:<10}: {count:>9,}")

    print("\nLow-speed data is diagnostic only, not part of the score.")
    print("Candidates are evidence flags, not final D labels.")
    print("No possession labels were generated.")
