"""Temporal rule-based possession estimation."""

from dataclasses import dataclass

import numpy as np

from tactiq.data_loader import TrackingData
from tactiq.features import FeatureData, extract_features


@dataclass
class PossessionResult:
    labels: np.ndarray
    confidence: np.ndarray


def estimate_possession(
    data: TrackingData,
    features: FeatureData,
    control_radius=2.5,
    min_distance_advantage=1.0,
    switch_frames=8,
    missing_ball_hold_frames=25,
    no_evidence_hold_frames=25,
    no_candidate_hold_frames=25,
):
    """Estimate H/A/X possession labels using temporal continuity.

    H = home possession while the ball is alive
    A = away possession while the ball is alive
    X = abstain / insufficient evidence

    D is deliberately not inferred in this version.

    A visible ball outside the normalized pitch boundary is labeled X.
    This is an abstention, not a dead-ball decision.
    """

    if control_radius <= 0:
        raise ValueError("control_radius must be positive")
    if min_distance_advantage < 0:
        raise ValueError("min_distance_advantage cannot be negative")
    if switch_frames < 1:
        raise ValueError("switch_frames must be at least 1")
    if missing_ball_hold_frames < 0:
        raise ValueError("missing_ball_hold_frames cannot be negative")
    if no_evidence_hold_frames < 0:
        raise ValueError("no_evidence_hold_frames cannot be negative")
    if no_candidate_hold_frames < 0:
        raise ValueError("no_candidate_hold_frames cannot be negative")

    n_frames = len(features.ball_visible)

    if data.n_frames != n_frames:
        raise ValueError("Tracking and feature frame counts do not match")

    labels = np.full(n_frames, "X", dtype="<U1")
    confidence = np.zeros(n_frames, dtype=np.float32)

    current_team = None
    pending_team = None
    pending_count = 0

    missing_count = 0
    no_evidence_count = 0
    no_candidate_count = 0

    for frame in range(n_frames):

        # Case 1: Ball is not visible.
        if not features.ball_visible[frame]:
            missing_count += 1
            no_evidence_count = 0
            no_candidate_count = 0
            pending_team = None
            pending_count = 0

            if (
                current_team is not None
                and missing_count <= missing_ball_hold_frames
            ):
                labels[frame] = current_team
                confidence[frame] = 0.25
            else:
                labels[frame] = "X"
                confidence[frame] = 0.0

                if missing_count > missing_ball_hold_frames:
                    current_team = None

            continue

        missing_count = 0

        # Case 2: Ball coordinates are outside the pitch boundary.
        # Abstain rather than incorrectly declaring H, A, or D.
        if features.ball_outside_boundary[frame]:
            labels[frame] = "X"
            confidence[frame] = 0.0

            # A boundary event breaks continuity of the previous estimate.
            current_team = None
            pending_team = None
            pending_count = 0
            no_evidence_count = 0
            no_candidate_count = 0
            continue

        home_distance = features.home_nearest_distance[frame]
        away_distance = features.away_nearest_distance[frame]

        # Case 3: Ball is visible, but player-position evidence is absent.
        if not np.isfinite(home_distance) or not np.isfinite(away_distance):
            no_evidence_count += 1
            no_candidate_count = 0
            pending_team = None
            pending_count = 0

            if (
                current_team is not None
                and no_evidence_count <= no_evidence_hold_frames
            ):
                labels[frame] = current_team
                confidence[frame] = 0.20
            else:
                labels[frame] = "X"
                confidence[frame] = 0.0

                if no_evidence_count > no_evidence_hold_frames:
                    current_team = None

            continue

        no_evidence_count = 0

        candidate = None
        candidate_confidence = 0.0

        home_advantage = away_distance - home_distance
        away_advantage = home_distance - away_distance

        # A team must be close to the ball and clearly closer than its opponent.
        # Strict positive advantage avoids assigning ties to the home team
        # when min_distance_advantage is configured as zero.
        if (
            home_distance <= control_radius
            and home_advantage > 0
            and home_advantage >= min_distance_advantage
        ):
            candidate = "H"
            candidate_confidence = min(
                1.0,
                home_advantage / max(min_distance_advantage * 3, 1.0),
            )

        elif (
            away_distance <= control_radius
            and away_advantage > 0
            and away_advantage >= min_distance_advantage
        ):
            candidate = "A"
            candidate_confidence = min(
                1.0,
                away_advantage / max(min_distance_advantage * 3, 1.0),
            )

        # Case 4: No clear candidate.
        # This may represent a contested ball or uncertain control.
        # Retain the previous estimate, but do not invent a new team.
        if candidate is None:
            no_candidate_count += 1
            pending_team = None
            pending_count = 0

            if (
                current_team is not None
                and no_candidate_count <= no_candidate_hold_frames
            ):
                labels[frame] = current_team
                confidence[frame] = 0.25
            else:
                labels[frame] = "X"
                confidence[frame] = 0.0
                if no_candidate_count > no_candidate_hold_frames:
                    current_team = None

            continue

        # A valid candidate resets the no-candidate timer.
        no_candidate_count = 0

        # First clear candidate initializes the state.
        if current_team is None:
            current_team = candidate
            labels[frame] = current_team
            confidence[frame] = candidate_confidence
            continue

        # Same team remains in control.
        if candidate == current_team:
            pending_team = None
            pending_count = 0
            labels[frame] = current_team
            confidence[frame] = candidate_confidence
            continue

        # Opposing candidate must persist before switching.
        if candidate == pending_team:
            pending_count += 1
        else:
            pending_team = candidate
            pending_count = 1

        if pending_count >= switch_frames:
            current_team = candidate
            pending_team = None
            pending_count = 0

        labels[frame] = current_team
        confidence[frame] = (
            candidate_confidence if current_team == candidate else 0.2
        )

    return PossessionResult(
        labels=labels,
        confidence=confidence,
    )


def print_possession_summary(result):
    """Print label counts and confidence diagnostics."""

    total = len(result.labels)

    print("\nSTRATIQ - POSSESSION ESTIMATION REPORT")
    print("=" * 48)
    print(f"Total frames : {total:,}")

    for label, name in (
        ("H", "Home"),
        ("A", "Away"),
        ("D", "Dead ball"),
        ("X", "Abstain"),
    ):
        count = int(np.sum(result.labels == label))
        print(f"{name:15}: {count:>9,} ({count / total * 100:6.2f}%)")

    print(
        "\nMean confidence: "
        f"{result.confidence.mean():.3f}"
    )
    print(
        "Estimated labels are heuristic outputs, "
        "not verified ground truth."
    )


def main():
    import argparse

    from tactiq.data_loader import load_tracking

    parser = argparse.ArgumentParser(
        description="Estimate football possession from tracking data"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument("--switch-frames", type=int, default=8)
    parser.add_argument("--missing-ball-hold-frames", type=int, default=25)
    parser.add_argument("--no-evidence-hold-frames", type=int, default=25)
    parser.add_argument("--no-candidate-hold-frames", type=int, default=25)
    args = parser.parse_args()

    data = load_tracking(args.data_dir)
    features = extract_features(data)

    result = estimate_possession(
        data,
        features,
        control_radius=args.control_radius,
        min_distance_advantage=args.distance_advantage,
        switch_frames=args.switch_frames,
        missing_ball_hold_frames=args.missing_ball_hold_frames,
        no_evidence_hold_frames=args.no_evidence_hold_frames,
        no_candidate_hold_frames=args.no_candidate_hold_frames,
    )

    print_possession_summary(result)


if __name__ == "__main__":
    main()
