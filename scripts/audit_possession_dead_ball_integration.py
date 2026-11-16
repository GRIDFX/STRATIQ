"""Shadow-mode audit of dead-ball evidence over possession predictions.

No ground truth is loaded or used.
No submission file is written.
The original possession estimator is not modified.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession
from tactiq.smart_tagging_features import extract_smart_tagging_features
from tactiq.tracking_dynamics import extract_tracking_dynamics
from tactiq.dead_ball import extract_dead_ball_evidence


LABELS = ("H", "A", "D", "X")


def print_distribution(title, labels):
    total = len(labels)

    print(f"\n{title}")
    print("-" * 65)

    for label in LABELS:
        count = int(np.sum(labels == label))
        percentage = count / max(total, 1) * 100

        print(
            f"{label}: {count:>9,} "
            f"({percentage:6.2f}%)"
        )


def count_runs(labels):
    if len(labels) == 0:
        return 0

    return int(1 + np.sum(labels[1:] != labels[:-1]))


def main():
    parser = argparse.ArgumentParser(
        description="Audit shadow integration of dead-ball candidates"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")

    print("Loading tracking data...")
    data = load_tracking(args.data_dir)

    print("Extracting tracking features...")
    features = extract_features(data)

    print("Estimating baseline possession...")
    baseline = estimate_possession(data, features)

    print("Extracting Smart Tagging context...")
    smart = extract_smart_tagging_features(
        args.xml,
        n_frames=data.n_frames,
        fps=args.fps,
    )

    print("Extracting dead-ball evidence...")
    dynamics = extract_tracking_dynamics(data)

    dead_ball = extract_dead_ball_evidence(
        features,
        smart_tagging=smart,
        dynamics=dynamics,
    )

    baseline_labels = baseline.labels.copy()
    shadow_labels = baseline_labels.copy()

    candidate = dead_ball.candidate

    # This is a simulation only. The production estimator is unchanged.
    prior_labels = baseline_labels[candidate].copy()
    shadow_labels[candidate] = "D"

    print("\nBASELINE POSSESSION DISTRIBUTION")
    print("=" * 65)
    print_distribution("Before shadow integration", baseline_labels)

    print("\nSHADOW POSSESSION DISTRIBUTION")
    print("=" * 65)
    print_distribution("After replacing candidates with D", shadow_labels)

    print("\nCANDIDATE FRAME OVERWRITE")
    print("=" * 65)
    print(f"Candidate frames : {int(candidate.sum()):,}")

    for label in LABELS:
        count = int(np.sum(prior_labels == label))
        print(f"Previously {label} : {count:>9,}")

    changed = int(np.sum(baseline_labels != shadow_labels))
    print(f"Changed labels   : {changed:,}")

    print("\nSEQUENCE DIAGNOSTICS")
    print("=" * 65)
    print(f"Baseline label runs : {count_runs(baseline_labels):,}")
    print(f"Shadow label runs   : {count_runs(shadow_labels):,}")
    print(
        "Baseline transitions: "
        f"{np.sum(baseline_labels[1:] != baseline_labels[:-1]):,}"
    )
    print(
        "Shadow transitions  : "
        f"{np.sum(shadow_labels[1:] != shadow_labels[:-1]):,}"
    )

    print("\nCANDIDATE SEGMENTS")
    print("=" * 65)

    edges = np.diff(
        np.concatenate(([False], candidate, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    durations = (ends - starts) / args.fps

    print(f"Segments                 : {len(starts):,}")
    print(f"Total candidate duration : {durations.sum():.2f} seconds")

    if len(durations):
        print(f"Median duration          : {np.median(durations):.2f} seconds")
        print(f"Maximum duration         : {np.max(durations):.2f} seconds")

    print("\nIMPORTANT")
    print("-" * 65)
    print("This is a shadow simulation, not a verified D classification.")
    print("No ground truth was loaded or used.")
    print("No submission CSV was written.")
    print("The production possession estimator was not modified.")


if __name__ == "__main__":
    main()
