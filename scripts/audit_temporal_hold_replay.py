"""Full estimator replay for different no-candidate hold durations.

Uses no ground truth. Does not modify production estimator or write CSV.
All scenarios replay the actual temporal possession state machine.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


SCENARIOS = [
    ("1 second", 25),
    ("2 seconds", 50),
    ("3 seconds", 75),
    ("5 seconds", 125),
    ("10 seconds", 250),
]


def transitions(labels):
    return int(np.sum(labels[1:] != labels[:-1]))


def run_count(labels, label):
    mask = labels == label
    if not mask.any():
        return 0

    edges = np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)
    )
    return int(np.sum(edges == 1))


def summarize(name, frames, result, baseline=None):
    labels = result.labels
    total = len(labels)

    h = int(np.sum(labels == "H"))
    a = int(np.sum(labels == "A"))
    d = int(np.sum(labels == "D"))
    x = int(np.sum(labels == "X"))

    print(f"\n{name}")
    print("-" * 105)
    print(f"H : {h:>9,} ({h / total * 100:6.2f}%)")
    print(f"A : {a:>9,} ({a / total * 100:6.2f}%)")
    print(f"D : {d:>9,} ({d / total * 100:6.2f}%)")
    print(f"X : {x:>9,} ({x / total * 100:6.2f}%)")
    print(f"Transitions         : {transitions(labels):,}")
    print(f"H runs              : {run_count(labels, 'H'):,}")
    print(f"A runs              : {run_count(labels, 'A'):,}")
    print(f"X runs              : {run_count(labels, 'X'):,}")
    print(f"Mean confidence     : {result.confidence.mean():.4f}")

    if baseline is not None:
        changed = baseline != labels
        changed_count = int(changed.sum())

        print(f"Changed vs baseline : {changed_count:,}")
        print(
            f"Baseline X -> H    : "
            f"{np.sum((baseline == 'X') & (labels == 'H')):,}"
        )
        print(
            f"Baseline X -> A    : "
            f"{np.sum((baseline == 'X') & (labels == 'A')):,}"
        )
        print(
            f"Baseline H/A -> X  : "
            f"{np.sum(np.isin(baseline, ['H', 'A']) & (labels == 'X')):,}"
        )
        print(
            f"Baseline H -> A    : "
            f"{np.sum((baseline == 'H') & (labels == 'A')):,}"
        )
        print(
            f"Baseline A -> H    : "
            f"{np.sum((baseline == 'A') & (labels == 'H')):,}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument("--switch-frames", type=int, default=8)
    parser.add_argument("--missing-ball-hold-frames", type=int, default=25)
    parser.add_argument("--no-evidence-hold-frames", type=int, default=25)
    args = parser.parse_args()

    if args.control_radius <= 0:
        parser.error("control-radius must be positive")
    if args.distance_advantage < 0:
        parser.error("distance-advantage cannot be negative")
    if args.switch_frames < 1:
        parser.error("switch-frames must be at least 1")

    print("Loading tracking data...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)

    print("\nFULL TEMPORAL HOLD REPLAY")
    print("=" * 105)
    print(f"Total frames          : {data.n_frames:,}")
    print(f"Control radius        : {args.control_radius:.2f} m")
    print(f"Distance advantage    : {args.distance_advantage:.2f} m")
    print(f"Switch confirmation   : {args.switch_frames} frames")
    print("Missing-ball hold     : unchanged")
    print("No-evidence hold      : unchanged")

    results = {}

    for name, hold_frames in SCENARIOS:
        print(f"\nRunning {name} ({hold_frames} frames)...")

        results[hold_frames] = estimate_possession(
            data,
            features,
            control_radius=args.control_radius,
            min_distance_advantage=args.distance_advantage,
            switch_frames=args.switch_frames,
            missing_ball_hold_frames=args.missing_ball_hold_frames,
            no_evidence_hold_frames=args.no_evidence_hold_frames,
            no_candidate_hold_frames=hold_frames,
        )

    baseline = results[25].labels

    summarize("BASELINE — 1 SECOND", 25, results[25])

    print("\nSCENARIO COMPARISON")
    print("=" * 105)
    print(
        f"{'Hold':<14} {'H':>9} {'A':>9} {'X':>9} "
        f"{'Transitions':>13} {'Changed':>10} "
        f"{'X->H':>9} {'X->A':>9}"
    )

    for name, hold_frames in SCENARIOS:
        result = results[hold_frames]
        labels = result.labels

        print(
            f"{name:<14} "
            f"{np.sum(labels == 'H'):>9,} "
            f"{np.sum(labels == 'A'):>9,} "
            f"{np.sum(labels == 'X'):>9,} "
            f"{transitions(labels):>13,} "
            f"{np.sum(labels != baseline):>10,} "
            f"{np.sum((baseline == 'X') & (labels == 'H')):>9,} "
            f"{np.sum((baseline == 'X') & (labels == 'A')):>9,}"
        )

    for name, hold_frames in SCENARIOS[1:]:
        summarize(
            f"DETAIL — {name.upper()}",
            hold_frames,
            results[hold_frames],
            baseline=baseline,
        )

    print("\nIMPORTANT")
    print("- Every scenario reruns the actual possession state machine.")
    print("- No ground truth was loaded or used.")
    print("- This is sensitivity analysis, not accuracy evaluation.")
    print("- No source estimator parameters or submission CSV were changed.")


if __name__ == "__main__":
    main()
