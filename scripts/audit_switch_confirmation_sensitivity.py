"""Compare possession switch-confirmation durations.

Diagnostic only: no ground truth, submission CSV, or production edits.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


def transitions(labels):
    return int(np.sum(labels[1:] != labels[:-1]))


def switch_transitions(labels, old, new):
    return int(np.sum((labels[:-1] == old) & (labels[1:] == new)))


def spell_lengths(labels, target):
    mask = labels == target
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return (ends - starts) / 25.0


def print_spells(labels):
    for label, name in (
        ("H", "Barcelona"),
        ("A", "Athletic Club"),
        ("X", "Abstain"),
    ):
        values = spell_lengths(labels, label)

        if len(values) == 0:
            print(f"{name:15}: no runs")
            continue

        print(
            f"{name:15}: runs={len(values):>5,} | "
            f"median={np.median(values):>6.2f}s | "
            f"P90={np.percentile(values, 90):>6.2f}s | "
            f"max={np.max(values):>7.2f}s"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument(
        "--switch-frames",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16],
    )
    parser.add_argument("--missing-ball-hold-frames", type=int, default=25)
    parser.add_argument("--no-evidence-hold-frames", type=int, default=25)
    parser.add_argument("--no-candidate-hold-frames", type=int, default=25)
    args = parser.parse_args()

    if args.control_radius <= 0:
        parser.error("control-radius must be positive")
    if args.distance_advantage < 0:
        parser.error("distance-advantage cannot be negative")
    if any(value < 1 for value in args.switch_frames):
        parser.error("Every switch-frames value must be >= 1")
    if 8 not in args.switch_frames:
        parser.error("Include 8 frames as the reference scenario")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)

    print("\nSWITCH CONFIRMATION SENSITIVITY AUDIT")
    print("=" * 90)
    print(f"Total frames             : {data.n_frames:,}")
    print(f"Control radius           : {args.control_radius:.2f} m")
    print(f"Distance advantage       : {args.distance_advantage:.2f} m")
    print(f"Missing-ball hold        : {args.missing_ball_hold_frames} frames")
    print(f"No-evidence hold         : {args.no_evidence_hold_frames} frames")
    print(f"No-candidate hold        : {args.no_candidate_hold_frames} frames")

    results = {}

    for value in args.switch_frames:
        print(f"\nRunning switch confirmation: {value} frames...")
        results[value] = estimate_possession(
            data,
            features,
            control_radius=args.control_radius,
            min_distance_advantage=args.distance_advantage,
            switch_frames=value,
            missing_ball_hold_frames=args.missing_ball_hold_frames,
            no_evidence_hold_frames=args.no_evidence_hold_frames,
            no_candidate_hold_frames=args.no_candidate_hold_frames,
        )

    reference = results[8].labels

    print("\nSCENARIO COMPARISON")
    print("=" * 120)
    print(
        f"{'Frames':>7} {'Seconds':>8} {'H':>9} {'A':>9} {'X':>9} "
        f"{'Transitions':>12} {'H->A':>8} {'A->H':>8} {'Changed':>10}"
    )

    for value in args.switch_frames:
        labels = results[value].labels
        seconds = value / 25.0

        print(
            f"{value:>7} {seconds:>8.2f} "
            f"{np.sum(labels == 'H'):>9,} "
            f"{np.sum(labels == 'A'):>9,} "
            f"{np.sum(labels == 'X'):>9,} "
            f"{transitions(labels):>12,} "
            f"{switch_transitions(labels, 'H', 'A'):>8,} "
            f"{switch_transitions(labels, 'A', 'H'):>8,} "
            f"{np.sum(labels != reference):>10,}"
        )

    for value in args.switch_frames:
        labels = results[value].labels
        changed = labels != reference

        print("\n" + "=" * 90)
        print(f"DETAIL — {value} FRAMES ({value / 25:.2f} SECONDS)")
        print("=" * 90)

        for label, name in (
            ("H", "Barcelona"),
            ("A", "Athletic Club"),
            ("X", "Abstain"),
        ):
            count = int(np.sum(labels == label))
            print(f"{name:15}: {count:>9,} ({count / len(labels) * 100:6.2f}%)")

        print(f"\nChanged frames vs 8-frame reference: {int(changed.sum()):,}")
        print(f"Reference H -> New A: {int(np.sum((reference == 'H') & (labels == 'A'))):,}")
        print(f"Reference A -> New H: {int(np.sum((reference == 'A') & (labels == 'H'))):,}")
        print(f"Reference H/A -> New X: {int(np.sum(np.isin(reference, ['H', 'A']) & (labels == 'X'))):,}")
        print(f"Reference X -> New H/A: {int(np.sum((reference == 'X') & np.isin(labels, ['H', 'A']))):,}")

        print("\nSPELL DIAGNOSTICS")
        print("-" * 90)
        print_spells(labels)

    print("\nIMPORTANT")
    print("- All scenarios use the same tracking features and distance thresholds.")
    print("- Only switch_frames changes.")
    print("- No ground truth or submission CSV was loaded or created.")
    print("- Production source code was not modified.")
    print("- Counts and spell lengths are diagnostics, not accuracy scores.")


if __name__ == "__main__":
    main()
