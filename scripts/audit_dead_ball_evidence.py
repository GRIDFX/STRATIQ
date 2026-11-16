"""Audit dead-ball evidence on a complete tracking match.

This is a diagnostics script only. It does not use ground truth,
does not alter tracking data, and does not generate submission labels.
"""

import argparse
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.smart_tagging_features import extract_smart_tagging_features
from tactiq.tracking_dynamics import extract_tracking_dynamics
from tactiq.dead_ball import (
    extract_dead_ball_evidence,
    print_dead_ball_summary,
)


def contiguous_runs(mask):
    """Return [start, end) intervals for consecutive True values."""
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")

    edges = np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)
    )

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    return list(zip(starts.tolist(), ends.tolist()))


def main():
    parser = argparse.ArgumentParser(
        description="Audit dead-ball evidence on tracking data"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the competition data directory",
    )
    parser.add_argument(
        "--xml",
        default=None,
        help="Optional Smart Tagging XML path",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--outside-margin-m", type=float, default=0.25)
    parser.add_argument(
        "--outside-persistence-frames",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--stationary-speed-mps",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.70,
    )
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    if args.xml and not Path(args.xml).is_file():
        parser.error(f"Smart Tagging XML not found: {args.xml}")

    print("Loading tracking data...")
    data = load_tracking(args.data_dir)

    print("Extracting tracking features...")
    features = extract_features(data)

    print("Extracting reported tracking speeds...")
    dynamics = extract_tracking_dynamics(data)

    smart_tagging = None
    if args.xml:
        print("Extracting Smart Tagging context...")
        smart_tagging = extract_smart_tagging_features(
            args.xml,
            n_frames=data.n_frames,
            fps=args.fps,
        )

    print("Calculating dead-ball evidence...")
    result = extract_dead_ball_evidence(
        features,
        smart_tagging=smart_tagging,
        dynamics=dynamics,
        outside_margin_m=args.outside_margin_m,
        outside_persistence_frames=args.outside_persistence_frames,
        stationary_speed_mps=args.stationary_speed_mps,
        candidate_threshold=args.candidate_threshold,
    )

    print_dead_ball_summary(result)

    runs = contiguous_runs(result.candidate)
    durations = np.asarray(
        [(end - start) / args.fps for start, end in runs],
        dtype=np.float64,
    )

    print("\nCANDIDATE SEGMENT REPORT")
    print("=" * 50)
    print(f"Candidate segments          : {len(runs):,}")

    if len(runs):
        print(f"Shortest segment            : {durations.min():.3f} s")
        print(f"Median segment              : {np.median(durations):.3f} s")
        print(f"Longest segment             : {durations.max():.3f} s")

        print("\nLongest candidate segments [start, end):")
        longest_indices = np.argsort(durations)[::-1][:10]

        for index in longest_indices:
            start, end = runs[int(index)]
            print(
                f"  frames {start:>7,}-{end:>7,} "
                f"| {durations[index]:>8.3f} s"
            )
    else:
        print("No candidate segments were found.")

    candidate_count = int(result.candidate.sum())

    if candidate_count:
        print("\nCANDIDATE EVIDENCE OVERLAP")
        print("=" * 50)
        print(
            "With persistent boundary evidence: "
            f"{np.sum(result.candidate & result.boundary_persistent):,}"
        )
        print(
            "With restart context              : "
            f"{np.sum(result.candidate & result.restart_context):,}"
        )
        print(
            "With low reported ball speed      : "
            f"{np.sum(result.candidate & result.ball_stationary):,}"
        )

    print("\nAudit completed. No possession labels were generated.")


if __name__ == "__main__":
    main()
