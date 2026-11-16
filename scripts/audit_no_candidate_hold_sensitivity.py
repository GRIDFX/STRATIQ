"""Project label-retention sensitivity during no-candidate intervals.

This is a limited shadow projection, not a full estimator replay.
It does not use ground truth or modify production labels.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


def contiguous_runs(mask):
    mask = np.asarray(mask, dtype=bool)

    edges = np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    return list(zip(starts.tolist(), ends.tolist()))


def build_no_candidate_mask(features, control_radius, advantage):
    home = features.home_nearest_distance
    away = features.away_nearest_distance

    valid = (
        features.ball_visible
        & ~features.ball_outside_boundary
        & np.isfinite(home)
        & np.isfinite(away)
    )

    home_candidate = (
        valid
        & (home <= control_radius)
        & ((away - home) > 0)
        & ((away - home) >= advantage)
    )

    away_candidate = (
        valid
        & (away <= control_radius)
        & ((home - away) > 0)
        & ((home - away) >= advantage)
    )

    return valid & ~home_candidate & ~away_candidate


def project_hold(labels, runs, hold_frames):
    """Extend the initial H/A label within each eligible run only."""

    projected = labels.copy()
    added = {"H": 0, "A": 0}
    eligible_runs = 0
    uninitialized_runs = 0

    for start, end in runs:
        initial_team = labels[start]

        if initial_team not in ("H", "A"):
            uninitialized_runs += 1
            continue

        eligible_runs += 1

        stop = (
            end
            if hold_frames is None
            else min(end, start + hold_frames)
        )

        target = projected[start:stop]
        replace = target == "X"
        added[initial_team] += int(replace.sum())
        target[replace] = initial_team

    return projected, added, eligible_runs, uninitialized_runs


def transitions(labels):
    return int(np.sum(labels[1:] != labels[:-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    args = parser.parse_args()

    if (
        args.fps <= 0
        or args.control_radius <= 0
        or args.distance_advantage < 0
    ):
        parser.error("Invalid FPS or possession parameters")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)

    print("Estimating baseline...")
    result = estimate_possession(
        data,
        features,
        control_radius=args.control_radius,
        min_distance_advantage=args.distance_advantage,
    )

    labels = result.labels

    no_candidate = build_no_candidate_mask(
        features,
        args.control_radius,
        args.distance_advantage,
    )
    runs = contiguous_runs(no_candidate)

    print("\nHOLD SENSITIVITY PROJECTION")
    print("=" * 105)
    print(f"Total frames             : {len(labels):,}")
    print(f"No-candidate frames      : {int(no_candidate.sum()):,}")
    print(f"No-candidate runs        : {len(runs):,}")
    print(f"Baseline H/A/X/D         : "
          f"{np.sum(labels == 'H'):,} / "
          f"{np.sum(labels == 'A'):,} / "
          f"{np.sum(labels == 'X'):,} / "
          f"{np.sum(labels == 'D'):,}")

    scenarios = [
        ("1 second (25 frames)", 25),
        ("2 seconds (50 frames)", 50),
        ("3 seconds (75 frames)", 75),
        ("5 seconds (125 frames)", 125),
        ("Unlimited within run", None),
    ]

    print("\nSCENARIO COMPARISON")
    print("-" * 105)
    print(
        f"{'Scenario':<25} {'Added H':>10} {'Added A':>10} "
        f"{'Total H':>10} {'Total A':>10} {'Total X':>10} "
        f"{'Transitions':>13}"
    )

    for name, hold_frames in scenarios:
        projected, added, eligible, uninitialized = project_hold(
            labels,
            runs,
            hold_frames,
        )

        print(
            f"{name:<25} "
            f"{added['H']:>10,} "
            f"{added['A']:>10,} "
            f"{np.sum(projected == 'H'):>10,} "
            f"{np.sum(projected == 'A'):>10,} "
            f"{np.sum(projected == 'X'):>10,} "
            f"{transitions(projected):>13,}"
        )

        print(
            f"  Eligible runs: {eligible:,}; "
            f"uninitialized runs: {uninitialized:,}"
        )

    print("\nLIMITATIONS")
    print("-" * 105)
    print("This projection only changes labels inside no-candidate runs.")
    print("It does not replay temporal state or opposing-team switch confirmation.")
    print("It is not a final estimator result or an accuracy measurement.")
    print("No ground truth was loaded.")
    print("No production labels or source modules were modified.")


if __name__ == "__main__":
    main()
