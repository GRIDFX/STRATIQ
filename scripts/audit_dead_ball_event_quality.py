"""Classify dead-ball candidate segments by boundary-transition evidence.

This script is diagnostic only. It does not create possession labels
or use ground truth.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.smart_tagging_features import extract_smart_tagging_features
from tactiq.tracking_dynamics import extract_tracking_dynamics
from tactiq.dead_ball import extract_dead_ball_evidence


def contiguous_runs(mask):
    edges = np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")

    print("Loading tracking data...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)

    smart = extract_smart_tagging_features(
        args.xml,
        n_frames=data.n_frames,
        fps=args.fps,
    )

    dynamics = extract_tracking_dynamics(data)

    evidence = extract_dead_ball_evidence(
        features,
        smart_tagging=smart,
        dynamics=dynamics,
    )

    runs = contiguous_runs(evidence.candidate)

    groups = {
        "bracketed": [],
        "exit_only": [],
        "entry_only": [],
        "unbracketed": [],
    }

    records = []

    for start, end in runs:
        before_inside = (
            start > 0
            and features.ball_inside_boundary[start - 1]
        )

        after_inside = (
            end < data.n_frames
            and features.ball_inside_boundary[end]
        )

        if before_inside and after_inside:
            category = "bracketed"
        elif before_inside:
            category = "exit_only"
        elif after_inside:
            category = "entry_only"
        else:
            category = "unbracketed"

        restart_frames = int(
            smart.restart_active[start:end].any(axis=1).sum()
        )

        segment_length = end - start
        outside = features.ball_outside_distance[start:end]
        outside = outside[np.isfinite(outside)]

        record = {
            "start": start,
            "end": end,
            "category": category,
            "duration": segment_length / args.fps,
            "restart": restart_frames > 0,
            "restart_pct": 100 * restart_frames / segment_length,
            "outside_median": (
                float(np.median(outside))
                if outside.size else np.nan
            ),
        }

        groups[category].append(record)
        records.append(record)

    print("\nBOUNDARY TRANSITION CLASSIFICATION")
    print("=" * 75)
    print(
        f"{'Category':<18} {'Segments':>10} {'Frames':>12} "
        f"{'Duration(s)':>14} {'With restart':>14}"
    )

    for category, items in groups.items():
        frames = sum(item["end"] - item["start"] for item in items)
        duration = sum(item["duration"] for item in items)
        restart = sum(item["restart"] for item in items)

        print(
            f"{category:<18} {len(items):>10,} "
            f"{frames:>12,} {duration:>14.2f} "
            f"{restart:>14,}"
        )

    print("\nRESTART OVERLAP BY BOUNDARY CLASS")
    print("=" * 75)

    for category, items in groups.items():
        with_restart = [item for item in items if item["restart"]]
        without_restart = len(items) - len(with_restart)

        print(
            f"{category:<18}: "
            f"with restart={len(with_restart):>3}, "
            f"without restart={without_restart:>3}"
        )

    print("\nLONGEST SEGMENTS PER CATEGORY")
    print("=" * 100)

    for category, items in groups.items():
        print(f"\n{category.upper()}")
        print("-" * 100)
        print(
            f"{'Frames':<23} {'Duration':>10} "
            f"{'Restart':>10} {'Restart %':>11} "
            f"{'Outside median':>16}"
        )

        for item in sorted(
            items,
            key=lambda value: value["duration"],
            reverse=True,
        )[:5]:
            print(
                f"{item['start']}-{item['end']:<17} "
                f"{item['duration']:>9.2f}s "
                f"{str(item['restart']):>10} "
                f"{item['restart_pct']:>10.1f}% "
                f"{item['outside_median']:>15.2f}m"
            )

    print("\nNo ground truth was used.")
    print("No possession labels or tracking values were modified.")


if __name__ == "__main__":
    main()
