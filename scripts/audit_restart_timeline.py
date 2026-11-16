"""Audit tracking behavior around individual Smart Tagging restart events."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.smart_tagging_features import _read_events
from tactiq.tracking_dynamics import extract_tracking_dynamics


def frame_stats(start, end, features, ball_speed):
    start = max(0, start)
    end = min(len(features.ball_visible), end)

    if end <= start:
        return None

    visible = features.ball_visible[start:end]
    outside = features.ball_outside_boundary[start:end]
    speeds = ball_speed[start:end]
    valid_speeds = speeds[np.isfinite(speeds)]

    return {
        "frames": end - start,
        "visible": float(np.mean(visible)),
        "outside": float(np.mean(outside)),
        "speed_count": len(valid_speeds),
        "speed_median": (
            float(np.median(valid_speeds))
            if len(valid_speeds)
            else np.nan
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="../more-than-a-hack-2027/data",
    )
    parser.add_argument(
        "--xml",
        default=(
            "../more-than-a-hack-2027/data/tactical/"
            "BAR-ATH_tactical_smart_tagging.xml"
        ),
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    args = parser.parse_args()

    if args.fps <= 0 or args.window_seconds <= 0:
        raise ValueError("FPS and window seconds must be positive.")

    print("Loading tracking data...")
    data = load_tracking(args.data_dir)

    print("Extracting features and speed...")
    features = extract_features(data)
    dynamics = extract_tracking_dynamics(data)

    print("Reading restart events...")
    events = _read_events(Path(args.xml))

    window = int(round(args.window_seconds * args.fps))
    groups = defaultdict(list)
    total_restart_events = 0
    skipped = 0

    for code, start_s, end_s, tags in events:
        if code != "SET PIECES":
            continue

        total_restart_events += 1
        restart_type = tags.get("Type", "(unknown)")

        start = int(round(start_s * args.fps))
        end = int(round(end_s * args.fps))

        if end <= start:
            skipped += 1
            continue

        before = frame_stats(
            start - window, start, features, dynamics.ball_speed
        )
        during = frame_stats(
            start, end, features, dynamics.ball_speed
        )
        after = frame_stats(
            end, end + window, features, dynamics.ball_speed
        )

        if before is None or during is None or after is None:
            skipped += 1
            continue

        groups[restart_type].append({
            "duration": end - start,
            "before": before,
            "during": during,
            "after": after,
        })

    print("\n" + "=" * 75)
    print("TACTIQ AI - RESTART EVENT TIMELINE AUDIT")
    print("=" * 75)
    print(f"Total SET PIECES events : {total_restart_events}")
    print(f"Analyzed events         : {sum(map(len, groups.values()))}")
    print(f"Skipped events          : {skipped}")
    print(f"Comparison window       : {args.window_seconds:.1f}s before/after")

    for restart_type, records in sorted(groups.items()):
        print(f"\n{restart_type} ({len(records)} events)")
        print("-" * 75)

        durations = np.array([r["duration"] for r in records]) / args.fps

        print(f"Median event duration   : {np.median(durations):.2f}s")
        print(
            f"Duration range          : "
            f"{np.min(durations):.2f}–{np.max(durations):.2f}s"
        )

        for period in ("before", "during", "after"):
            visible = np.mean([r[period]["visible"] for r in records])
            outside = np.mean([r[period]["outside"] for r in records])

            speed_values = [
                r[period]["speed_median"]
                for r in records
                if np.isfinite(r[period]["speed_median"])
            ]

            median_speed = (
                np.median(speed_values)
                if speed_values
                else np.nan
            )

            print(
                f"{period.capitalize():<10} "
                f"ball visible={visible:.1%} | "
                f"outside={outside:.1%} | "
                f"median event speed={median_speed:.3f} m/s"
                if np.isfinite(median_speed)
                else
                f"{period.capitalize():<10} "
                f"ball visible={visible:.1%} | "
                f"outside={outside:.1%} | "
                "median event speed=unavailable"
            )

    print("\nLIMITATIONS")
    print("-" * 75)
    print("Events are analyzed individually; overlapping events may be counted twice.")
    print("Before/during/after are temporal comparisons, not ground-truth labels.")
    print("A restart event does not mean every frame in its interval is dead ball.")
    print("No possession labels or estimator parameters were changed.")


if __name__ == "__main__":
    main()
