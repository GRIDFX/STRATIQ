"""Descriptive temporal audit of tracking behavior after restart annotations.

This script does not create possession or dead-ball labels.
Restart annotation start is only a temporal reference, not the exact
moment when play resumes.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.smart_tagging_features import _read_events
from tactiq.tracking_dynamics import extract_tracking_dynamics


WINDOWS = (
    ("0-1s", 0.0, 1.0),
    ("1-2s", 1.0, 2.0),
    ("2-3s", 2.0, 3.0),
    ("3-5s", 3.0, 5.0),
    ("5-10s", 5.0, 10.0),
)


def calculate_window(start, end, features, ball_speed):
    """Calculate descriptive statistics for one frame interval."""
    n_frames = len(features.ball_visible)

    start = max(0, start)
    end = min(n_frames, end)

    if end <= start:
        return None

    visible = features.ball_visible[start:end]
    outside = features.ball_outside_boundary[start:end]
    speeds = ball_speed[start:end]
    valid_speeds = speeds[np.isfinite(speeds)]

    return {
        "frames": end - start,
        "visible_rate": float(np.mean(visible)),
        "outside_rate": float(np.mean(outside)),
        "speed_count": int(len(valid_speeds)),
        "speed_median": (
            float(np.median(valid_speeds))
            if len(valid_speeds)
            else np.nan
        ),
    }


def summarize_events(records, window_name):
    """Summarize event-level measurements, avoiding frame pooling."""
    selected = [
        record[window_name]
        for record in records
        if record.get(window_name) is not None
    ]

    if not selected:
        return None

    visible = np.array([item["visible_rate"] for item in selected])
    outside = np.array([item["outside_rate"] for item in selected])

    event_speeds = np.array([
        item["speed_median"]
        for item in selected
        if np.isfinite(item["speed_median"])
    ])

    return {
        "events": len(selected),
        "frames": sum(item["frames"] for item in selected),
        "visible": float(np.mean(visible)),
        "outside": float(np.mean(outside)),
        "speed_events": len(event_speeds),
        "speed_median": (
            float(np.median(event_speeds))
            if len(event_speeds)
            else np.nan
        ),
    }


def print_summary(title, records):
    print(f"\n{title}")
    print("=" * 105)
    print(
        f"{'Window':<10}"
        f"{'Events':>9}"
        f"{'Frames':>12}"
        f"{'Visible':>12}"
        f"{'Outside':>12}"
        f"{'Speed events':>15}"
        f"{'Median speed':>17}"
    )
    print("-" * 105)

    for window_name, _, _ in WINDOWS:
        result = summarize_events(records, window_name)

        if result is None:
            continue

        speed_text = (
            f"{result['speed_median']:.3f}"
            if np.isfinite(result["speed_median"])
            else "N/A"
        )

        print(
            f"{window_name:<10}"
            f"{result['events']:>9,}"
            f"{result['frames']:>12,}"
            f"{result['visible']:>11.1%}"
            f"{result['outside']:>11.1%}"
            f"{result['speed_events']:>15,}"
            f"{speed_text:>17}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Audit tracking behavior after restart annotation starts."
    )
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
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("FPS must be positive.")

    data_dir = Path(args.data_dir)
    xml_path = Path(args.xml)

    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    print("Loading tracking data...")
    data = load_tracking(data_dir)

    print("Extracting tracking features...")
    features = extract_features(data)

    print("Extracting reported ball speed...")
    dynamics = extract_tracking_dynamics(data)

    print("Reading restart annotations...")
    events = _read_events(xml_path)

    grouped = defaultdict(list)
    skipped = 0
    total = 0

    for code, start_s, end_s, tags in events:
        if code != "SET PIECES":
            continue

        total += 1
        restart_type = tags.get("Type", "(unknown)")
        start_frame = int(round(start_s * args.fps))

        record = {
            "start": start_s,
            "end": end_s,
            "type": restart_type,
        }

        valid_windows = 0

        for window_name, offset_start, offset_end in WINDOWS:
            frame_start = start_frame + int(round(offset_start * args.fps))
            frame_end = start_frame + int(round(offset_end * args.fps))

            result = calculate_window(
                frame_start,
                frame_end,
                features,
                dynamics.ball_speed,
            )

            record[window_name] = result

            if result is not None:
                valid_windows += 1

        if valid_windows == 0:
            skipped += 1
            continue

        grouped[restart_type].append(record)

    all_records = [
        record
        for records in grouped.values()
        for record in records
    ]

    print("\n" + "=" * 105)
    print("TACTIQ AI - RESTART TEMPORAL AUDIT")
    print("=" * 105)
    print(f"Total SET PIECES events : {total}")
    print(f"Analyzed events         : {len(all_records)}")
    print(f"Skipped events          : {skipped}")
    print(f"Frame rate              : {args.fps:.2f} FPS")
    print("Reference               : Start of Smart Tagging annotation")

    print_summary("ALL RESTART TYPES", all_records)

    for restart_type in sorted(grouped):
        print_summary(
            f"RESTART TYPE: {restart_type} "
            f"({len(grouped[restart_type])} events)",
            grouped[restart_type],
        )

    print("\nINTERPRETATION LIMITS")
    print("-" * 105)
    print("Annotation start is not necessarily the exact restart moment.")
    print("Windows are measured from annotation start, not from ball movement.")
    print("Overlapping events may contribute the same tracking frames.")
    print("Visibility, outside-boundary status, and speed are not D labels.")
    print("Speed statistics use only the provider-reported valid observations.")
    print("No ground-truth labels or estimator parameters were used.")
    print("No possession or dead-ball labels were generated.")


if __name__ == "__main__":
    main()
