"""Inspect the largest ball displacements across missing intervals."""

import argparse
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH


def get_missing_runs(missing):
    padded = np.concatenate(([False], missing, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def is_inside(x, y):
    return 0 <= x <= 1 and 0 <= y <= 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../more-than-a-hack-2027/data")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    data = load_tracking(Path(args.data_dir))
    ball = data.ball

    visible = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    runs = get_missing_runs(~visible)

    records = []

    for start, end in runs:
        before = start - 1
        after = end

        # Exclude intervals touching the beginning or end of the dataset.
        if before < 0 or after >= len(ball):
            continue

        if not visible[before] or not visible[after]:
            continue

        x1, y1 = ball[before, :2]
        x2, y2 = ball[after, :2]

        dx = (x2 - x1) * PITCH_LENGTH
        dy = (y2 - y1) * PITCH_WIDTH
        distance = float(np.hypot(dx, dy))

        records.append({
            "start": start,
            "end": end,
            "duration": (end - start) / 25,
            "before": (float(x1), float(y1)),
            "after": (float(x2), float(y2)),
            "distance": distance,
            "before_inside": is_inside(x1, y1),
            "after_inside": is_inside(x2, y2),
            "dx": float(dx),
            "dy": float(dy),
        })

    records.sort(key=lambda item: item["distance"], reverse=True)

    print("\nTACTIQ AI - EXTREME BALL GAP INSPECTION")
    print("=" * 95)
    print(f"Total missing intervals          : {len(runs)}")
    print(f"Complete intervals inspected     : {len(records)}")
    print(f"Intervals excluded at data edges : {len(runs) - len(records)}")
    print(f"Top cases displayed              : {min(args.top, len(records))}")

    print("\nLARGEST DISPLACEMENTS")
    print("-" * 95)

    for index, item in enumerate(records[:args.top], 1):
        bx, by = item["before"]
        ax, ay = item["after"]

        print(f"\nCASE {index}")
        print(f"Missing frames : {item['start']} - {item['end'] - 1}")
        print(f"Duration       : {item['duration']:.2f} seconds")
        print(f"Before frame   : {item['start'] - 1}")
        print(f"After frame    : {item['end']}")
        print(f"Before XY      : ({bx:.5f}, {by:.5f})")
        print(f"After XY       : ({ax:.5f}, {ay:.5f})")
        print(f"Delta X / Y    : {item['dx']:.2f}m / {item['dy']:.2f}m")
        print(f"Displacement   : {item['distance']:.2f}m")
        print(f"Before inside  : {item['before_inside']}")
        print(f"After inside   : {item['after_inside']}")

    print("\nNo tracking data or possession labels were modified.")


if __name__ == "__main__":
    main()
