"""Inspect frame-by-frame ball coordinates around short missing intervals."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../more-than-a-hack-2027/data")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    data = load_tracking(Path(args.data_dir))
    ball = data.ball

    visible = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    runs = get_missing_runs(~visible)

    candidates = []

    for start, end in runs:
        duration = (end - start) / 25

        if duration > 1:
            continue

        if start == 0 or end >= len(ball):
            continue

        before = start - 1
        after = end

        if not visible[before] or not visible[after]:
            continue

        dx = (ball[after, 0] - ball[before, 0]) * PITCH_LENGTH
        dy = (ball[after, 1] - ball[before, 1]) * PITCH_WIDTH

        candidates.append({
            "start": start,
            "end": end,
            "duration": duration,
            "distance": float(np.hypot(dx, dy)),
        })

    candidates.sort(key=lambda item: item["distance"], reverse=True)

    print("\nSTRATIQ - SHORT BALL GAP FRAME INSPECTION")
    print("=" * 85)
    print(f"Eligible intervals: {len(candidates)}")
    print(f"Inspecting: {min(args.top, len(candidates))} cases")

    for index, item in enumerate(candidates[:args.top], 1):
        start = item["start"]
        end = item["end"]

        print(f"\nCASE {index}")
        print("-" * 85)
        print(f"Missing frames : {start} - {end - 1}")
        print(f"Duration       : {item['duration']:.2f}s")
        print(f"Endpoint jump  : {item['distance']:.2f}m")
        print("FRAME       X          Y          SPEED       STEP DISTANCE")

        first = max(0, start - args.window)
        last = min(len(ball), end + args.window + 1)

        previous_frame = None

        for frame in range(first, last):
            x, y, speed = ball[frame]

            if not np.isfinite(x) or not np.isfinite(y):
                print(f"{frame:>6}   MISSING")
                previous_frame = None
                continue

            step = float("nan")

            if previous_frame is not None and previous_frame == frame - 1:
                px, py = ball[previous_frame, :2]
                step = float(np.hypot(
                    (x - px) * PITCH_LENGTH,
                    (y - py) * PITCH_WIDTH,
                ))

            speed_text = f"{speed:>9.3f}" if np.isfinite(speed) else "      N/A"
            step_text = f"{step:>12.2f}" if np.isfinite(step) else "          N/A"

            print(
                f"{frame:>6}   {x:>9.5f}  {y:>9.5f}  "
                f"{speed_text}  {step_text}"
            )

            previous_frame = frame

    print("\nNo tracking data or possession labels were modified.")


if __name__ == "__main__":
    main()
