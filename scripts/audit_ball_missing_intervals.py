"""Analyze ball positions around consecutive missing intervals."""

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
    args = parser.parse_args()

    data = load_tracking(Path(args.data_dir))
    ball = data.ball

    visible = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    runs = get_missing_runs(~visible)

    groups = {
        "Short (<2s)": [],
        "Medium (2-10s)": [],
        "Long (>10s)": [],
    }

    for start, end in runs:
        duration = (end - start) / 25

        if duration < 2:
            group = "Short (<2s)"
        elif duration <= 10:
            group = "Medium (2-10s)"
        else:
            group = "Long (>10s)"

        before = start - 1
        after = end

        displacement = np.nan
        before_inside = False
        after_inside = False

        if before >= 0 and after < len(ball):
            if visible[before] and visible[after]:
                dx = (ball[after, 0] - ball[before, 0]) * PITCH_LENGTH
                dy = (ball[after, 1] - ball[before, 1]) * PITCH_WIDTH
                displacement = float(np.hypot(dx, dy))

                before_inside = (
                    0 <= ball[before, 0] <= 1
                    and 0 <= ball[before, 1] <= 1
                )
                after_inside = (
                    0 <= ball[after, 0] <= 1
                    and 0 <= ball[after, 1] <= 1
                )

        groups[group].append({
            "start": start,
            "end": end,
            "duration": duration,
            "displacement": displacement,
            "before_inside": before_inside,
            "after_inside": after_inside,
        })

    print("\nSTRATIQ - MISSING BALL INTERVAL AUDIT")
    print("=" * 70)
    print(f"Total intervals: {len(runs)}")

    for name, events in groups.items():
        distances = [
            event["displacement"]
            for event in events
            if np.isfinite(event["displacement"])
        ]

        print(f"\n{name}")
        print("-" * 70)
        print(f"Intervals                  : {len(events)}")
        print(f"Both endpoints visible     : {len(distances)}")

        if distances:
            print(f"Median displacement        : {np.median(distances):.2f} m")
            print(f"Mean displacement          : {np.mean(distances):.2f} m")
            print(f"P95 displacement           : {np.percentile(distances, 95):.2f} m")

        print(
            "Before inside / after inside: "
            f"{sum(e['before_inside'] for e in events)} / "
            f"{sum(e['after_inside'] for e in events)}"
        )

    print("\nFIRST 20 INTERVALS")
    print("-" * 70)
    print("ID  START_FRAME  END_FRAME  DURATION  DISPLACEMENT")

    for index, (start, end) in enumerate(runs[:20], 1):
        before = start - 1
        after = end
        distance = float("nan")

        if before >= 0 and after < len(ball):
            if visible[before] and visible[after]:
                dx = (ball[after, 0] - ball[before, 0]) * PITCH_LENGTH
                dy = (ball[after, 1] - ball[before, 1]) * PITCH_WIDTH
                distance = float(np.hypot(dx, dy))

        print(
            f"{index:>2}  {start:>11}  {end:>9}  "
            f"{(end-start)/25:>7.2f}s  {distance:>10.2f}m"
        )

    print("\nNo possession or dead-ball labels were generated.")


if __name__ == "__main__":
    main()
