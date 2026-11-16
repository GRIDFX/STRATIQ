"""Descriptive audit of ball tracking characteristics."""

import argparse
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH


def calculate_missing_runs(missing):
    """Return lengths of consecutive missing-ball intervals."""
    padded = np.concatenate(([False], missing, [False]))
    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return ends - starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="../more-than-a-hack-2027/data",
    )
    args = parser.parse_args()

    data = load_tracking(Path(args.data_dir))
    ball = data.ball

    x = ball[:, 0]
    y = ball[:, 1]
    reported_speed = ball[:, 2]

    visible = np.isfinite(x) & np.isfinite(y)
    missing = ~visible

    inside = (
        visible
        & (x >= 0)
        & (x <= 1)
        & (y >= 0)
        & (y <= 1)
    )

    outside = visible & ~inside

    valid_speed = np.isfinite(reported_speed)
    speed_values = reported_speed[valid_speed]

    print("\nSTRATIQ - BALL CHARACTERISTICS AUDIT")
    print("=" * 65)

    print("\n1. BALL VISIBILITY")
    print("-" * 65)
    print(f"Total frames       : {len(ball):,}")
    print(f"Visible            : {visible.sum():,} ({visible.mean()*100:.2f}%)")
    print(f"Missing            : {missing.sum():,} ({missing.mean()*100:.2f}%)")
    print(f"Inside boundary    : {inside.sum():,}")
    print(f"Outside boundary   : {outside.sum():,}")

    print("\n2. REPORTED BALL SPEED")
    print("-" * 65)
    print(f"Valid speed        : {valid_speed.sum():,}")
    print(f"Missing speed      : {(~valid_speed).sum():,}")

    if speed_values.size:
        print(f"Mean speed         : {np.mean(speed_values):.3f} m/s")
        print(f"Median speed       : {np.median(speed_values):.3f} m/s")
        print(f"P95 speed          : {np.percentile(speed_values, 95):.3f} m/s")
        print(f"Maximum speed      : {np.max(speed_values):.3f} m/s")

        for threshold in (0.1, 0.5, 1.0):
            count = np.sum(speed_values <= threshold)
            print(
                f"Speed <= {threshold:.1f} m/s : "
                f"{count:,} ({count / len(speed_values)*100:.2f}%)"
            )

    print("\n3. POSITION-BASED MOVEMENT")
    print("-" * 65)

    consecutive = visible[1:] & visible[:-1]

    dx = np.diff(x) * PITCH_LENGTH
    dy = np.diff(y) * PITCH_WIDTH
    displacement = np.hypot(dx, dy)

    valid_displacement = consecutive & np.isfinite(displacement)
    distances = displacement[valid_displacement]

    print(f"Consecutive visible pairs : {valid_displacement.sum():,}")

    if distances.size:
        derived_speed = distances * 25.0
        print(f"Mean displacement         : {np.mean(distances):.3f} m/frame")
        print(f"Median displacement       : {np.median(distances):.3f} m/frame")
        print(f"Derived speed median      : {np.median(derived_speed):.3f} m/s")
        print(f"Derived speed P95         : {np.percentile(derived_speed, 95):.3f} m/s")

        for threshold in (0.1, 0.5, 1.0):
            count = np.sum(derived_speed <= threshold)
            print(
                f"Derived speed <= {threshold:.1f} m/s: "
                f"{count:,} ({count / len(derived_speed)*100:.2f}%)"
            )

    print("\n4. CONSECUTIVE MISSING-BALL INTERVALS")
    print("-" * 65)

    runs = calculate_missing_runs(missing)

    if runs.size:
        print(f"Number of intervals : {len(runs):,}")
        print(f"Mean duration       : {np.mean(runs)/25:.2f} seconds")
        print(f"Median duration     : {np.median(runs)/25:.2f} seconds")
        print(f"Maximum duration    : {np.max(runs)/25:.2f} seconds")

        for threshold in (1, 2, 5, 10):
            count = np.sum(runs >= threshold * 25)
            print(f"Intervals >= {threshold:2d}s : {count:,}")

    print("\n5. IMPORTANT NOTES")
    print("-" * 65)
    print("Position displacement is not necessarily ball speed.")
    print("Missing ball does not automatically mean dead ball.")
    print("No possession or dead-ball labels were generated.")


if __name__ == "__main__":
    main()
