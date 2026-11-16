"""Descriptive audit of player movement and proximity around the ball."""

import argparse
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH
from tactiq.features import extract_features


def describe(name, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    print(f"\n{name}")
    print("-" * 75)

    if len(values) == 0:
        print("No valid observations.")
        return

    print(f"Valid observations : {len(values):,}")
    print(f"Mean                : {np.mean(values):.3f}")
    print(f"Median              : {np.median(values):.3f}")
    print(f"P95                 : {np.percentile(values, 95):.3f}")
    print(f"Maximum             : {np.max(values):.3f}")


def audit_team(name, data, features, distances, tracks, speeds):
    print(f"\n{'=' * 75}\n{name.upper()}\n{'=' * 75}")

    valid = np.isfinite(distances) & (tracks >= 0)
    near25 = valid & (distances <= 2.5)
    near5 = valid & (distances <= 5.0)

    print(f"Frames with valid nearest player : {np.sum(valid):,}")
    print(f"Nearest player within 2.5m       : {np.sum(near25):,}")
    print(f"Nearest player within 5m         : {np.sum(near5):,}")

    print("\nNEAREST PLAYER DISTANCE BINS")
    print("-" * 75)

    bins = [
        ("0-2.5m", valid & (distances <= 2.5)),
        ("2.5-5m", valid & (distances > 2.5) & (distances <= 5)),
        ("5-10m", valid & (distances > 5) & (distances <= 10)),
        (">10m", valid & (distances > 10)),
    ]

    for label, mask in bins:
        count = int(np.sum(mask))
        percentage = count / max(np.sum(valid), 1) * 100
        speed_valid = speeds[mask & np.isfinite(speeds)]

        speed_median = (
            f"{np.median(speed_valid):.2f} m/s"
            if len(speed_valid) else "N/A"
        )

        print(
            f"{label:>8}: {count:>8,} "
            f"({percentage:6.2f}%) | "
            f"median reported speed: {speed_median}"
        )

    describe("Nearest player reported speed (m/s)", speeds[valid])

    # Compare the same track in consecutive frames.
    same_track = (
        valid[1:]
        & valid[:-1]
        & (tracks[1:] == tracks[:-1])
    )

    changed_track = (
        valid[1:]
        & valid[:-1]
        & (tracks[1:] != tracks[:-1])
    )

    print("\nNEAREST TRACK CONTINUITY")
    print("-" * 75)
    print(f"Consecutive valid comparisons : {np.sum(valid[1:] & valid[:-1]):,}")
    print(f"Same nearest track             : {np.sum(same_track):,}")
    print(f"Changed nearest track          : {np.sum(changed_track):,}")

    movement = []
    speed_comparison = []

    for frame in np.where(same_track)[0] + 1:
        track = int(tracks[frame])

        previous_xy = data.players[frame - 1, track, :2]
        current_xy = data.players[frame, track, :2]

        if not (
            np.isfinite(previous_xy).all()
            and np.isfinite(current_xy).all()
        ):
            continue

        dx = (current_xy[0] - previous_xy[0]) * PITCH_LENGTH
        dy = (current_xy[1] - previous_xy[1]) * PITCH_WIDTH

        displacement = float(np.hypot(dx, dy))
        movement.append(displacement)

        reported = speeds[frame]

        if np.isfinite(reported):
            speed_comparison.append((displacement * 25, reported))

    describe("Same nearest track displacement (m/frame)", movement)

    if speed_comparison:
        comparison = np.asarray(speed_comparison)

        print("\nDERIVED VS REPORTED SPEED")
        print("-" * 75)
        print(f"Matched observations : {len(comparison):,}")
        print(f"Derived median       : {np.median(comparison[:, 0]):.3f} m/s")
        print(f"Reported median      : {np.median(comparison[:, 1]):.3f} m/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../more-than-a-hack-2027/data")
    args = parser.parse_args()

    data = load_tracking(Path(args.data_dir))
    features = extract_features(data)

    ball_visible = features.ball_visible

    print("\nTACTIQ AI - PLAYER / BALL CONTEXT AUDIT")
    print("=" * 75)
    print(f"Total frames          : {data.n_frames:,}")
    print(f"Frames with visible ball: {np.sum(ball_visible):,}")

    audit_team(
        "Barcelona",
        data,
        features,
        features.home_nearest_distance,
        features.home_nearest_track,
        features.home_nearest_speed,
    )

    audit_team(
        "Athletic Club",
        data,
        features,
        features.away_nearest_distance,
        features.away_nearest_track,
        features.away_nearest_speed,
    )

    print("\nNo tracking data or possession labels were modified.")


if __name__ == "__main__":
    main()
