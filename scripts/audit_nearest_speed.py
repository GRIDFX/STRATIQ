"""Descriptive audit of reported speeds for players nearest to the ball."""

import argparse

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


def summarize_speed(name, values, mask):
    selected = values[mask]
    valid = selected[np.isfinite(selected)]

    print(f"\n{name}")
    print(f"  Valid speed observations : {len(valid):,} / {mask.sum():,}")

    if len(valid):
        print(f"  Mean                     : {np.mean(valid):.3f} m/s")
        print(f"  Median                   : {np.median(valid):.3f} m/s")
        print(f"  95th percentile          : {np.percentile(valid, 95):.3f} m/s")
        print(f"  Maximum                  : {np.max(valid):.3f} m/s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="../more-than-a-hack-2027/data",
    )
    args = parser.parse_args()

    print("Loading tracking data...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)

    print("Estimating current possession...")
    result = estimate_possession(data, features)

    valid_frame = (
        features.ball_visible
        & features.ball_inside_boundary
        & np.isfinite(features.home_nearest_distance)
        & np.isfinite(features.away_nearest_distance)
    )

    both_far = (
        valid_frame
        & (features.home_nearest_distance > 2.5)
        & (features.away_nearest_distance > 2.5)
    )

    no_candidate_x = both_far & (result.labels == "X")

    print("\n" + "=" * 55)
    print("NEAREST PLAYER SPEED AUDIT")
    print("=" * 55)

    print(f"Total frames                 : {len(result.labels):,}")
    print(f"Valid visible frames         : {valid_frame.sum():,}")
    print(f"Both teams farther than 2.5m : {both_far.sum():,}")
    print(f"Both far and estimator X     : {no_candidate_x.sum():,}")

    summarize_speed(
        "Home nearest-player speed — all valid frames",
        features.home_nearest_speed,
        valid_frame,
    )
    summarize_speed(
        "Away nearest-player speed — all valid frames",
        features.away_nearest_speed,
        valid_frame,
    )
    summarize_speed(
        "Home nearest-player speed — both teams far",
        features.home_nearest_speed,
        both_far,
    )
    summarize_speed(
        "Away nearest-player speed — both teams far",
        features.away_nearest_speed,
        both_far,
    )

    both_speed_valid = (
        valid_frame
        & np.isfinite(features.home_nearest_speed)
        & np.isfinite(features.away_nearest_speed)
    )

    print("\nSpeed comparison")
    print(f"  Frames with both speeds    : {both_speed_valid.sum():,}")

    if both_speed_valid.any():
        home_faster = (
            features.home_nearest_speed[both_speed_valid]
            > features.away_nearest_speed[both_speed_valid]
        )
        away_faster = (
            features.away_nearest_speed[both_speed_valid]
            > features.home_nearest_speed[both_speed_valid]
        )
        tied = ~(home_faster | away_faster)

        print(f"  Home nearest player faster : {home_faster.sum():,}")
        print(f"  Away nearest player faster : {away_faster.sum():,}")
        print(f"  Equal speeds                : {tied.sum():,}")

    print("\nNote: This is descriptive analysis only.")
    print("Speed is not treated as proof of ball possession.")
    print("No ground-truth labels or parameters are changed.")


if __name__ == "__main__":
    main()
