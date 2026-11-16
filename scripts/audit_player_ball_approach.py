"""Audit changes in nearest-player-to-ball distance over consecutive frames."""

import argparse
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features


def audit_team(name, distances, tracks):
    distances = np.asarray(distances, dtype=float)
    tracks = np.asarray(tracks)

    valid_pairs = (
        np.isfinite(distances[1:])
        & np.isfinite(distances[:-1])
        & (tracks[1:] >= 0)
        & (tracks[1:] == tracks[:-1])
    )

    indices = np.where(valid_pairs)[0] + 1

    if len(indices) == 0:
        print(f"\n{name}: no valid consecutive pairs")
        return

    # Positive = distance increased; negative = distance decreased.
    delta = distances[indices] - distances[indices - 1]
    current_distance = distances[indices]

    print(f"\n{'=' * 72}")
    print(name.upper())
    print("=" * 72)
    print(f"Same-track consecutive pairs : {len(delta):,}")

    print("\nDISTANCE CHANGE PER FRAME")
    print("-" * 72)
    print(f"Mean change   : {np.mean(delta):.4f} m")
    print(f"Median change : {np.median(delta):.4f} m")
    print(f"P05           : {np.percentile(delta, 5):.4f} m")
    print(f"P95           : {np.percentile(delta, 95):.4f} m")

    # A small tolerance avoids treating tiny coordinate fluctuations
    # as meaningful approach/retreat behavior.
    tolerance = 0.05
    approaching = delta < -tolerance
    moving_away = delta > tolerance
    stable = ~(approaching | moving_away)

    print("\nDIRECTION OF DISTANCE CHANGE")
    print("-" * 72)

    for label, mask in [
        ("Approaching (< -0.05m)", approaching),
        ("Moving away (> +0.05m)", moving_away),
        ("Relatively stable", stable),
    ]:
        count = int(np.sum(mask))
        print(f"{label:30}: {count:>9,} ({count / len(delta) * 100:6.2f}%)")

    print("\nBY CURRENT DISTANCE")
    print("-" * 72)

    bins = [
        ("0-2.5m", current_distance <= 2.5),
        ("2.5-5m", (current_distance > 2.5) & (current_distance <= 5)),
        ("5-10m", (current_distance > 5) & (current_distance <= 10)),
        (">10m", current_distance > 10),
    ]

    for label, mask in bins:
        count = int(np.sum(mask))
        if count == 0:
            continue

        approach_pct = np.mean(approaching[mask]) * 100
        away_pct = np.mean(moving_away[mask]) * 100
        stable_pct = np.mean(stable[mask]) * 100

        print(
            f"{label:>8}: {count:>8,} | "
            f"approaching {approach_pct:5.1f}% | "
            f"away {away_pct:5.1f}% | "
            f"stable {stable_pct:5.1f}%"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../more-than-a-hack-2027/data")
    args = parser.parse_args()

    data = load_tracking(Path(args.data_dir))
    features = extract_features(data)

    print("\nTACTIQ AI - PLAYER/BALL DISTANCE TREND AUDIT")
    print("=" * 72)
    print("Only consecutive frames with the same nearest-player track are used.")
    print("No possession labels or tracking data are modified.")

    audit_team(
        "Barcelona",
        features.home_nearest_distance,
        features.home_nearest_track,
    )

    audit_team(
        "Athletic Club",
        features.away_nearest_distance,
        features.away_nearest_track,
    )


if __name__ == "__main__":
    main()
