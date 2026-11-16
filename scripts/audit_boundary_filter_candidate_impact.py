"""Audit how excluding out-of-pitch player coordinates changes candidates."""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH
from tactiq.features import extract_features


def candidate_state(home, away, ball_valid, radius, advantage):
    n = len(home)
    state = np.full(n, 9, dtype=np.int8)  # invalid

    valid = (
        ball_valid
        & np.isfinite(home)
        & np.isfinite(away)
    )

    state[valid] = 0  # no clear candidate

    home_gap = away[valid] - home[valid]
    away_gap = home[valid] - away[valid]

    home_indices = np.flatnonzero(valid)
    home_candidate = (
        (home[valid] <= radius)
        & (home_gap > 0)
        & (home_gap >= advantage)
    )
    away_candidate = (
        (away[valid] <= radius)
        & (away_gap > 0)
        & (away_gap >= advantage)
    )

    state[home_indices[home_candidate]] = 1
    state[home_indices[away_candidate]] = -1

    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    args = parser.parse_args()

    if args.control_radius <= 0 or args.distance_advantage < 0:
        parser.error("Invalid candidate parameters.")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Extracting original features...")
    original = extract_features(data)

    n = data.n_frames

    ball_valid = (
        original.ball_visible
        & ~original.ball_outside_boundary
    )

    player_xy = data.players[:, :, :2].astype(np.float32, copy=True)
    ball_xy = data.ball[:, :2].astype(np.float32, copy=True)

    player_valid = (
        np.isfinite(player_xy).all(axis=2)
        & (player_xy[:, :, 0] >= 0)
        & (player_xy[:, :, 0] <= 1)
        & (player_xy[:, :, 1] >= 0)
        & (player_xy[:, :, 1] <= 1)
    )

    player_xy[:, :, 0] *= PITCH_LENGTH
    player_xy[:, :, 1] *= PITCH_WIDTH
    ball_xy[:, 0] *= PITCH_LENGTH
    ball_xy[:, 1] *= PITCH_WIDTH

    delta = player_xy - ball_xy[:, None, :]
    distances = np.hypot(delta[:, :, 0], delta[:, :, 1])

    distances[~player_valid] = np.inf
    distances[~ball_valid, :] = np.inf

    home_mask = data.team_codes == 0
    away_mask = data.team_codes == 1

    home_filtered = np.min(distances[:, home_mask], axis=1)
    away_filtered = np.min(distances[:, away_mask], axis=1)

    original_state = candidate_state(
        original.home_nearest_distance,
        original.away_nearest_distance,
        ball_valid,
        args.control_radius,
        args.distance_advantage,
    )

    filtered_state = candidate_state(
        home_filtered,
        away_filtered,
        ball_valid,
        args.control_radius,
        args.distance_advantage,
    )

    comparable = (
        (original_state != 9)
        & (filtered_state != 9)
    )

    changed = comparable & (original_state != filtered_state)

    print("\nBOUNDARY FILTER CANDIDATE IMPACT")
    print("=" * 72)
    print(f"Total frames                 : {n:,}")
    print(f"Comparable frames            : {int(comparable.sum()):,}")
    print(f"Same candidate classification: {int((comparable & ~changed).sum()):,}")
    print(f"Changed classification       : {int(changed.sum()):,}")

    labels = {
        1: "H candidate",
        -1: "A candidate",
        0: "No clear candidate",
    }

    print("\nCLASSIFICATION CHANGES")
    print("-" * 72)

    for old in (1, -1, 0):
        for new in (1, -1, 0):
            mask = comparable & (original_state == old) & (filtered_state == new)
            count = int(mask.sum())

            if count:
                print(
                    f"{labels[old]:20} -> {labels[new]:20}: {count:>8,}"
                )

    print("\nCHANGED FRAMES BY BASELINE STATE")
    print("-" * 72)

    for state, name in ((1, "H"), (-1, "A"), (0, "No candidate")):
        mask = changed & (original_state == state)
        print(f"{name:15}: {int(mask.sum()):,}")

    print("\nIMPORTANT")
    print("- Original state uses the existing production feature extraction.")
    print("- Filtered state excludes out-of-pitch player coordinates.")
    print("- This compares candidate availability, not possession accuracy.")
    print("- No ground truth or submission CSV was used.")
    print("- No production source code was modified.")


if __name__ == "__main__":
    main()
