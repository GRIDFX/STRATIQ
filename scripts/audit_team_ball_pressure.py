"""Descriptive audit of team proximity and player-ball pressure.

This script does not generate possession labels, use ground truth,
or modify production source code.

Only finite player coordinates inside the normalized pitch are
included in the pressure calculation.
"""

import argparse

import numpy as np

from tactiq.data_loader import (
    load_tracking,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from tactiq.features import extract_features


def describe(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]

    if valid.size == 0:
        return "no valid observations"

    return (
        f"median={np.median(valid):.2f} m | "
        f"P90={np.percentile(valid, 90):.2f} m | "
        f"P95={np.percentile(valid, 95):.2f} m"
    )


def team_distance_features(distances, team_mask, radii):
    team_dist = distances[:, team_mask]

    valid_count = np.isfinite(team_dist).sum(axis=1)
    nearest = np.min(team_dist, axis=1)

    # The two closest tracked players in each team.
    two_nearest = np.partition(team_dist, kth=1, axis=1)[:, :2]
    second_nearest = two_nearest[:, 1].copy()

    nearest[valid_count < 1] = np.nan
    second_nearest[valid_count < 2] = np.nan

    nearby = {
        radius: np.sum(team_dist <= radius, axis=1)
        for radius in radii
    }

    return {
        "valid_count": valid_count,
        "nearest": nearest,
        "second_nearest": second_nearest,
        "nearby": nearby,
    }


def print_pressure_comparison(home, away, eligible, radii):
    for radius in radii:
        h_count = home["nearby"][radius][eligible]
        a_count = away["nearby"][radius][eligible]

        h_more = int(np.sum(h_count > a_count))
        a_more = int(np.sum(a_count > h_count))
        equal = int(np.sum(h_count == a_count))

        total = len(h_count)

        print(f"\nRadius: {radius:.1f} meters")
        print("-" * 65)
        print(f"Barcelona has more nearby players : {h_more:>9,} "
              f"({100*h_more/total:.2f}%)")
        print(f"Athletic has more nearby players  : {a_more:>9,} "
              f"({100*a_more/total:.2f}%)")
        print(f"Equal nearby-player count         : {equal:>9,} "
              f"({100*equal/total:.2f}%)")
        print(
            f"Mean nearby players — Barcelona  : "
            f"{np.mean(h_count):.3f}"
        )
        print(
            f"Mean nearby players — Athletic   : "
            f"{np.mean(a_count):.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Audit team-level player-ball proximity."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[2.5, 5.0, 10.0],
    )
    args = parser.parse_args()

    if any(radius <= 0 for radius in args.radii):
        parser.error("All radii must be positive.")

    if len(set(args.radii)) != len(args.radii):
        parser.error("Radius values must be unique.")

    radii = sorted(args.radii)

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Extracting existing features...")
    features = extract_features(data)

    if data.n_frames != len(features.ball_visible):
        raise ValueError("Tracking and feature frame counts differ.")

    home_mask = data.team_codes == 0
    away_mask = data.team_codes == 1

    if not home_mask.any() or not away_mask.any():
        raise ValueError("Both team tracking groups must be present.")

    player_xy = data.players[:, :, :2].astype(np.float32, copy=True)
    ball_xy = data.ball[:, :2].astype(np.float32, copy=True)

    player_finite = np.isfinite(player_xy).all(axis=2)

    player_inside = (
        player_finite
        & (player_xy[:, :, 0] >= 0)
        & (player_xy[:, :, 0] <= 1)
        & (player_xy[:, :, 1] >= 0)
        & (player_xy[:, :, 1] <= 1)
    )

    finite_outside_players = player_finite & ~player_inside

    ball_valid = (
        features.ball_visible
        & features.ball_inside_boundary
    )

    # Convert normalized coordinates to meters.
    player_xy[:, :, 0] *= PITCH_LENGTH
    player_xy[:, :, 1] *= PITCH_WIDTH

    ball_xy[:, 0] *= PITCH_LENGTH
    ball_xy[:, 1] *= PITCH_WIDTH

    delta = player_xy - ball_xy[:, np.newaxis, :]
    distances = np.hypot(delta[:, :, 0], delta[:, :, 1])

    distances[~player_inside] = np.inf
    distances[~ball_valid, :] = np.inf

    home = team_distance_features(distances, home_mask, radii)
    away = team_distance_features(distances, away_mask, radii)

    eligible = (
        ball_valid
        & (home["valid_count"] >= 1)
        & (away["valid_count"] >= 1)
    )

    eligible_second = (
        eligible
        & (home["valid_count"] >= 2)
        & (away["valid_count"] >= 2)
    )

    total = data.n_frames
    eligible_count = int(eligible.sum())

    print("\nTEAM BALL-PRESSURE AUDIT")
    print("=" * 75)
    print(f"Total frames                         : {total:,}")
    print(f"Visible ball inside pitch            : {int(ball_valid.sum()):,}")
    print(f"Visible ball outside or invalid      : {total-int(ball_valid.sum()):,}")
    print(f"Finite player coordinates             : {int(player_finite.sum()):,}")
    print(f"Finite player coordinates outside     : {int(finite_outside_players.sum()):,}")
    print(f"Frames with >=1 valid player per team : {eligible_count:,}")
    print(f"Frames with >=2 valid players per team: {int(eligible_second.sum()):,}")

    if eligible_count == 0:
        raise RuntimeError("No eligible frames for pressure analysis.")

    print("\n1. NEAREST PLAYER DISTANCES")
    print("-" * 75)
    print(
        "Barcelona nearest : "
        + describe(home["nearest"][eligible])
    )
    print(
        "Athletic nearest  : "
        + describe(away["nearest"][eligible])
    )

    nearest_margin = (
        away["nearest"][eligible]
        - home["nearest"][eligible]
    )

    print(
        "Nearest-distance margin (away - home): "
        + describe(nearest_margin)
    )
    print("Positive margin means Barcelona is geometrically closer.")

    home_closer = int(np.sum(nearest_margin > 0))
    away_closer = int(np.sum(nearest_margin < 0))
    ties = int(np.sum(nearest_margin == 0))

    print(f"Barcelona closer : {home_closer:,}")
    print(f"Athletic closer  : {away_closer:,}")
    print(f"Exact ties       : {ties:,}")

    print("\n2. SECOND-NEAREST PLAYER DISTANCES")
    print("-" * 75)
    print(
        "Barcelona second nearest : "
        + describe(home["second_nearest"][eligible_second])
    )
    print(
        "Athletic second nearest  : "
        + describe(away["second_nearest"][eligible_second])
    )

    home_gap = (
        home["second_nearest"][eligible_second]
        - home["nearest"][eligible_second]
    )
    away_gap = (
        away["second_nearest"][eligible_second]
        - away["nearest"][eligible_second]
    )

    print(
        "Barcelona second-nearest gap : "
        + describe(home_gap)
    )
    print(
        "Athletic second-nearest gap  : "
        + describe(away_gap)
    )

    print("\n3. NEARBY PLAYER COUNTS")
    print("-" * 75)
    print_pressure_comparison(home, away, eligible, radii)

    print("\n4. SIMULTANEOUS PROXIMITY")
    print("-" * 75)

    for radius in radii:
        h_near = home["nearest"] <= radius
        a_near = away["nearest"] <= radius

        both = int(np.sum(eligible & h_near & a_near))
        only_home = int(np.sum(eligible & h_near & ~a_near))
        only_away = int(np.sum(eligible & ~h_near & a_near))
        neither = int(np.sum(eligible & ~h_near & ~a_near))

        print(f"\nRadius {radius:.1f} m")
        print(f"Both teams have a player nearby : {both:,}")
        print(f"Only Barcelona nearby           : {only_home:,}")
        print(f"Only Athletic nearby            : {only_away:,}")
        print(f"Neither team nearby              : {neither:,}")

    print("\n5. MULTI-PLAYER PROXIMITY")
    print("-" * 75)

    for radius in radii:
        h_two = home["nearby"][radius] >= 2
        a_two = away["nearby"][radius] >= 2

        both = int(np.sum(eligible & h_two & a_two))
        home_only = int(np.sum(eligible & h_two & ~a_two))
        away_only = int(np.sum(eligible & ~h_two & a_two))

        print(f"\nAt least 2 players within {radius:.1f} m")
        print(f"Both teams : {both:,}")
        print(f"Barcelona  : {home_only:,}")
        print(f"Athletic   : {away_only:,}")

    print("\nIMPORTANT")
    print("- Pressure is a geometric proximity proxy, not possession truth.")
    print("- Second-nearest features require at least two valid players per team.")
    print("- Out-of-pitch player coordinates are excluded from this audit.")
    print("- No ground truth or submission CSV was loaded or created.")
    print("- No production feature or estimator source code was modified.")


if __name__ == "__main__":
    main()
