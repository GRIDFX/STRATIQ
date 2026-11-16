"""Analyze player-ball pressure in geometrically ambiguous frames.

Descriptive diagnostics only. No ground truth, training, or production edits.
Out-of-pitch player coordinates are excluded.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not len(values):
        return "N/A"

    return (
        f"median={np.median(values):.2f} m | "
        f"P90={np.percentile(values, 90):.2f} m"
    )


def team_metrics(distances, mask, radii):
    team = distances[:, mask]
    valid_count = np.isfinite(team).sum(axis=1)

    nearest = np.min(team, axis=1)
    second = np.partition(team, kth=1, axis=1)[:, 1].copy()

    nearest[valid_count < 1] = np.nan
    second[valid_count < 2] = np.nan

    counts = {
        radius: np.sum(team <= radius, axis=1)
        for radius in radii
    }

    return {
        "valid_count": valid_count,
        "nearest": nearest,
        "second": second,
        "counts": counts,
    }


def report_group(name, selected, home, away, radii):
    total = int(selected.sum())

    print(f"\n{name}")
    print("=" * 72)
    print(f"Frames: {total:,}")

    if total == 0:
        return

    h_near = home["nearest"][selected]
    a_near = away["nearest"][selected]

    margin = a_near - h_near

    print("Nearest distances")
    print(f"  Barcelona : {describe(h_near)}")
    print(f"  Athletic  : {describe(a_near)}")
    print(f"  Margin A-H: {describe(margin)}")

    second_mask = (
        selected
        & np.isfinite(home["second"])
        & np.isfinite(away["second"])
    )

    print("Second-nearest distances (both teams have >=2 players)")
    print(f"  Eligible frames: {int(second_mask.sum()):,}")
    print(f"  Barcelona     : {describe(home['second'][second_mask])}")
    print(f"  Athletic      : {describe(away['second'][second_mask])}")

    print("\nNearby player comparison")

    for radius in radii:
        h = home["counts"][radius][selected]
        a = away["counts"][radius][selected]

        h_more = int(np.sum(h > a))
        a_more = int(np.sum(a > h))
        equal = int(np.sum(a == h))

        pressure_margin = h - a
        nearest_sign = np.sign(margin)
        pressure_sign = np.sign(pressure_margin)

        non_tie = (nearest_sign != 0) & (pressure_sign != 0)
        agreement = int(np.sum(
            non_tie & (nearest_sign == pressure_sign)
        ))

        print(f"\n  Radius {radius:.1f} m")
        print(f"    Mean count Barcelona : {np.mean(h):.3f}")
        print(f"    Mean count Athletic  : {np.mean(a):.3f}")
        print(f"    Barcelona more       : {h_more:,}")
        print(f"    Athletic more        : {a_more:,}")
        print(f"    Equal                : {equal:,}")
        print(
            f"    Pressure/nearest direction agreement: "
            f"{agreement:,}/{int(non_tie.sum()):,}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[2.5, 5.0, 10.0],
    )
    args = parser.parse_args()

    if args.control_radius <= 0:
        parser.error("control-radius must be positive")
    if args.distance_advantage < 0:
        parser.error("distance-advantage cannot be negative")
    if any(r <= 0 for r in args.radii):
        parser.error("All radii must be positive")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Preparing valid in-pitch player coordinates...")

    player_xy = data.players[:, :, :2].astype(np.float32, copy=True)
    ball_xy = data.ball[:, :2].astype(np.float32, copy=True)

    player_valid = (
        np.isfinite(player_xy).all(axis=2)
        & (player_xy[:, :, 0] >= 0)
        & (player_xy[:, :, 0] <= 1)
        & (player_xy[:, :, 1] >= 0)
        & (player_xy[:, :, 1] <= 1)
    )

    ball_valid = (
        np.isfinite(ball_xy).all(axis=1)
        & (ball_xy[:, 0] >= 0)
        & (ball_xy[:, 0] <= 1)
        & (ball_xy[:, 1] >= 0)
        & (ball_xy[:, 1] <= 1)
    )

    player_xy[:, :, 0] *= PITCH_LENGTH
    player_xy[:, :, 1] *= PITCH_WIDTH
    ball_xy[:, 0] *= PITCH_LENGTH
    ball_xy[:, 1] *= PITCH_WIDTH

    delta = player_xy - ball_xy[:, np.newaxis, :]
    distances = np.hypot(delta[:, :, 0], delta[:, :, 1])

    distances[~player_valid] = np.inf
    distances[~ball_valid, :] = np.inf

    home_mask = data.team_codes == 0
    away_mask = data.team_codes == 1

    radii = sorted(set(args.radii))

    home = team_metrics(distances, home_mask, radii)
    away = team_metrics(distances, away_mask, radii)

    eligible = (
        ball_valid
        & (home["valid_count"] >= 1)
        & (away["valid_count"] >= 1)
    )

    h = home["nearest"]
    a = away["nearest"]

    home_advantage = a - h
    away_advantage = h - a

    candidate_home = (
        eligible
        & (h <= args.control_radius)
        & (home_advantage > 0)
        & (home_advantage >= args.distance_advantage)
    )

    candidate_away = (
        eligible
        & (a <= args.control_radius)
        & (away_advantage > 0)
        & (away_advantage >= args.distance_advantage)
    )

    no_candidate = eligible & ~(candidate_home | candidate_away)

    both_far = (
        no_candidate
        & (h > args.control_radius)
        & (a > args.control_radius)
    )

    near_but_unclear = no_candidate & ~both_far

    print("\nAMBIGUOUS PRESSURE FEATURE AUDIT")
    print("=" * 72)
    print(f"Total frames               : {data.n_frames:,}")
    print(f"Ball inside pitch          : {int(ball_valid.sum()):,}")
    print(f"Eligible frames            : {int(eligible.sum()):,}")
    print(f"Home geometric candidates  : {int(candidate_home.sum()):,}")
    print(f"Away geometric candidates  : {int(candidate_away.sum()):,}")
    print(f"No clear candidate         : {int(no_candidate.sum()):,}")
    print(f"  Both teams farther than {args.control_radius:.1f} m: "
          f"{int(both_far.sum()):,}")
    print(f"  At least one team within radius, unclear: "
          f"{int(near_but_unclear.sum()):,}")

    report_group("ALL ELIGIBLE FRAMES", eligible, home, away, radii)
    report_group("HOME CANDIDATE FRAMES", candidate_home, home, away, radii)
    report_group("AWAY CANDIDATE FRAMES", candidate_away, home, away, radii)
    report_group("NO-CANDIDATE FRAMES", no_candidate, home, away, radii)
    report_group("NO-CANDIDATE — BOTH TEAMS FAR", both_far, home, away, radii)
    report_group(
        "NO-CANDIDATE — NEAR BUT UNCLEAR",
        near_but_unclear,
        home,
        away,
        radii,
    )

    print("\nIMPORTANT")
    print("- Nearby-player count is a geometric pressure proxy.")
    print("- Agreement with nearest distance is not possession accuracy.")
    print("- Second-nearest statistics require at least two valid players per team.")
    print("- No ground truth, model training, or submission CSV was used.")
    print("- No production source code was modified.")


if __name__ == "__main__":
    main()
