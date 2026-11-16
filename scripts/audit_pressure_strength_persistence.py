"""Measure strength and persistence of team pressure imbalance.

Descriptive diagnostic only. No ground truth, model training, or production edits.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH


def get_run_lengths(signal):
    """Lengths of contiguous runs with a constant non-zero direction."""
    padded = np.concatenate(([0], signal, [0]))
    edges = np.diff(padded)

    starts = np.flatnonzero(edges == 1) + 0
    ends = np.flatnonzero(edges == -1)

    # A transition from +1 to -1 closes one run and starts another.
    runs = []
    start = None
    direction = 0

    for i, value in enumerate(signal):
        value = int(value)

        if value != direction:
            if direction != 0 and start is not None:
                runs.append(i - start)

            start = i if value != 0 else None
            direction = value

    if direction != 0 and start is not None:
        runs.append(len(signal) - start)

    return np.asarray(runs, dtype=np.int32)


def report_signal(name, signal, nearest_direction):
    active = signal != 0
    total = int(active.sum())

    print(f"\n{name}")
    print("-" * 72)
    print(f"Directional frames : {total:,}")
    print(f"Barcelona stronger : {int(np.sum(signal == 1)):,}")
    print(f"Athletic stronger  : {int(np.sum(signal == -1)):,}")

    if total == 0:
        print("No qualifying pressure signal.")
        return

    comparable = active & (nearest_direction != 0)
    same = comparable & (signal == nearest_direction)
    opposite = comparable & (signal == -nearest_direction)

    comparable_count = int(comparable.sum())

    print(
        f"Same direction as nearest player : "
        f"{int(same.sum()):,}/{comparable_count:,}"
    )
    print(
        f"Opposite direction               : "
        f"{int(opposite.sum()):,}/{comparable_count:,}"
    )

    runs = get_run_lengths(signal)

    if len(runs) == 0:
        print("No contiguous runs.")
        return

    print(f"Directional runs : {len(runs):,}")
    print(f"Median duration  : {np.median(runs) / 25:.2f} s")
    print(f"P90 duration     : {np.percentile(runs, 90) / 25:.2f} s")
    print(f"Maximum duration : {np.max(runs) / 25:.2f} s")

    print("\nPersistence thresholds")
    for minimum in (8, 16, 25):
        selected = runs >= minimum
        selected_runs = int(selected.sum())
        selected_frames = int(runs[selected].sum())
        coverage = 100 * selected_frames / total

        print(
            f">={minimum:2} frames ({minimum / 25:.2f}s): "
            f"{selected_runs:>5,} runs | "
            f"{selected_frames:>8,} frames | "
            f"{coverage:5.1f}% of directional frames"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument("--radii", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument(
        "--min-count-gaps",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    args = parser.parse_args()

    if args.control_radius <= 0 or args.distance_advantage < 0:
        parser.error("Invalid candidate parameters.")
    if any(radius <= 0 for radius in args.radii):
        parser.error("All radii must be positive.")
    if any(gap < 1 for gap in args.min_count_gaps):
        parser.error("Minimum count gaps must be >= 1.")

    radii = sorted(set(args.radii))
    gaps = sorted(set(args.min_count_gaps))

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

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

    home_dist = distances[:, home_mask]
    away_dist = distances[:, away_mask]

    home_valid = np.isfinite(home_dist).any(axis=1)
    away_valid = np.isfinite(away_dist).any(axis=1)

    home_nearest = np.min(home_dist, axis=1)
    away_nearest = np.min(away_dist, axis=1)

    eligible = ball_valid & home_valid & away_valid

    # Avoid inf - inf arithmetic on invalid frames.
    home_advantage = np.zeros(data.n_frames, dtype=np.float32)
    home_advantage[eligible] = (
        away_nearest[eligible] - home_nearest[eligible]
    )

    home_candidate = (
        eligible
        & (home_nearest <= args.control_radius)
        & (home_advantage > 0)
        & (home_advantage >= args.distance_advantage)
    )

    away_candidate = (
        eligible
        & (away_nearest <= args.control_radius)
        & (-home_advantage > 0)
        & (-home_advantage >= args.distance_advantage)
    )

    no_candidate = eligible & ~(home_candidate | away_candidate)

    both_far = (
        no_candidate
        & (home_nearest > args.control_radius)
        & (away_nearest > args.control_radius)
    )
    near_unclear = no_candidate & ~both_far

    nearest_direction = np.zeros(data.n_frames, dtype=np.int8)
    nearest_direction[eligible & (home_advantage > 0)] = 1
    nearest_direction[eligible & (home_advantage < 0)] = -1

    print("\nPRESSURE STRENGTH & PERSISTENCE AUDIT")
    print("=" * 75)
    print(f"Total frames       : {data.n_frames:,}")
    print(f"Eligible frames    : {int(eligible.sum()):,}")
    print(f"No-candidate       : {int(no_candidate.sum()):,}")
    print(f"Near but unclear   : {int(near_unclear.sum()):,}")
    print(f"Both teams far     : {int(both_far.sum()):,}")

    for radius in radii:
        home_count = np.sum(
            home_dist <= radius,
            axis=1,
        )
        away_count = np.sum(
            away_dist <= radius,
            axis=1,
        )

        margin = home_count - away_count

        print("\n" + "=" * 75)
        print(f"RADIUS {radius:.1f} METERS")
        print("=" * 75)

        for group_name, group_mask in (
            ("NEAR-BUT-UNCLEAR", near_unclear),
            ("BOTH-TEAMS-FAR", both_far),
        ):
            print(f"\nGROUP: {group_name}")
            print("-" * 75)
            print(f"Eligible group frames: {int(group_mask.sum()):,}")

            for minimum_gap in gaps:
                mask = group_mask & (np.abs(margin) >= minimum_gap)

                signal = np.zeros(data.n_frames, dtype=np.int8)
                signal[mask] = np.sign(margin[mask]).astype(np.int8)

                report_signal(
                    f"Minimum player-count advantage: {minimum_gap}",
                    signal,
                    nearest_direction,
                )

    print("\nIMPORTANT")
    print("- Pressure imbalance is geometric context, not possession truth.")
    print("- Minimum count gap means one team has at least N more nearby players.")
    print("- Runs break when the direction changes or the gap falls below threshold.")
    print("- No ground truth, training, or submission CSV was used.")
    print("- Production source code was not modified.")


if __name__ == "__main__":
    main()
