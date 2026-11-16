"""Measure temporal persistence of team player-ball pressure.

Descriptive audit only. No possession training, ground truth, or
production estimator changes.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH


def get_runs(signal):
    """Return contiguous non-zero runs with a constant direction."""
    runs = []
    start = None
    current = 0

    for i, value in enumerate(signal):
        value = int(value)

        if value != current:
            if current != 0 and start is not None:
                runs.append((start, i, current))
            start = i if value != 0 else None
            current = value

    if current != 0 and start is not None:
        runs.append((start, len(signal), current))

    return runs


def describe_runs(runs):
    if not runs:
        print("No directional pressure runs.")
        return

    lengths = np.array([end - start for start, end, _ in runs])
    print(f"Directional runs : {len(runs):,}")
    print(f"Median duration  : {np.median(lengths) / 25:.2f} s")
    print(f"P90 duration     : {np.percentile(lengths, 90) / 25:.2f} s")
    print(f"Maximum duration : {np.max(lengths) / 25:.2f} s")

    print("\nSUSTAINED PRESSURE")
    for threshold in (8, 16, 25):
        selected = lengths >= threshold
        run_count = int(selected.sum())
        frame_count = int(lengths[selected].sum())
        total_signal_frames = int(lengths.sum())

        coverage = (
            100 * frame_count / total_signal_frames
            if total_signal_frames else 0
        )

        print(
            f">= {threshold:2} frames ({threshold / 25:.2f}s): "
            f"{run_count:>5,} runs | "
            f"{frame_count:>8,} frames | "
            f"{coverage:5.1f}% of directional frames"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[2.5, 5.0, 10.0])
    args = parser.parse_args()

    if args.control_radius <= 0 or args.distance_advantage < 0:
        parser.error("Invalid candidate parameters.")
    if any(r <= 0 for r in args.radii):
        parser.error("All pressure radii must be positive.")

    radii = sorted(set(args.radii))

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

    home_valid_count = np.isfinite(home_dist).sum(axis=1)
    away_valid_count = np.isfinite(away_dist).sum(axis=1)

    home_nearest = np.min(home_dist, axis=1)
    away_nearest = np.min(away_dist, axis=1)

    eligible = (
        ball_valid
        & (home_valid_count >= 1)
        & (away_valid_count >= 1)
    )

    home_advantage = away_nearest - home_nearest
    away_advantage = home_nearest - away_nearest

    home_candidate = (
        eligible
        & (home_nearest <= args.control_radius)
        & (home_advantage > 0)
        & (home_advantage >= args.distance_advantage)
    )

    away_candidate = (
        eligible
        & (away_nearest <= args.control_radius)
        & (away_advantage > 0)
        & (away_advantage >= args.distance_advantage)
    )

    no_candidate = eligible & ~(home_candidate | away_candidate)

    both_far = (
        no_candidate
        & (home_nearest > args.control_radius)
        & (away_nearest > args.control_radius)
    )

    near_unclear = no_candidate & ~both_far

    print("\nTEMPORAL PRESSURE PERSISTENCE AUDIT")
    print("=" * 75)
    print(f"Total frames       : {data.n_frames:,}")
    print(f"Eligible frames    : {int(eligible.sum()):,}")
    print(f"No-candidate      : {int(no_candidate.sum()):,}")
    print(f"Both teams far    : {int(both_far.sum()):,}")
    print(f"Near but unclear  : {int(near_unclear.sum()):,}")

    for radius in radii:
        home_count = np.sum(home_dist <= radius, axis=1)
        away_count = np.sum(away_dist <= radius, axis=1)

        difference = home_count - away_count

        signal = np.zeros(data.n_frames, dtype=np.int8)
        signal[no_candidate & (difference > 0)] = 1
        signal[no_candidate & (difference < 0)] = -1

        directional_frames = int(np.sum(signal != 0))
        ties = int(np.sum(no_candidate & (difference == 0)))

        runs = get_runs(signal)

        print("\n" + "=" * 75)
        print(f"RADIUS {radius:.1f} METERS — NO-CANDIDATE FRAMES")
        print("=" * 75)
        print(f"Directional pressure frames : {directional_frames:,}")
        print(f"Equal player-count frames   : {ties:,}")
        print(f"Barcelona pressure stronger : "
              f"{int(np.sum(signal == 1)):,}")
        print(f"Athletic pressure stronger  : "
              f"{int(np.sum(signal == -1)):,}")

        # Compare pressure direction with nearest-distance direction.
        comparable = no_candidate & (difference != 0) & (
            home_nearest != away_nearest
        )

        agreement = comparable & (
            np.sign(difference) == np.sign(home_advantage)
        )

        comparable_count = int(comparable.sum())
        agreement_count = int(agreement.sum())

        print(
            f"Pressure/nearest agreement  : "
            f"{agreement_count:,}/{comparable_count:,}"
        )

        print("\nCONSECUTIVE DIRECTIONAL RUNS")
        print("-" * 75)
        describe_runs(runs)

        # Repeat for the near-but-unclear subgroup.
        near_signal = signal.copy()
        near_signal[~near_unclear] = 0
        near_runs = get_runs(near_signal)

        print("\nNEAR-BUT-UNCLEAR SUBGROUP")
        print("-" * 75)
        print(f"Directional frames : {int(np.sum(near_signal != 0)):,}")
        describe_runs(near_runs)

        # Repeat for the both-far subgroup.
        far_signal = signal.copy()
        far_signal[~both_far] = 0
        far_runs = get_runs(far_signal)

        print("\nBOTH-TEAMS-FAR SUBGROUP")
        print("-" * 75)
        print(f"Directional frames : {int(np.sum(far_signal != 0)):,}")
        describe_runs(far_runs)

    print("\nIMPORTANT")
    print("- A directional run is pressure imbalance, not possession.")
    print("- Ties and frames outside no-candidate conditions break a run.")
    print("- Consecutive frames are not independent observations.")
    print("- No ground truth, model training, or submission CSV was used.")
    print("- Production source code was not modified.")


if __name__ == "__main__":
    main()
