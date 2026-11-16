"""Compare geometric pressure signals with production baseline state.

Diagnostic only. Baseline labels are heuristic outputs, not ground truth.
No production code, model parameters, or submission CSV are modified.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


def contiguous_runs(signal):
    """Return (length, direction) for contiguous nonzero constant signals."""
    runs = []
    start = None
    current = 0

    for index, raw_value in enumerate(signal):
        value = int(raw_value)

        if value != current:
            if current != 0 and start is not None:
                runs.append((index - start, current))

            current = value
            start = index if value != 0 else None

    if current != 0 and start is not None:
        runs.append((len(signal) - start, current))

    return runs


def report_runs(runs):
    if not runs:
        print("  No directional runs.")
        return

    lengths = np.array([length for length, _ in runs])

    print(f"  Runs             : {len(runs):,}")
    print(f"  Median duration  : {np.median(lengths) / 25:.2f} s")
    print(f"  P90 duration     : {np.percentile(lengths, 90) / 25:.2f} s")
    print(f"  Maximum duration : {np.max(lengths) / 25:.2f} s")

    for threshold in (8, 16, 25):
        selected = lengths >= threshold
        count = int(selected.sum())
        frames = int(lengths[selected].sum())
        print(
            f"  >= {threshold:2} frames ({threshold / 25:.2f}s): "
            f"{count:>5,} runs | {frames:>8,} frames"
        )


def analyze_pressure_group(name, group, labels, pressure_signal):
    total = int(group.sum())

    print(f"\n{name}")
    print("-" * 72)
    print(f"Frames: {total:,}")

    if total == 0:
        return

    base_h = group & (labels == "H")
    base_a = group & (labels == "A")
    base_x = group & (labels == "X")

    print("\nBaseline output distribution")
    print(f"  H: {int(base_h.sum()):,}")
    print(f"  A: {int(base_a.sum()):,}")
    print(f"  X: {int(base_x.sum()):,}")

    known = base_h | base_a
    directed = group & (pressure_signal != 0) & known

    same = (
        (base_h & (pressure_signal == 1))
        | (base_a & (pressure_signal == -1))
    )
    opposite = (
        (base_h & (pressure_signal == -1))
        | (base_a & (pressure_signal == 1))
    )
    neutral = known & (pressure_signal == 0)

    known_count = int(known.sum())
    directed_count = int(directed.sum())

    print("\nPressure compared with retained baseline team")
    print(f"  Baseline H/A frames : {known_count:,}")
    print(f"  Pressure directional: {directed_count:,}")
    print(f"  Pressure agrees     : {int(same.sum()):,}")
    print(f"  Pressure disagrees  : {int(opposite.sum()):,}")
    print(f"  Pressure neutral    : {int(neutral.sum()):,}")

    if known_count:
        print(f"  Directional coverage: {100 * directed_count / known_count:.2f}%")
        print(f"  Agreement rate      : {100 * same.sum() / known_count:.2f}%")
        print(f"  Disagreement rate   : {100 * opposite.sum() / known_count:.2f}%")

    # Consecutive frames where pressure disagrees with a retained H/A label.
    disagreement = np.zeros(len(labels), dtype=np.int8)
    disagreement[group & (base_h | base_a) & (
        ((labels == "H") & (pressure_signal == -1))
        | ((labels == "A") & (pressure_signal == 1))
    )] = 1

    disagreement_runs = contiguous_runs(disagreement)

    print("\nConsecutive pressure disagreement")
    report_runs(disagreement_runs)

    # For baseline X, retain the directional pressure signal.
    x_signal = pressure_signal.copy()
    x_signal[~base_x] = 0
    x_signal[~group] = 0

    print("\nPressure direction while baseline outputs X")
    print(f"  Directional frames, H pressure: {int(np.sum(x_signal == 1)):,}")
    print(f"  Directional frames, A pressure: {int(np.sum(x_signal == -1)):,}")
    print(f"  Neutral frames                 : {int(np.sum(group & base_x & (pressure_signal == 0))):,}")

    x_runs = contiguous_runs(x_signal)
    report_runs(x_runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument("--radii", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument("--min-count-gaps", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()

    if args.control_radius <= 0 or args.distance_advantage < 0:
        parser.error("Invalid candidate parameters.")
    if any(radius <= 0 for radius in args.radii):
        parser.error("All radii must be positive.")
    if any(gap < 1 for gap in args.min_count_gaps):
        parser.error("Minimum count gaps must be >= 1.")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Extracting production features...")
    features = extract_features(data)

    print("Running production baseline...")
    baseline = estimate_possession(
        data,
        features,
        control_radius=args.control_radius,
        min_distance_advantage=args.distance_advantage,
        switch_frames=8,
        missing_ball_hold_frames=25,
        no_evidence_hold_frames=25,
        no_candidate_hold_frames=25,
    )

    home_distance = features.home_nearest_distance
    away_distance = features.away_nearest_distance

    valid_distance = (
        features.ball_visible
        & ~features.ball_outside_boundary
        & np.isfinite(home_distance)
        & np.isfinite(away_distance)
    )

    home_advantage = np.zeros(data.n_frames, dtype=np.float32)
    away_advantage = np.zeros(data.n_frames, dtype=np.float32)

    home_advantage[valid_distance] = (
        away_distance[valid_distance] - home_distance[valid_distance]
    )
    away_advantage[valid_distance] = -home_advantage[valid_distance]

    home_candidate = (
        valid_distance
        & (home_distance <= args.control_radius)
        & (home_advantage > 0)
        & (home_advantage >= args.distance_advantage)
    )
    away_candidate = (
        valid_distance
        & (away_distance <= args.control_radius)
        & (away_advantage > 0)
        & (away_advantage >= args.distance_advantage)
    )

    no_candidate = valid_distance & ~(home_candidate | away_candidate)

    both_far = (
        no_candidate
        & (home_distance > args.control_radius)
        & (away_distance > args.control_radius)
    )
    near_unclear = no_candidate & ~both_far

    # Pressure distances exclude players with invalid or out-of-pitch XY.
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

    print("\nPRESSURE VS BASELINE STATE AUDIT")
    print("=" * 72)
    print(f"Total frames       : {data.n_frames:,}")
    print(f"No-candidate       : {int(no_candidate.sum()):,}")
    print(f"Near but unclear   : {int(near_unclear.sum()):,}")
    print(f"Both teams far     : {int(both_far.sum()):,}")

    for radius in sorted(set(args.radii)):
        home_count = np.sum(home_dist <= radius, axis=1)
        away_count = np.sum(away_dist <= radius, axis=1)
        difference = home_count - away_count

        print("\n" + "=" * 72)
        print(f"PRESSURE RADIUS {radius:.1f} METERS")
        print("=" * 72)

        for minimum_gap in sorted(set(args.min_count_gaps)):
            signal = np.zeros(data.n_frames, dtype=np.int8)

            signal[no_candidate & (difference >= minimum_gap)] = 1
            signal[no_candidate & (difference <= -minimum_gap)] = -1

            print("\n" + "-" * 72)
            print(f"MINIMUM COUNT ADVANTAGE: {minimum_gap}")
            print("-" * 72)

            analyze_pressure_group(
                "ALL NO-CANDIDATE FRAMES",
                no_candidate,
                baseline.labels,
                signal,
            )
            analyze_pressure_group(
                "NEAR-BUT-UNCLEAR",
                near_unclear,
                baseline.labels,
                signal,
            )
            analyze_pressure_group(
                "BOTH TEAMS FAR",
                both_far,
                baseline.labels,
                signal,
            )

    print("\nIMPORTANT")
    print("- Baseline H/A labels in no-candidate frames may be held states.")
    print("- Baseline X is abstention, not proof of no possession.")
    print("- Pressure agreement or disagreement is not accuracy.")
    print("- Pressure counts exclude out-of-pitch player coordinates.")
    print("- No ground truth or submission CSV was used.")
    print("- No production source code was modified.")


if __name__ == "__main__":
    main()
