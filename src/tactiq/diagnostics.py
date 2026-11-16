"""Technical diagnostics for TACTIQ AI possession estimation."""

import argparse

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


def summarize_runs(labels):
    """Summarize consecutive runs of identical labels."""
    if len(labels) == 0:
        return {}

    changes = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[changes[1:], len(labels)]

    result = {}
    for start, end in zip(changes, ends):
        label = str(labels[start])
        result.setdefault(label, []).append(int(end - start))

    return result


def diagnose(data_dir):
    data = load_tracking(data_dir)
    features = extract_features(data)
    result = estimate_possession(data, features)

    labels = result.labels
    confidence = result.confidence
    total = len(labels)

    if total != data.n_frames:
        raise ValueError(
            f"Frame mismatch: labels={total}, data={data.n_frames}"
        )

    if not set(np.unique(labels)).issubset({"H", "A", "D", "X"}):
        raise ValueError("Invalid possession label detected.")

    if not np.isfinite(confidence).all():
        raise ValueError("Confidence contains NaN or infinite values.")

    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Confidence must be between 0 and 1.")

    ball_visible = np.asarray(features.ball_visible, dtype=bool)
    ball_outside = np.asarray(
        features.ball_outside_boundary,
        dtype=bool,
    )
    ball_inside = np.asarray(
        features.ball_inside_boundary,
        dtype=bool,
    )

    player_visible = np.isfinite(data.players[:, :, :2]).all(axis=2)
    players_per_frame = player_visible.sum(axis=1)

    home_distance_valid = np.isfinite(features.home_nearest_distance)
    away_distance_valid = np.isfinite(features.away_nearest_distance)
    both_distances_valid = home_distance_valid & away_distance_valid

    print("\n" + "=" * 60)
    print("TACTIQ AI - TECHNICAL DIAGNOSTICS")
    print("=" * 60)

    print(f"Total frames                  : {total:,}")
    print(f"Frames with ball              : {ball_visible.sum():,}")
    print(f"Frames without ball           : {(~ball_visible).sum():,}")
    print(f"Frames with players           : {(players_per_frame > 0).sum():,}")
    print(f"Frames without players        : {(players_per_frame == 0).sum():,}")

    print("\nBALL BOUNDARY")
    print(f"Ball inside boundary          : {ball_inside.sum():,}")
    print(f"Ball outside boundary         : {ball_outside.sum():,}")
    print(
        "Boundary flags consistent    : "
        f"{np.all((ball_inside | ball_outside) == ball_visible)}"
    )

    print("\nLABEL DISTRIBUTION")
    for label in ("H", "A", "D", "X"):
        count = int(np.sum(labels == label))
        print(f"{label}: {count:>9,} ({count / total * 100:6.2f}%)")

    print("\nX DIAGNOSTIC")
    x_mask = labels == "X"

    x_missing_ball = x_mask & ~ball_visible
    x_outside_boundary = x_mask & ball_outside
    x_no_players = x_mask & ball_visible & (players_per_frame == 0)

    x_missing_team_distance = (
        x_mask
        & ball_inside
        & (players_per_frame > 0)
        & ~both_distances_valid
    )

    x_visible_with_distance = (
        x_mask
        & ball_inside
        & (players_per_frame > 0)
        & both_distances_valid
    )

    print(f"X with missing ball            : {x_missing_ball.sum():,}")
    print(f"X with ball outside boundary   : {x_outside_boundary.sum():,}")
    print(f"X with no visible players      : {x_no_players.sum():,}")
    print(
        "X with missing team distance  : "
        f"{x_missing_team_distance.sum():,}"
    )
    print(
        "X with visible ball, players, "
        f"and both distances           : {x_visible_with_distance.sum():,}"
    )

    x_categories = (
        x_missing_ball.astype(np.int8)
        + x_outside_boundary.astype(np.int8)
        + x_no_players.astype(np.int8)
        + x_missing_team_distance.astype(np.int8)
        + x_visible_with_distance.astype(np.int8)
    )

    print(
        "X category accounting         : "
        f"{int(np.sum(x_categories[x_mask] == 1)):,} assigned once, "
        f"{int(np.sum(x_categories[x_mask] == 0)):,} unclassified, "
        f"{int(np.sum(x_categories[x_mask] > 1)):,} overlapping"
    )

    print("\nDISTANCE AVAILABILITY")
    print(f"Home nearest distance valid   : {home_distance_valid.sum():,}")
    print(f"Away nearest distance valid   : {away_distance_valid.sum():,}")
    print(f"Both team distances valid     : {both_distances_valid.sum():,}")

    print("\nCONFIDENCE BY LABEL")
    for label in ("H", "A", "D", "X"):
        mask = labels == label
        if mask.any():
            print(
                f"{label}: mean={confidence[mask].mean():.3f}, "
                f"median={np.median(confidence[mask]):.3f}, "
                f"count={mask.sum():,}"
            )

    print("\nPOSSESSION RUNS")
    runs = summarize_runs(labels)
    for label in ("H", "A", "D", "X"):
        durations = runs.get(label, [])
        if durations:
            print(
                f"{label}: runs={len(durations):,}, "
                f"mean={np.mean(durations):.1f} frames, "
                f"median={np.median(durations):.1f} frames, "
                f"max={max(durations):,} frames"
            )

    team_switches = np.sum(
        ((labels[:-1] == "H") & (labels[1:] == "A"))
        | ((labels[:-1] == "A") & (labels[1:] == "H"))
    )
    print(f"\nDirect H <-> A transitions: {team_switches:,}")

    print("\nTECHNICAL CHECKS")
    print(f"Frame count correct           : {total == data.n_frames}")
    print(f"Labels valid                  : True")
    print(f"Confidence finite             : {np.isfinite(confidence).all()}")
    print(
        "Confidence in range           : "
        f"{np.all((confidence >= 0) & (confidence <= 1))}"
    )
    print(
        "X categories cover all X      : "
        f"{np.all(x_categories[x_mask] == 1)}"
    )

    print("\nInterpretation:")
    print("- X categories are diagnostic groupings, not ground truth.")
    print("- X with both distances valid does not necessarily mean a bug.")
    print("- Boundary-outside does not automatically mean dead ball.")
    print("- These statistics do not measure possession accuracy.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="../more-than-a-hack-2027/data",
    )
    args = parser.parse_args()
    diagnose(args.data_dir)


if __name__ == "__main__":
    main()
