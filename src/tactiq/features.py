from dataclasses import dataclass

import numpy as np

from tactiq.data_loader import (
    TrackingData,
    PITCH_LENGTH,
    PITCH_WIDTH,
)


@dataclass
class FeatureData:
    """Numerical features for every match frame."""

    ball_visible: np.ndarray
    ball_inside_boundary: np.ndarray
    ball_outside_boundary: np.ndarray
    ball_outside_distance: np.ndarray

    home_nearest_distance: np.ndarray
    away_nearest_distance: np.ndarray
    distance_difference: np.ndarray
    home_nearest_track: np.ndarray
    away_nearest_track: np.ndarray
    home_players_nearby: np.ndarray
    away_players_nearby: np.ndarray
    home_nearest_speed: np.ndarray
    away_nearest_speed: np.ndarray


def extract_features(data: TrackingData, radius=5.0):
    """Calculate ball, boundary, and player-related features.

    Distances are measured in meters.
    Speeds are read directly from the tracking provider's speed field.
    Track indices refer to the original tracking slots.
    Missing values remain NaN where applicable.

    Boundary features describe normalized coordinate limits only.
    They do not determine whether the ball is actually dead.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and greater than zero")

    if data.players.ndim != 3 or data.players.shape[2] < 3:
        raise ValueError("data.players must have shape (frames, tracks, 3)")
    if data.ball.ndim != 2 or data.ball.shape[1] < 2:
        raise ValueError("data.ball must have shape (frames, 3)")
    if data.players.shape[0] != data.ball.shape[0]:
        raise ValueError("Player and ball frame counts do not match")

    # Keep original normalized ball coordinates for boundary checks.
    ball_normalized = data.ball[:, :2].astype(np.float32).copy()

    ball_visible = np.isfinite(ball_normalized).all(axis=1)

    x = ball_normalized[:, 0]
    y = ball_normalized[:, 1]

    # Distance beyond the normalized pitch rectangle.
    # X is converted using pitch length; Y using pitch width.
    outside_x = np.maximum(
        np.maximum(-x, x - 1.0),
        0.0,
    ) * PITCH_LENGTH

    outside_y = np.maximum(
        np.maximum(-y, y - 1.0),
        0.0,
    ) * PITCH_WIDTH

    ball_outside_distance = np.hypot(outside_x, outside_y)
    ball_outside_distance[~ball_visible] = np.nan

    ball_outside_boundary = (
        ball_visible & (ball_outside_distance > 0)
    )

    ball_inside_boundary = (
        ball_visible & ~ball_outside_boundary
    )

    # Convert normalized coordinates into meters for player distances.
    player_xy = data.players[:, :, :2].astype(np.float32).copy()
    ball_xy = ball_normalized.copy()

    player_xy[:, :, 0] *= PITCH_LENGTH
    player_xy[:, :, 1] *= PITCH_WIDTH

    ball_xy[:, 0] *= PITCH_LENGTH
    ball_xy[:, 1] *= PITCH_WIDTH

    # Read provider-reported player speeds.
    player_speed = data.players[:, :, 2].astype(np.float32).copy()
    player_speed[~np.isfinite(player_speed)] = np.nan

    # Distance from every player to the ball.
    delta = player_xy - ball_xy[:, np.newaxis, :]
    distances = np.hypot(delta[:, :, 0], delta[:, :, 1])

    distances[~np.isfinite(distances)] = np.inf

    home_mask = data.team_codes == 0
    away_mask = data.team_codes == 1

    if len(data.team_codes) != data.players.shape[1]:
        raise ValueError("Team-code count must match track count")
    if not home_mask.any() or not away_mask.any():
        raise ValueError("Both teams must have at least one tracking slot")

    def team_features(team_mask):
        team_distances = distances[:, team_mask]
        team_speeds = player_speed[:, team_mask]

        nearest_local = np.argmin(team_distances, axis=1)
        nearest_distance = np.min(team_distances, axis=1)

        # Convert local track index to original global track index.
        global_tracks = np.flatnonzero(team_mask)
        nearest_track = global_tracks[nearest_local].astype(np.int16)

        # Speed must come from the same track as the nearest player.
        nearest_speed = team_speeds[
            np.arange(len(nearest_local)),
            nearest_local,
        ].astype(np.float32, copy=True)

        nearby_count = np.sum(
            team_distances <= radius,
            axis=1,
        ).astype(np.float32)

        # No visible player or no valid player position.
        no_player = ~np.isfinite(nearest_distance)
        nearest_distance[no_player] = np.nan
        nearest_track[no_player] = -1
        nearest_speed[no_player] = np.nan
        nearby_count[no_player] = np.nan

        return (
            nearest_distance,
            nearest_track,
            nearby_count,
            nearest_speed,
        )

    (
        home_distance,
        home_track,
        home_count,
        home_speed,
    ) = team_features(home_mask)

    (
        away_distance,
        away_track,
        away_count,
        away_speed,
    ) = team_features(away_mask)

    # A missing ball means distance-based features cannot be calculated.
    home_distance[~ball_visible] = np.nan
    away_distance[~ball_visible] = np.nan
    home_track[~ball_visible] = -1
    away_track[~ball_visible] = -1
    home_count[~ball_visible] = np.nan
    away_count[~ball_visible] = np.nan
    home_speed[~ball_visible] = np.nan
    away_speed[~ball_visible] = np.nan

    # Positive: home is closer. Negative: away is closer.
    difference = away_distance - home_distance

    return FeatureData(
        ball_visible=ball_visible,
        ball_inside_boundary=ball_inside_boundary,
        ball_outside_boundary=ball_outside_boundary,
        ball_outside_distance=ball_outside_distance,
        home_nearest_distance=home_distance,
        away_nearest_distance=away_distance,
        distance_difference=difference,
        home_nearest_track=home_track,
        away_nearest_track=away_track,
        home_players_nearby=home_count,
        away_players_nearby=away_count,
        home_nearest_speed=home_speed,
        away_nearest_speed=away_speed,
    )


def print_feature_summary(features, radius=5.0):
    """Print aggregate feature diagnostics."""

    visible = features.ball_visible
    total = len(visible)
    visible_count = int(visible.sum())

    print("\nSTRATIQ - FEATURE ENGINEERING REPORT")
    print("=" * 45)
    print(f"Total frames          : {total:,}")
    print(f"Frames with ball      : {visible_count:,}")
    print(f"Frames without ball   : {total - visible_count:,}")
    print(f"Nearby-player radius  : {radius:.1f} meters")

    print("\nBall boundary:")
    print(
        f"  Inside boundary     : "
        f"{features.ball_inside_boundary.sum():,}"
    )
    print(
        f"  Outside boundary   : "
        f"{features.ball_outside_boundary.sum():,}"
    )

    valid_outside = features.ball_outside_distance[
        features.ball_outside_boundary
    ]

    if len(valid_outside):
        print(
            f"  Mean outside dist. : "
            f"{np.mean(valid_outside):.2f} m"
        )
        print(
            f"  Maximum distance   : "
            f"{np.max(valid_outside):.2f} m"
        )

    if visible_count == 0:
        print("\nNo visible ball positions available.")
        return

    print("\nNearest player distance:")
    print(
        "  Home                : "
        f"{np.nanmean(features.home_nearest_distance[visible]):.2f} m"
    )
    print(
        "  Away                : "
        f"{np.nanmean(features.away_nearest_distance[visible]):.2f} m"
    )

    print("\nAverage players within radius:")
    print(
        "  Home                : "
        f"{np.nanmean(features.home_players_nearby[visible]):.2f}"
    )
    print(
        "  Away                : "
        f"{np.nanmean(features.away_players_nearby[visible]):.2f}"
    )

    print("\nNearest player reported speed:")
    print(
        "  Home                : "
        f"{np.nanmean(features.home_nearest_speed[visible]):.2f} m/s"
    )
    print(
        "  Away                : "
        f"{np.nanmean(features.away_nearest_speed[visible]):.2f} m/s"
    )

    home_closer = (
        features.home_nearest_distance[visible]
        < features.away_nearest_distance[visible]
    )
    away_closer = (
        features.away_nearest_distance[visible]
        < features.home_nearest_distance[visible]
    )

    print("\nDistance comparison (not possession labels):")
    print(f"  Barcelona closer    : {home_closer.sum():,}")
    print(f"  Athletic Club closer: {away_closer.sum():,}")

    print("\nFeature extraction completed successfully.")


def main():
    import argparse

    from tactiq.data_loader import load_tracking

    parser = argparse.ArgumentParser(
        description="Extract possession-related tracking features"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to competition data directory",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Nearby-player radius in meters (default: 5)",
    )
    args = parser.parse_args()

    if args.radius <= 0:
        parser.error("--radius must be greater than zero")

    data = load_tracking(args.data_dir)
    features = extract_features(data, radius=args.radius)
    print_feature_summary(features, radius=args.radius)


if __name__ == "__main__":
    main()
