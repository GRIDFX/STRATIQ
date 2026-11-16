"""Read reported movement speeds from the original tracking data.

This module deliberately does not derive velocity from consecutive
coordinates. Some tracking episodes contain implausible coordinate
displacements, while the provided speed field is generally consistent
with coordinate-derived speed at the match-wide level.
"""

from dataclasses import dataclass

import numpy as np

from tactiq.data_loader import load_tracking


@dataclass
class TrackingDynamics:
    """Reported movement speeds for every frame and tracking slot."""

    player_speed: np.ndarray
    ball_speed: np.ndarray


def extract_tracking_dynamics(data):
    """Extract the speed values supplied by the tracking provider.

    Player speed shape: (n_frames, n_tracks)
    Ball speed shape:   (n_frames,)

    Missing values remain NaN. Speeds are not reconstructed from
    coordinates and are not interpreted as possession labels.
    """
    player_speed = np.asarray(
        data.players[:, :, 2],
        dtype=np.float32,
    ).copy()

    ball_speed = np.asarray(
        data.ball[:, 2],
        dtype=np.float32,
    ).copy()

    # Treat non-finite values consistently as missing.
    player_speed[~np.isfinite(player_speed)] = np.nan
    ball_speed[~np.isfinite(ball_speed)] = np.nan

    return TrackingDynamics(
        player_speed=player_speed,
        ball_speed=ball_speed,
    )


def _report_speed(name, values):
    valid = values[np.isfinite(values)]

    print(f"\n{name}")
    print("-" * 45)
    print(f"Valid observations:   {len(valid):,}")
    print(f"Missing observations: {values.size - len(valid):,}")

    if valid.size == 0:
        print("No valid speed observations.")
        return

    print(f"Mean:                 {np.mean(valid):.3f} m/s")
    print(f"Median:               {np.median(valid):.3f} m/s")
    print(f"95th percentile:      {np.percentile(valid, 95):.3f} m/s")
    print(f"Maximum:              {np.max(valid):.3f} m/s")


def print_summary(dynamics):
    print("\nSTRATIQ - REPORTED TRACKING SPEED REPORT")
    print("=" * 60)
    _report_speed("REPORTED PLAYER SPEED", dynamics.player_speed)
    _report_speed("REPORTED BALL SPEED", dynamics.ball_speed)

    print("\nInterpretation")
    print("-" * 45)
    print("Speeds are read directly from the supplied tracking data.")
    print("No velocity is derived from consecutive coordinates.")
    print("No possession labels are generated.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect reported player and ball speeds"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to competition data directory",
    )
    args = parser.parse_args()

    data = load_tracking(args.data_dir)
    dynamics = extract_tracking_dynamics(data)
    print_summary(dynamics)


if __name__ == "__main__":
    main()
