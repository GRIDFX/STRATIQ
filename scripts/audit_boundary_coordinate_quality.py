"""Compare ball and player coordinate quality against normalized pitch bounds."""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.dead_ball import extract_dead_ball_evidence
from tactiq.smart_tagging_features import extract_smart_tagging_features
from tactiq.tracking_dynamics import extract_tracking_dynamics


def summarize_coordinates(name, xy):
    """Report normalized coordinate ranges and boundary violations."""

    xy = np.asarray(xy, dtype=np.float64)

    if xy.shape[-1] != 2:
        raise ValueError(f"{name}: expected final coordinate dimension 2")

    flat = xy.reshape(-1, 2)
    valid = np.isfinite(flat).all(axis=1)
    values = flat[valid]

    print(f"\n{name}")
    print("-" * 65)
    print(f"Valid coordinate observations : {len(values):,}")
    print(f"Missing/invalid observations  : {len(flat) - len(values):,}")

    if not len(values):
        print("No valid coordinates.")
        return

    x = values[:, 0]
    y = values[:, 1]

    outside_x = (x < 0) | (x > 1)
    outside_y = (y < 0) | (y > 1)
    outside_any = outside_x | outside_y

    print(
        f"Outside X boundary           : "
        f"{outside_x.sum():,} ({outside_x.mean() * 100:.3f}%)"
    )
    print(
        f"Outside Y boundary           : "
        f"{outside_y.sum():,} ({outside_y.mean() * 100:.3f}%)"
    )
    print(
        f"Outside either boundary      : "
        f"{outside_any.sum():,} ({outside_any.mean() * 100:.3f}%)"
    )

    for axis_name, axis_values in (("X", x), ("Y", y)):
        percentiles = np.percentile(
            axis_values,
            [0, 1, 50, 99, 100],
        )
        print(
            f"{axis_name} min/p01/median/p99/max      : "
            + " / ".join(f"{value:.4f}" for value in percentiles)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Audit normalized ball and player coordinate quality"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--xml", default=None)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")

    print("Loading tracking data...")
    data = load_tracking(args.data_dir)

    ball_xy = np.asarray(data.ball[:, :2])
    player_xy = np.asarray(data.players[:, :, :2])

    print("\nMATCH COORDINATE QUALITY")
    print("=" * 65)
    print(f"Frames       : {data.n_frames:,}")
    print(f"Player slots : {player_xy.shape[1]}")

    summarize_coordinates("BALL COORDINATES", ball_xy)
    summarize_coordinates("ALL PLAYER COORDINATES", player_xy)

    team_codes = np.asarray(data.team_codes)

    print("\nPLAYER COORDINATES BY TEAM CODE")
    print("=" * 65)

    for code in np.unique(team_codes):
        mask = team_codes == code
        summarize_coordinates(
            f"TEAM CODE {code} ({int(mask.sum())} slots)",
            player_xy[:, mask, :],
        )

    # Count how many player slots are outside the pitch in each frame.
    valid_player = np.isfinite(player_xy).all(axis=2)
    outside_player = (
        (player_xy[:, :, 0] < 0)
        | (player_xy[:, :, 0] > 1)
        | (player_xy[:, :, 1] < 0)
        | (player_xy[:, :, 1] > 1)
    ) & valid_player

    outside_count = outside_player.sum(axis=1)

    print("\nPLAYER OUTSIDE-BOUNDARY FRAME SUMMARY")
    print("=" * 65)
    print(f"Frames with any visible player : {valid_player.any(axis=1).sum():,}")
    print(f"Frames with outside player     : {(outside_count > 0).sum():,}")
    print(f"Frames with 3+ outside players : {(outside_count >= 3).sum():,}")
    print(f"Maximum outside players/frame : {outside_count.max(initial=0)}")

    # Per-slot diagnostic to identify whether violations are isolated.
    print("\nPLAYER TRACK SLOTS WITH MOST OUTSIDE OBSERVATIONS")
    print("=" * 65)
    slot_records = []

    for slot in range(player_xy.shape[1]):
        valid = valid_player[:, slot]
        count = int(valid.sum())

        if count == 0:
            continue

        outside = int(outside_player[:, slot].sum())
        slot_records.append((outside / count, outside, count, slot))

    for rate, outside, count, slot in sorted(
        slot_records,
        reverse=True,
    )[:10]:
        print(
            f"Track slot {slot:>3}: "
            f"{outside:>7,}/{count:>7,} "
            f"({rate * 100:6.2f}%)"
        )

    if args.xml:
        print("\nDEAD-BALL CANDIDATE COORDINATE CHECK")
        print("=" * 65)

        features = extract_features(data)
        dynamics = extract_tracking_dynamics(data)

        smart = extract_smart_tagging_features(
            args.xml,
            n_frames=data.n_frames,
            fps=args.fps,
        )

        evidence = extract_dead_ball_evidence(
            features,
            smart_tagging=smart,
            dynamics=dynamics,
        )

        candidate_xy = ball_xy[evidence.candidate]
        summarize_coordinates(
            "BALL COORDINATES IN CANDIDATE FRAMES",
            candidate_xy,
        )

    print("\nCoordinate audit completed.")
    print("No tracking values or possession labels were modified.")


if __name__ == "__main__":
    main()
