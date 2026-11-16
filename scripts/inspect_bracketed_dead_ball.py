"""Detailed inspection of bracketed dead-ball candidate segments."""

import argparse
import numpy as np

from tactiq.data_loader import (
    load_tracking,
    PITCH_LENGTH,
    PITCH_WIDTH,
)
from tactiq.features import extract_features
from tactiq.smart_tagging_features import extract_smart_tagging_features
from tactiq.tracking_dynamics import extract_tracking_dynamics
from tactiq.dead_ball import extract_dead_ball_evidence


def contiguous_runs(mask):
    edges = np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def xy_text(ball, frame):
    if frame < 0 or frame >= len(ball):
        return "N/A"

    xy = ball[frame, :2]

    if not np.isfinite(xy).all():
        return "missing"

    return f"({xy[0]:.4f}, {xy[1]:.4f})"


def displacement_m(a, b):
    if not (
        np.isfinite(a[:2]).all()
        and np.isfinite(b[:2]).all()
    ):
        return np.nan

    dx = (b[0] - a[0]) * PITCH_LENGTH
    dy = (b[1] - a[1]) * PITCH_WIDTH
    return float(np.hypot(dx, dy))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("fps must be positive")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)
    features = extract_features(data)

    smart = extract_smart_tagging_features(
        args.xml,
        n_frames=data.n_frames,
        fps=args.fps,
    )

    dynamics = extract_tracking_dynamics(data)

    evidence = extract_dead_ball_evidence(
        features,
        smart_tagging=smart,
        dynamics=dynamics,
    )

    ball = data.ball
    runs = contiguous_runs(evidence.candidate)

    bracketed = [
        (start, end)
        for start, end in runs
        if (
            start > 0
            and end < data.n_frames
            and features.ball_inside_boundary[start - 1]
            and features.ball_inside_boundary[end]
        )
    ]

    print("\nBRACKETED SEGMENT DETAIL")
    print("=" * 75)
    print(f"Bracketed segments: {len(bracketed)}")

    for number, (start, end) in enumerate(bracketed, start=1):
        segment = ball[start:end]

        outside_distance = features.ball_outside_distance[start:end]
        outside_distance = outside_distance[
            np.isfinite(outside_distance)
        ]

        displacement = np.hypot(
            np.diff(segment[:, 0]) * PITCH_LENGTH,
            np.diff(segment[:, 1]) * PITCH_WIDTH,
        )

        displacement = displacement[np.isfinite(displacement)]

        large_jumps = int(np.sum(displacement > 3.0))

        speed = dynamics.ball_speed[start:end]
        valid_speed = speed[np.isfinite(speed)]

        home = features.home_nearest_distance[start:end]
        away = features.away_nearest_distance[start:end]

        home_valid = home[np.isfinite(home)]
        away_valid = away[np.isfinite(away)]

        restart = smart.restart_active[start:end].any(axis=1)

        print(f"\nSEGMENT {number}")
        print("-" * 75)
        print(
            f"Frames             : [{start}, {end})"
        )
        print(
            f"Time               : {start / args.fps:.2f}s"
        )
        print(
            f"Duration           : {(end - start) / args.fps:.2f}s"
        )

        print("\nCoordinates:")
        print(f"  Before           : {xy_text(ball, start - 1)}")
        print(f"  First outside    : {xy_text(ball, start)}")
        print(f"  Middle            : {xy_text(ball, (start + end - 1) // 2)}")
        print(f"  Last outside     : {xy_text(ball, end - 1)}")
        print(f"  After             : {xy_text(ball, end)}")

        entry_jump = displacement_m(ball[start - 1], ball[start])
        exit_jump = displacement_m(ball[end - 1], ball[end])

        print("\nBoundary crossing:")
        print(f"  Entry displacement: {entry_jump:.3f} m")
        print(f"  Exit displacement : {exit_jump:.3f} m")
        print(
            f"  Outside median    : "
            f"{np.median(outside_distance):.3f} m"
        )
        print(
            f"  Outside maximum   : "
            f"{np.max(outside_distance):.3f} m"
        )

        print("\nTracking diagnostics:")
        print(
            f"  Internal jumps >3m/frame : {large_jumps}"
        )
        print(
            f"  Ball speed available     : "
            f"{len(valid_speed)}/{end-start}"
        )
        print(
            f"  Home nearest median      : "
            f"{np.median(home_valid):.2f} m"
            if len(home_valid) else "  Home nearest median      : N/A"
        )
        print(
            f"  Away nearest median      : "
            f"{np.median(away_valid):.2f} m"
            if len(away_valid) else "  Away nearest median      : N/A"
        )
        print(
            f"  Restart overlap         : "
            f"{int(restart.sum())}/{end-start} frames"
        )

    print("\nInspection completed.")
    print("No labels or tracking data were modified.")


if __name__ == "__main__":
    main()
