"""Audit unusually large ball movements between consecutive visible frames."""

import argparse
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../more-than-a-hack-2027/data")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()

    data = load_tracking(Path(args.data_dir))
    ball = data.ball

    visible = np.isfinite(ball[:, :2]).all(axis=1)

    valid_pairs = visible[:-1] & visible[1:]
    frame_indices = np.where(valid_pairs)[0]

    dx = np.diff(ball[:, 0]) * PITCH_LENGTH
    dy = np.diff(ball[:, 1]) * PITCH_WIDTH
    distances = np.hypot(dx, dy)

    jumps = distances[valid_pairs]
    jump_frames = frame_indices

    print("\nTACTIQ AI - CONSECUTIVE BALL FRAME JUMP AUDIT")
    print("=" * 85)
    print(f"Total frames              : {len(ball):,}")
    print(f"Visible consecutive pairs : {len(jumps):,}")

    print("\nDISPLACEMENT DISTRIBUTION")
    print("-" * 85)

    for threshold in [0.5, 1, 2, 3, 5, 10, 20]:
        count = int(np.sum(jumps > threshold))
        percentage = count / max(len(jumps), 1) * 100
        print(
            f"Above {threshold:>5.1f} m/frame : "
            f"{count:>8,} ({percentage:.3f}%)"
        )

    print("\nSUMMARY")
    print("-" * 85)
    print(f"Median displacement       : {np.median(jumps):.3f} m/frame")
    print(f"P95 displacement          : {np.percentile(jumps, 95):.3f} m/frame")
    print(f"P99 displacement          : {np.percentile(jumps, 99):.3f} m/frame")
    print(f"Maximum displacement      : {np.max(jumps):.3f} m/frame")

    selected = np.where(jumps > args.threshold)[0]
    order = selected[np.argsort(jumps[selected])[::-1]]

    print(f"\nTOP {min(args.top, len(order))} JUMPS ABOVE {args.threshold:.1f} M/FRAME")
    print("-" * 85)
    print("FROM_FRAME  TO_FRAME  DISPLACEMENT  FROM_X  FROM_Y  TO_X  TO_Y")

    for index in order[:args.top]:
        frame = int(jump_frames[index])
        next_frame = frame + 1

        x1, y1 = ball[frame, :2]
        x2, y2 = ball[next_frame, :2]

        print(
            f"{frame:>10}  {next_frame:>8}  "
            f"{jumps[index]:>12.2f}  "
            f"{x1:>6.3f}  {y1:>6.3f}  "
            f"{x2:>6.3f}  {y2:>6.3f}"
        )

    print("\nNo tracking data or possession labels were modified.")


if __name__ == "__main__":
    main()
