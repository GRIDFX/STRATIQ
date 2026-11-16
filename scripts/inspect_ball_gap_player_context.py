"""Compare ball gaps with nearby player tracking continuity."""

import argparse
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH
from tactiq.features import extract_features


def get_missing_runs(missing):
    padded = np.concatenate(([False], missing, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def player_continuity(players, before, after):
    """Compare same track slots across the two frames."""
    p1 = players[before, :, :2]
    p2 = players[after, :, :2]

    valid = (
        np.isfinite(p1).all(axis=1)
        & np.isfinite(p2).all(axis=1)
    )

    if not np.any(valid):
        return 0, float("nan"), float("nan")

    delta_x = (p2[valid, 0] - p1[valid, 0]) * PITCH_LENGTH
    delta_y = (p2[valid, 1] - p1[valid, 1]) * PITCH_WIDTH
    distances = np.hypot(delta_x, delta_y)

    return (
        int(np.sum(valid)),
        float(np.median(distances)),
        float(np.percentile(distances, 95)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../more-than-a-hack-2027/data")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()

    data = load_tracking(Path(args.data_dir))
    features = extract_features(data)

    ball = data.ball
    visible = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    runs = get_missing_runs(~visible)

    candidates = []

    for start, end in runs:
        duration = (end - start) / 25

        if duration > 1 or start == 0 or end >= len(ball):
            continue

        before, after = start - 1, end

        if not visible[before] or not visible[after]:
            continue

        dx = (ball[after, 0] - ball[before, 0]) * PITCH_LENGTH
        dy = (ball[after, 1] - ball[before, 1]) * PITCH_WIDTH

        candidates.append({
            "start": start,
            "end": end,
            "distance": float(np.hypot(dx, dy)),
        })

    candidates.sort(key=lambda item: item["distance"], reverse=True)

    print("\nSTRATIQ - BALL GAP / PLAYER CONTEXT")
    print("=" * 100)
    print(f"Eligible short intervals: {len(candidates)}")

    for index, item in enumerate(candidates[:args.top], 1):
        before = item["start"] - 1
        after = item["end"]

        count, median_move, p95_move = player_continuity(
            data.players, before, after
        )

        print(f"\nCASE {index}: frames {item['start']}-{item['end'] - 1}")
        print("-" * 100)
        print(f"Ball endpoint displacement : {item['distance']:.2f} m")
        print(f"Same player tracks visible  : {count}")
        print(f"Player movement median     : {median_move:.2f} m")
        print(f"Player movement P95        : {p95_move:.2f} m")

        print(
            "FRAME    BALL_X    BALL_Y    HOME_NEAR    AWAY_NEAR"
            "    HOME_TRACK    AWAY_TRACK"
        )

        first = max(0, item["start"] - args.window)
        last = min(len(ball), item["end"] + args.window + 1)

        for frame in range(first, last):
            if visible[frame]:
                bx, by = ball[frame, :2]
                ball_text = f"{bx:>8.3f}  {by:>8.3f}"
            else:
                ball_text = "   MISSING   "

            print(
                f"{frame:>6}  {ball_text}  "
                f"{features.home_nearest_distance[frame]:>9.2f}  "
                f"{features.away_nearest_distance[frame]:>9.2f}  "
                f"{features.home_nearest_track[frame]:>11}  "
                f"{features.away_nearest_track[frame]:>11}"
            )

    print("\nNo tracking data or possession labels were modified.")


if __name__ == "__main__":
    main()
