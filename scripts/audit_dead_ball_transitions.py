"""Inspect raw ball coordinates around dead-ball candidate segments."""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.smart_tagging_features import extract_smart_tagging_features
from tactiq.tracking_dynamics import extract_tracking_dynamics
from tactiq.dead_ball import extract_dead_ball_evidence


def contiguous_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def coordinate(frame, ball, visible):
    if frame < 0 or frame >= len(visible):
        return "N/A"

    if not visible[frame]:
        return "missing"

    x, y = ball[frame, :2]
    return f"({x:.4f}, {y:.4f})"


def state(frame, features):
    if frame < 0 or frame >= len(features.ball_visible):
        return "N/A"

    if not features.ball_visible[frame]:
        return "missing"

    if features.ball_inside_boundary[frame]:
        return "inside"

    if features.ball_outside_boundary[frame]:
        return "outside"

    return "unknown"


def outside_sides(ball_segment):
    x = ball_segment[:, 0]
    y = ball_segment[:, 1]

    counts = {
        "X<0": int(np.sum(x < 0)),
        "X>1": int(np.sum(x > 1)),
        "Y<0": int(np.sum(y < 0)),
        "Y>1": int(np.sum(y > 1)),
    }

    active = [
        side for side, count in counts.items()
        if count == max(counts.values()) and count > 0
    ]

    return "/".join(active) if active else "unknown"


def fmt_median(values):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    return f"{np.median(valid):.2f}" if valid.size else "N/A"


def inspect_record(record, ball, features, dynamics, fps):
    start, end = record
    segment = ball[start:end]
    segment_features = slice(start, end)

    outside = features.ball_outside_distance[segment_features]
    outside_valid = outside[np.isfinite(outside)]

    home = features.home_nearest_distance[segment_features]
    away = features.away_nearest_distance[segment_features]

    speed = dynamics.ball_speed[segment_features]
    speed_valid = speed[np.isfinite(speed)]

    before = start - 1
    after = end

    print(
        f"\nFrames [{start}, {end}) | "
        f"time {start / fps:.2f}s | "
        f"duration {(end - start) / fps:.2f}s"
    )

    print(
        f"  Dominant outside side : {outside_sides(segment)}"
    )

    if outside_valid.size:
        print(
            f"  Outside distance     : "
            f"median={np.median(outside_valid):.2f}m, "
            f"max={np.max(outside_valid):.2f}m"
        )

    print(
        f"  Coordinates          : "
        f"before={coordinate(before, ball, features.ball_visible)}, "
        f"first={coordinate(start, ball, features.ball_visible)}"
    )
    print(
        f"                         "
        f"middle={coordinate((start + end - 1) // 2, ball, features.ball_visible)}, "
        f"last={coordinate(end - 1, ball, features.ball_visible)}, "
        f"after={coordinate(after, ball, features.ball_visible)}"
    )

    print(
        f"  Boundary state       : "
        f"before={state(before, features)}, "
        f"after={state(after, features)}"
    )

    print(
        f"  Nearest player dist. : "
        f"home median={fmt_median(home)}m, "
        f"away median={fmt_median(away)}m"
    )

    print(
        f"  Ball speed           : "
        f"{speed_valid.size}/{end-start} available, "
        f"median={fmt_median(speed_valid)}m/s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    if args.fps <= 0 or args.limit < 1:
        parser.error("fps and limit must be positive")

    print("Loading match data...")
    data = load_tracking(args.data_dir)

    features = extract_features(data)
    dynamics = extract_tracking_dynamics(data)

    smart = extract_smart_tagging_features(
        args.xml,
        n_frames=data.n_frames,
        fps=args.fps,
    )

    result = extract_dead_ball_evidence(
        features,
        smart_tagging=smart,
        dynamics=dynamics,
    )

    runs = contiguous_runs(result.candidate)
    restart_mask = smart.restart_active.any(axis=1)

    records = []

    for start, end in runs:
        records.append({
            "start": start,
            "end": end,
            "duration": (end - start) / args.fps,
            "restart_overlap": bool(restart_mask[start:end].any()),
        })

    longest = sorted(
        records,
        key=lambda record: record["duration"],
        reverse=True,
    )[:args.limit]

    no_restart = sorted(
        [
            record for record in records
            if not record["restart_overlap"]
        ],
        key=lambda record: record["duration"],
        reverse=True,
    )[:args.limit]

    ball = np.asarray(data.ball)

    print("\nLONGEST CANDIDATE SEGMENTS")
    print("=" * 65)
    for record in longest:
        inspect_record(
            (record["start"], record["end"]),
            ball,
            features,
            dynamics,
            args.fps,
        )

    print("\nLONGEST SEGMENTS WITHOUT RESTART OVERLAP")
    print("=" * 65)
    for record in no_restart:
        inspect_record(
            (record["start"], record["end"]),
            ball,
            features,
            dynamics,
            args.fps,
        )

    print("\nCoordinate inspection completed.")
    print("No labels were modified or generated.")


if __name__ == "__main__":
    main()
