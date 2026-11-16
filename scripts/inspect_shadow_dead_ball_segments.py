"""Inspect neighboring baseline possession labels around D candidates.

No ground truth is loaded. This is diagnostic analysis only.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession
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


def nearest_team(labels, start, direction, max_distance):
    """Find nearest H/A label before or after a candidate segment."""
    for distance in range(1, max_distance + 1):
        index = start - distance if direction == "before" else start + distance

        if index < 0 or index >= len(labels):
            return None, None

        if labels[index] in ("H", "A"):
            return labels[index], distance

    return None, None


def count_context(labels, start, end, window):
    left = labels[max(0, start - window):start]
    right = labels[end:min(len(labels), end + window)]

    def counts(values):
        return {
            label: int(np.sum(values == label))
            for label in ("H", "A", "X")
        }

    return counts(left), counts(right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--context-seconds", type=float, default=1.0)
    parser.add_argument("--search-seconds", type=float, default=10.0)
    args = parser.parse_args()

    if args.fps <= 0 or args.context_seconds < 0 or args.search_seconds <= 0:
        parser.error("Invalid FPS or context window")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    features = extract_features(data)
    baseline = estimate_possession(data, features)

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

    labels = baseline.labels
    candidate = evidence.candidate
    runs = contiguous_runs(candidate)

    context_frames = int(round(args.context_seconds * args.fps))
    search_frames = int(round(args.search_seconds * args.fps))

    records = []

    for start, end in runs:
        before_team, before_gap = nearest_team(
            labels, start, "before", search_frames
        )
        after_team, after_gap = nearest_team(
            labels, end - 1, "after", search_frames
        )

        left_counts, right_counts = count_context(
            labels, start, end, context_frames
        )

        restart_overlap = int(
            smart.restart_active[start:end].any(axis=1).sum()
        )

        outside = features.ball_outside_distance[start:end]
        outside = outside[np.isfinite(outside)]

        duration = (end - start) / args.fps

        records.append({
            "start": start,
            "end": end,
            "duration": duration,
            "before": before_team,
            "after": after_team,
            "before_gap": before_gap,
            "after_gap": after_gap,
            "left": left_counts,
            "right": right_counts,
            "restart_pct": 100 * restart_overlap / (end - start),
            "outside_median": (
                float(np.median(outside)) if outside.size else np.nan
            ),
        })

    print("\nNEIGHBORING POSSESSION RELATIONSHIP")
    print("=" * 70)

    categories = {
        "H -> D -> H": 0,
        "H -> D -> A": 0,
        "A -> D -> H": 0,
        "A -> D -> A": 0,
        "H -> D -> unknown": 0,
        "A -> D -> unknown": 0,
        "unknown -> D -> H": 0,
        "unknown -> D -> A": 0,
        "unknown -> D -> unknown": 0,
    }

    for record in records:
        before = record["before"] or "unknown"
        after = record["after"] or "unknown"
        key = f"{before} -> D -> {after}"
        categories[key] += 1

    for key, count in categories.items():
        if count:
            print(f"{key:<30}: {count:>4}")

    print("\nIMMEDIATE ONE-SECOND CONTEXT")
    print("=" * 70)

    left_total = {"H": 0, "A": 0, "X": 0}
    right_total = {"H": 0, "A": 0, "X": 0}

    for record in records:
        for label in left_total:
            left_total[label] += record["left"][label]
            right_total[label] += record["right"][label]

    print("Frames immediately before candidate segments:")
    print(left_total)
    print("Frames immediately after candidate segments:")
    print(right_total)

    print("\nRESTART OVERLAP BY SEGMENT")
    print("=" * 70)

    overlap = sum(record["restart_pct"] > 0 for record in records)
    no_overlap = len(records) - overlap

    print(f"Segments overlapping restart : {overlap}")
    print(f"Segments without overlap     : {no_overlap}")

    print("\nLONGEST 20 SEGMENTS WITH NEIGHBOR CONTEXT")
    print("=" * 110)
    print(
        "FRAMES             DUR(s)  BEFORE  GAP(s)  AFTER  GAP(s)  "
        "RESTART(%)  OUTSIDE MED(m)"
    )

    for record in sorted(
        records,
        key=lambda item: item["duration"],
        reverse=True,
    )[:20]:
        before_gap = (
            f"{record['before_gap'] / args.fps:.2f}"
            if record["before_gap"] is not None else "N/A"
        )
        after_gap = (
            f"{record['after_gap'] / args.fps:.2f}"
            if record["after_gap"] is not None else "N/A"
        )
        print(
            f"{record['start']:>7}-{record['end']:<7} "
            f"{record['duration']:>7.2f} "
            f"{str(record['before'] or '-'):>7} "
            f"{before_gap:>7} "
            f"{str(record['after'] or '-'):>6} "
            f"{after_gap:>7} "
            f"{record['restart_pct']:>10.1f} "
            f"{record['outside_median']:>15.2f}"
        )

    print("\nNo ground truth was used.")
    print("No possession labels were modified.")
    print("No submission CSV was written.")


if __name__ == "__main__":
    main()
