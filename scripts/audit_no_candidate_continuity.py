"""Audit temporal behavior during frames without an eligible H/A candidate.

This script uses only tracking features and the estimator's own predictions.
It does not use ground truth, tune parameters, or modify submission labels.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


def contiguous_runs(mask):
    mask = np.asarray(mask, dtype=bool)

    edges = np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)
    )

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    return list(zip(starts.tolist(), ends.tolist()))


def nearest_team(labels, index, direction, max_frames):
    """Find the nearest predicted H/A label around an interval."""
    for offset in range(1, max_frames + 1):
        position = (
            index - offset
            if direction == "before"
            else index + offset
        )

        if position < 0 or position >= len(labels):
            break

        if labels[position] in ("H", "A"):
            return str(labels[position]), offset

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--search-seconds", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if args.fps <= 0 or args.search_seconds <= 0 or args.top < 1:
        parser.error("fps, search-seconds, and top must be positive")

    print("Loading tracking data...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)

    print("Estimating baseline possession...")
    result = estimate_possession(data, features)
    labels = result.labels

    home = features.home_nearest_distance
    away = features.away_nearest_distance

    valid = (
        features.ball_visible
        & ~features.ball_outside_boundary
        & np.isfinite(home)
        & np.isfinite(away)
    )

    home_candidate = (
        valid
        & (home <= 2.5)
        & ((away - home) > 0)
        & ((away - home) >= 1.0)
    )

    away_candidate = (
        valid
        & (away <= 2.5)
        & ((home - away) > 0)
        & ((home - away) >= 1.0)
    )

    no_candidate = valid & ~home_candidate & ~away_candidate
    runs = contiguous_runs(no_candidate)

    search_frames = int(round(args.search_seconds * args.fps))
    records = []

    for start, end in runs:
        segment_labels = labels[start:end]

        home_count = int(np.sum(segment_labels == "H"))
        away_count = int(np.sum(segment_labels == "A"))
        x_count = int(np.sum(segment_labels == "X"))

        before_team, before_gap = nearest_team(
            labels, start, "before", search_frames
        )
        after_team, after_gap = nearest_team(
            labels, end - 1, "after", search_frames
        )

        duration_frames = end - start
        duration_seconds = duration_frames / args.fps

        records.append({
            "start": start,
            "end": end,
            "frames": duration_frames,
            "duration": duration_seconds,
            "H": home_count,
            "A": away_count,
            "X": x_count,
            "before": before_team,
            "after": after_team,
            "before_gap": before_gap,
            "after_gap": after_gap,
        })

    total_no_candidate = int(no_candidate.sum())
    total_held = int(np.sum(((labels == "H") | (labels == "A")) & no_candidate))
    total_x = int(np.sum((labels == "X") & no_candidate))

    print("\nNO-CANDIDATE CONTINUITY SUMMARY")
    print("=" * 72)
    print(f"Valid visible frames          : {valid.sum():,}")
    print(f"No-candidate frames           : {total_no_candidate:,}")
    print(f"No-candidate frames labeled H/A: {total_held:,}")
    print(f"No-candidate frames labeled X : {total_x:,}")
    print(f"Contiguous no-candidate runs  : {len(runs):,}")

    if total_no_candidate:
        print(
            f"H/A retention share           : "
            f"{100 * total_held / total_no_candidate:.2f}%"
        )
        print(
            f"X share within no-candidate   : "
            f"{100 * total_x / total_no_candidate:.2f}%"
        )

    durations = np.asarray(
        [record["duration"] for record in records],
        dtype=np.float64,
    )

    if durations.size:
        print("\nRUN DURATION DISTRIBUTION")
        print("-" * 72)
        print(f"Mean duration   : {np.mean(durations):.2f} s")
        print(f"Median duration : {np.median(durations):.2f} s")
        print(f"P95 duration    : {np.percentile(durations, 95):.2f} s")
        print(f"Maximum duration: {np.max(durations):.2f} s")

        bins = [
            ("< 1 second", durations < 1),
            ("1-3 seconds", (durations >= 1) & (durations < 3)),
            ("3-5 seconds", (durations >= 3) & (durations < 5)),
            ("5-10 seconds", (durations >= 5) & (durations < 10)),
            (">= 10 seconds", durations >= 10),
        ]

        for name, mask in bins:
            print(
                f"{name:<18}: {int(mask.sum()):>7,} runs "
                f"| {durations[mask].sum():>9.2f} seconds"
            )

    print("\nPREVIOUS TEAM -> NEXT TEAM")
    print("-" * 72)

    combinations = {}

    for record in records:
        before = record["before"] or "unknown"
        after = record["after"] or "unknown"
        key = f"{before} -> {after}"
        combinations[key] = combinations.get(key, 0) + 1

    for key, count in sorted(combinations.items()):
        print(f"{key:<22}: {count:>7,}")

    print("\nRETENTION BEHAVIOR INSIDE NO-CANDIDATE RUNS")
    print("-" * 72)

    runs_with_hold = sum(record["H"] + record["A"] > 0 for record in records)
    runs_all_x = sum(record["X"] == record["frames"] for record in records)
    runs_with_x = sum(record["X"] > 0 for record in records)

    print(f"Runs containing H/A labels : {runs_with_hold:,}")
    print(f"Runs containing any X      : {runs_with_x:,}")
    print(f"Runs entirely labeled X    : {runs_all_x:,}")

    print(f"\nLONGEST {min(args.top, len(records))} NO-CANDIDATE RUNS")
    print("=" * 110)
    print(
        "FRAMES             DUR(s)     H      A      X "
        "BEFORE  AFTER  PREV_GAP(s)  NEXT_GAP(s)"
    )

    for record in sorted(
        records,
        key=lambda item: item["frames"],
        reverse=True,
    )[:args.top]:
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
            f"{record['H']:>6} "
            f"{record['A']:>6} "
            f"{record['X']:>6} "
            f"{str(record['before'] or '-'):>7} "
            f"{str(record['after'] or '-'):>6} "
            f"{before_gap:>12} "
            f"{after_gap:>12}"
        )

    print("\nNo ground truth was loaded or used.")
    print("No parameters were tuned.")
    print("No possession labels or submission files were modified.")


if __name__ == "__main__":
    main()
