"""Inspect dead-ball candidate segments without ground-truth labels."""

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


def nearest_context_gap(start, end, context_runs):
    """Minimum frame gap between two half-open intervals."""
    if not context_runs:
        return None

    gaps = [
        max(0, context_start - end, start - context_end)
        for context_start, context_end in context_runs
    ]
    return min(gaps)


def fmt(value, precision=2):
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be greater than zero")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)
    dynamics = extract_tracking_dynamics(data)

    print("Loading Smart Tagging...")
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

    candidate_runs = contiguous_runs(result.candidate)
    context_mask = smart.restart_active.any(axis=1)
    context_runs = contiguous_runs(context_mask)

    records = []

    for start, end in candidate_runs:
        outside = features.ball_outside_distance[start:end]
        outside = outside[np.isfinite(outside)]

        speed = dynamics.ball_speed[start:end]
        speed = speed[np.isfinite(speed)]

        overlap_frames = int(context_mask[start:end].sum())
        segment_frames = end - start
        gap = nearest_context_gap(start, end, context_runs)

        records.append({
            "start": start,
            "end": end,
            "duration": segment_frames / args.fps,
            "outside_median": (
                float(np.median(outside)) if outside.size else np.nan
            ),
            "outside_max": (
                float(np.max(outside)) if outside.size else np.nan
            ),
            "restart_pct": 100.0 * overlap_frames / segment_frames,
            "restart_gap": (
                gap / args.fps if gap is not None else None
            ),
            "speed_median": (
                float(np.median(speed)) if speed.size else np.nan
            ),
            "speed_valid_pct": (
                100.0 * speed.size / segment_frames
            ),
        })

    candidate_distances = features.ball_outside_distance[result.candidate]
    candidate_speeds = dynamics.ball_speed[result.candidate]
    valid_speeds = candidate_speeds[np.isfinite(candidate_speeds)]

    print("\nCANDIDATE DISTANCE AND SPEED")
    print("=" * 65)
    print(f"Candidate frames             : {int(result.candidate.sum()):,}")

    if candidate_distances.size:
        print(
            "Outside distance median     : "
            f"{np.median(candidate_distances):.3f} m"
        )
        print(
            "Outside distance 95th pct   : "
            f"{np.percentile(candidate_distances, 95):.3f} m"
        )
        print(
            "Outside distance maximum    : "
            f"{np.max(candidate_distances):.3f} m"
        )

    print(f"Candidate frames with speed  : {valid_speeds.size:,}")

    if valid_speeds.size:
        print(f"Candidate speed median       : {np.median(valid_speeds):.3f} m/s")
        print(f"Candidate speed 95th pct     : {np.percentile(valid_speeds, 95):.3f} m/s")

    print("\nRESTART CONTEXT RELATIONSHIP")
    print("=" * 65)

    with_restart = sum(record["restart_pct"] > 0 for record in records)
    without_restart = len(records) - with_restart
    within_2s = sum(
        record["restart_gap"] is not None
        and record["restart_gap"] <= 2.0
        for record in records
    )
    within_5s = sum(
        record["restart_gap"] is not None
        and record["restart_gap"] <= 5.0
        for record in records
    )

    print(f"Candidate segments           : {len(records):,}")
    print(f"Segments overlapping restart : {with_restart:,}")
    print(f"Segments without overlap     : {without_restart:,}")
    print(f"Nearest restart within 2 sec : {within_2s:,}")
    print(f"Nearest restart within 5 sec : {within_5s:,}")

    def print_records(title, selected):
        print(f"\n{title}")
        print("-" * 125)
        print(
            f"{'Frames':<23} {'Duration':>9} {'Outside med.':>13} "
            f"{'Outside max':>12} {'Restart %':>10} "
            f"{'Restart gap':>12} {'Speed med.':>11} {'Speed valid':>12}"
        )

        for record in selected:
            frame_range = (
                f"{record['start']}-{record['end']}"
            )
            restart_gap = (
                fmt(record["restart_gap"], 2) + "s"
                if record["restart_gap"] is not None
                else "N/A"
            )

            print(
                f"{frame_range:<23} "
                f"{record['duration']:>8.2f}s "
                f"{fmt(record['outside_median']):>12}m "
                f"{fmt(record['outside_max']):>11}m "
                f"{record['restart_pct']:>9.1f}% "
                f"{restart_gap:>12} "
                f"{fmt(record['speed_median']):>10} "
                f"{record['speed_valid_pct']:>10.1f}%"
            )

    longest = sorted(
        records,
        key=lambda record: record["duration"],
        reverse=True,
    )[:15]

    print_records("15 LONGEST CANDIDATE SEGMENTS", longest)

    no_restart = [
        record for record in records
        if record["restart_pct"] == 0
    ]
    no_restart = sorted(
        no_restart,
        key=lambda record: record["duration"],
        reverse=True,
    )[:10]

    print_records("10 LONGEST SEGMENTS WITHOUT RESTART OVERLAP", no_restart)

    print("\nInspection completed.")
    print("This report does not generate or verify possession labels.")


if __name__ == "__main__":
    main()
