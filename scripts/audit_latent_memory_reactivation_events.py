"""Event-level audit of team candidates returning after long no-candidate gaps.

This is a read-only diagnostic. It uses tracking/features and the current
production estimator only to identify the last emitted team before each gap.
It does not load ground truth, change production code, or write submissions.
"""

import argparse

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


MEMORY_SCENARIOS = [("2 seconds", 50), ("5 seconds", 125)]


def classify_frames(features, control_radius, min_distance_advantage):
    """Create frame condition, candidate team, and distance-advantage arrays."""
    n_frames = len(features.ball_visible)
    statuses = np.full(n_frames, "no_candidate", dtype="<U20")
    candidates = np.full(n_frames, "", dtype="<U1")
    advantages = np.full(n_frames, np.nan, dtype=np.float32)

    for frame in range(n_frames):
        if not features.ball_visible[frame]:
            statuses[frame] = "missing_ball"
            continue

        if features.ball_outside_boundary[frame]:
            statuses[frame] = "outside_boundary"
            continue

        home = features.home_nearest_distance[frame]
        away = features.away_nearest_distance[frame]

        if not np.isfinite(home) or not np.isfinite(away):
            statuses[frame] = "no_evidence"
            continue

        home_advantage = float(away - home)
        away_advantage = float(home - away)

        if (
            home <= control_radius
            and home_advantage > 0
            and home_advantage >= min_distance_advantage
        ):
            statuses[frame] = "candidate"
            candidates[frame] = "H"
            advantages[frame] = home_advantage
        elif (
            away <= control_radius
            and away_advantage > 0
            and away_advantage >= min_distance_advantage
        ):
            statuses[frame] = "candidate"
            candidates[frame] = "A"
            advantages[frame] = away_advantage

    return statuses, candidates, advantages


def contiguous_runs(mask):
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts, ends))


def collect_events(
    statuses,
    candidates,
    advantages,
    baseline_labels,
    output_hold_frames,
    switch_frames,
):
    events = []
    no_candidate_mask = statuses == "no_candidate"

    for start, end in contiguous_runs(no_candidate_mask):
        gap_frames = end - start
        if gap_frames <= output_hold_frames:
            continue
        if start == 0 or end >= len(statuses):
            continue

        previous_team = baseline_labels[start - 1]
        if previous_team not in ("H", "A"):
            continue

        # Only analyze a direct return from no-candidate to a clear candidate.
        if statuses[end] != "candidate":
            continue

        returning_team = candidates[end]
        if returning_team not in ("H", "A"):
            continue

        streak_end = end
        while (
            streak_end < len(statuses)
            and statuses[streak_end] == "candidate"
            and candidates[streak_end] == returning_team
        ):
            streak_end += 1

        streak_frames = streak_end - end
        strength_window = advantages[end:min(streak_end, end + switch_frames)]
        finite_strength = strength_window[np.isfinite(strength_window)]

        if finite_strength.size:
            first_advantage = float(advantages[end])
            median_advantage = float(np.median(finite_strength))
            mean_advantage = float(np.mean(finite_strength))
        else:
            first_advantage = np.nan
            median_advantage = np.nan
            mean_advantage = np.nan

        events.append({
            "start": int(start),
            "end": int(end - 1),
            "reacquire_frame": int(end),
            "gap_frames": int(gap_frames),
            "gap_seconds": gap_frames / 25.0,
            "previous_team": str(previous_team),
            "returning_team": str(returning_team),
            "kind": "same" if previous_team == returning_team else "opposite",
            "candidate_streak": int(streak_frames),
            "confirmed": streak_frames >= switch_frames,
            "first_advantage": first_advantage,
            "median_advantage": median_advantage,
            "mean_advantage": mean_advantage,
            "memory_2s": gap_frames <= 50,
            "memory_5s": gap_frames <= 125,
        })

    return events


def summarize_scenario(events, name, memory_frames, switch_frames):
    active = [event for event in events if event["gap_frames"] <= memory_frames]
    same = [event for event in active if event["kind"] == "same"]
    opposite = [event for event in active if event["kind"] == "opposite"]

    print(f"\n{name.upper()} — {memory_frames} FRAMES")
    print("-" * 78)
    print(f"Eligible reactivation events : {len(active):,}")
    print(f"Same-team reactivations       : {len(same):,}")
    print(f"Opposite-team reactivations   : {len(opposite):,}")
    print(
        f"Opposite candidate confirmed : "
        f"{sum(event['confirmed'] for event in opposite):,} / {len(opposite):,}"
    )
    print(
        f"Opposite candidate unconfirmed: "
        f"{sum(not event['confirmed'] for event in opposite):,} / {len(opposite):,}"
    )

    if opposite:
        first = np.array([e["first_advantage"] for e in opposite], dtype=float)
        median = np.array([e["median_advantage"] for e in opposite], dtype=float)
        streaks = np.array([e["candidate_streak"] for e in opposite], dtype=int)
        print(f"Opposite first-gap advantage median : {np.nanmedian(first):.3f} m")
        print(f"Opposite first-8 advantage median   : {np.nanmedian(median):.3f} m")
        print(f"Opposite candidate streak median   : {np.median(streaks):.1f} frames")
        print(f"Opposite candidate streak P90       : {np.percentile(streaks, 90):.1f} frames")


def print_opposite_events(events, memory_frames, switch_frames, limit=25):
    selected = [
        e for e in events
        if e["gap_frames"] <= memory_frames and e["kind"] == "opposite"
    ]
    selected.sort(
        key=lambda e: (e["confirmed"], e["gap_frames"], e["median_advantage"]),
        reverse=True,
    )

    print(f"\nOPPOSITE-TEAM REACTIVATIONS — {memory_frames} FRAME MEMORY")
    print("-" * 112)
    print(
        f"{'GAP FRAMES':<20} {'GAP(s)':>7} {'OLD':>5} {'NEW':>5} "
        f"{'STREAK':>7} {'8-FRAME?':>9} {'FIRST GAP':>10} {'MEDIAN':>9}"
    )

    for event in selected[:limit]:
        confirmed = "YES" if event["confirmed"] else "NO"
        print(
            f"{event['start']:>7}-{event['end']:<11} "
            f"{event['gap_seconds']:>7.2f} "
            f"{event['previous_team']:>5} "
            f"{event['returning_team']:>5} "
            f"{event['candidate_streak']:>7} "
            f"{confirmed:>9} "
            f"{event['first_advantage']:>10.3f} "
            f"{event['median_advantage']:>9.3f}"
        )

    if len(selected) > limit:
        print(f"... {len(selected) - limit:,} more events omitted")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect candidate reactivation after long no-candidate intervals."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument("--switch-frames", type=int, default=8)
    parser.add_argument("--missing-ball-hold-frames", type=int, default=25)
    parser.add_argument("--no-evidence-hold-frames", type=int, default=25)
    parser.add_argument("--output-hold-frames", type=int, default=25)
    args = parser.parse_args()

    if args.control_radius <= 0:
        parser.error("control-radius must be positive")
    if args.distance_advantage < 0:
        parser.error("distance-advantage cannot be negative")
    if args.switch_frames < 1:
        parser.error("switch-frames must be at least 1")
    if args.output_hold_frames < 0:
        parser.error("output-hold-frames cannot be negative")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)
    print("Extracting features...")
    features = extract_features(data)

    print("Running production baseline...")
    baseline = estimate_possession(
        data,
        features,
        control_radius=args.control_radius,
        min_distance_advantage=args.distance_advantage,
        switch_frames=args.switch_frames,
        missing_ball_hold_frames=args.missing_ball_hold_frames,
        no_evidence_hold_frames=args.no_evidence_hold_frames,
        no_candidate_hold_frames=args.output_hold_frames,
    )

    statuses, candidates, advantages = classify_frames(
        features,
        args.control_radius,
        args.distance_advantage,
    )

    events = collect_events(
        statuses,
        candidates,
        advantages,
        baseline.labels,
        args.output_hold_frames,
        args.switch_frames,
    )

    print("\nLATENT MEMORY REACTIVATION EVENT AUDIT")
    print("=" * 78)
    print(f"Total frames                    : {data.n_frames:,}")
    print(f"No-candidate output hold        : {args.output_hold_frames} frames")
    print(f"Switch confirmation             : {args.switch_frames} frames")
    print(f"No-candidate runs > output hold : "
          f"{sum((b - a) > args.output_hold_frames for a, b in contiguous_runs(statuses == 'no_candidate')):,}")
    print(f"Direct candidate reactivations  : {len(events):,}")
    print("\nNo ground truth is loaded or used.")

    for name, memory_frames in MEMORY_SCENARIOS:
        summarize_scenario(events, name, memory_frames, args.switch_frames)
        print_opposite_events(events, memory_frames, args.switch_frames)

    print("\nINTERPRETATION GUIDE")
    print("- A same-team reactivation agrees with the last baseline team before the gap.")
    print("- An opposite-team reactivation challenges that remembered team.")
    print("- An 8-frame confirmation means the returning candidate stayed continuous long enough.")
    print("- Distance advantage is tracking proximity evidence, not verified possession truth.")
    print("- This is event-level diagnostics, not an accuracy evaluation.")
    print("- No production source code or submission CSV was modified.")


if __name__ == "__main__":
    main()
