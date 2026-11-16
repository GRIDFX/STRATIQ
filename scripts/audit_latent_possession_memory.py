"""Audit latent possession memory without changing production code.

Output labels become X after the configured output hold expires.
The internal team state may remain available for a separate, longer
memory-retention period, but only for no-candidate intervals.

No ground truth is loaded. No submission CSV is written.
"""

import argparse

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


SCENARIOS = [
    ("Baseline memory — 1 second", 25),
    ("Latent memory — 2 seconds", 50),
    ("Latent memory — 5 seconds", 125),
]


def classify_frame(
    features,
    frame,
    control_radius,
    min_distance_advantage,
):
    """Return frame condition, candidate team, and candidate confidence."""

    if not features.ball_visible[frame]:
        return "missing_ball", None, 0.0

    if features.ball_outside_boundary[frame]:
        return "outside_boundary", None, 0.0

    home_distance = features.home_nearest_distance[frame]
    away_distance = features.away_nearest_distance[frame]

    if not np.isfinite(home_distance) or not np.isfinite(away_distance):
        return "no_evidence", None, 0.0

    home_advantage = away_distance - home_distance
    away_advantage = home_distance - away_distance

    if (
        home_distance <= control_radius
        and home_advantage > 0
        and home_advantage >= min_distance_advantage
    ):
        confidence = min(
            1.0,
            home_advantage / max(min_distance_advantage * 3, 1.0),
        )
        return "candidate", "H", confidence

    if (
        away_distance <= control_radius
        and away_advantage > 0
        and away_advantage >= min_distance_advantage
    ):
        confidence = min(
            1.0,
            away_advantage / max(min_distance_advantage * 3, 1.0),
        )
        return "candidate", "A", confidence

    return "no_candidate", None, 0.0


def replay_with_latent_memory(
    data,
    features,
    output_hold_frames=25,
    memory_frames=25,
    control_radius=2.5,
    min_distance_advantage=1.0,
    switch_frames=8,
    missing_ball_hold_frames=25,
    no_evidence_hold_frames=25,
):
    """Replay the production state machine with separate output and memory limits."""

    if output_hold_frames < 0:
        raise ValueError("output_hold_frames cannot be negative")
    if memory_frames < output_hold_frames:
        raise ValueError("memory_frames must be >= output_hold_frames")

    n_frames = len(features.ball_visible)

    labels = np.full(n_frames, "X", dtype="<U1")
    confidence = np.zeros(n_frames, dtype=np.float32)
    statuses = np.empty(n_frames, dtype="<U20")
    remembered = np.zeros(n_frames, dtype=bool)

    current_team = None
    pending_team = None
    pending_count = 0
    pending_from_memory = False

    missing_count = 0
    no_evidence_count = 0
    no_candidate_count = 0

    metrics = {
        "latent_x_frames": 0,
        "same_team_reactivations": 0,
        "opposite_team_reactivations": 0,
        "expired_memory_reentries": 0,
        "memory_confirmed_switches": 0,
    }

    for frame in range(n_frames):
        status, candidate, candidate_confidence = classify_frame(
            features,
            frame,
            control_radius,
            min_distance_advantage,
        )
        statuses[frame] = status

        # Case 1: Ball is missing.
        if status == "missing_ball":
            missing_count += 1
            no_evidence_count = 0
            no_candidate_count = 0
            pending_team = None
            pending_count = 0
            pending_from_memory = False

            if (
                current_team is not None
                and missing_count <= missing_ball_hold_frames
            ):
                labels[frame] = current_team
                confidence[frame] = 0.25
            else:
                labels[frame] = "X"
                confidence[frame] = 0.0

                if missing_count > missing_ball_hold_frames:
                    current_team = None

            remembered[frame] = current_team is not None
            continue

        missing_count = 0

        # Case 2: Ball is outside the pitch boundary.
        if status == "outside_boundary":
            labels[frame] = "X"
            confidence[frame] = 0.0

            current_team = None
            pending_team = None
            pending_count = 0
            pending_from_memory = False
            no_evidence_count = 0
            no_candidate_count = 0

            remembered[frame] = False
            continue

        # Case 3: Ball is visible, but player evidence is absent.
        if status == "no_evidence":
            no_evidence_count += 1
            no_candidate_count = 0
            pending_team = None
            pending_count = 0
            pending_from_memory = False

            if (
                current_team is not None
                and no_evidence_count <= no_evidence_hold_frames
            ):
                labels[frame] = current_team
                confidence[frame] = 0.20
            else:
                labels[frame] = "X"
                confidence[frame] = 0.0

                if no_evidence_count > no_evidence_hold_frames:
                    current_team = None

            remembered[frame] = current_team is not None
            continue

        no_evidence_count = 0

        # Case 4: No clear team candidate.
        if status == "no_candidate":
            no_candidate_count += 1
            pending_team = None
            pending_count = 0
            pending_from_memory = False

            if (
                current_team is not None
                and no_candidate_count <= output_hold_frames
            ):
                labels[frame] = current_team
                confidence[frame] = 0.25
            else:
                labels[frame] = "X"
                confidence[frame] = 0.0

                # Important difference:
                # Output may be X while current_team is still remembered.
                if no_candidate_count > memory_frames:
                    current_team = None

            if (
                labels[frame] == "X"
                and current_team is not None
            ):
                metrics["latent_x_frames"] += 1

            remembered[frame] = current_team is not None
            continue

        # Case 5: A clear candidate has returned.
        was_long_no_candidate = (
            no_candidate_count > output_hold_frames
        )

        reactivation_kind = None

        if was_long_no_candidate:
            if current_team is None:
                metrics["expired_memory_reentries"] += 1
                reactivation_kind = "expired"
            elif candidate == current_team:
                metrics["same_team_reactivations"] += 1
                reactivation_kind = "same"
            else:
                metrics["opposite_team_reactivations"] += 1
                reactivation_kind = "opposite"

        no_candidate_count = 0

        # First clear candidate initializes the state.
        if current_team is None:
            current_team = candidate
            pending_team = None
            pending_count = 0
            pending_from_memory = False

            labels[frame] = current_team
            confidence[frame] = candidate_confidence
            remembered[frame] = True
            continue

        # Same team remains in control.
        if candidate == current_team:
            pending_team = None
            pending_count = 0
            pending_from_memory = False

            labels[frame] = current_team
            confidence[frame] = candidate_confidence
            remembered[frame] = True
            continue

        # Opposing candidate must persist before switching.
        if candidate == pending_team:
            pending_count += 1
        else:
            pending_team = candidate
            pending_count = 1
            pending_from_memory = reactivation_kind == "opposite"

        if pending_count >= switch_frames:
            current_team = candidate

            if pending_from_memory:
                metrics["memory_confirmed_switches"] += 1

            pending_team = None
            pending_count = 0
            pending_from_memory = False

        labels[frame] = current_team
        confidence[frame] = (
            candidate_confidence if current_team == candidate else 0.2
        )
        remembered[frame] = current_team is not None

    return {
        "labels": labels,
        "confidence": confidence,
        "statuses": statuses,
        "remembered": remembered,
        "metrics": metrics,
    }


def count_transitions(labels):
    return int(np.sum(labels[1:] != labels[:-1]))


def count_runs(labels, label):
    mask = labels == label

    if not mask.any():
        return 0

    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    return int(np.sum(edges == 1))


def changed_episodes(changed):
    padded = np.concatenate(([False], changed, [False]))
    edges = np.diff(padded.astype(np.int8))

    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    return list(zip(starts, ends))


def print_label_summary(labels):
    total = len(labels)

    for label, name in (
        ("H", "Barcelona"),
        ("A", "Athletic Club"),
        ("D", "Dead ball"),
        ("X", "Abstain"),
    ):
        count = int(np.sum(labels == label))
        print(f"{name:15}: {count:>9,} ({count / total * 100:6.2f}%)")

    print(f"Transitions     : {count_transitions(labels):,}")
    print(f"H runs          : {count_runs(labels, 'H'):,}")
    print(f"A runs          : {count_runs(labels, 'A'):,}")
    print(f"X runs          : {count_runs(labels, 'X'):,}")


def print_comparison(name, result, baseline_labels):
    labels = result["labels"]
    changed = labels != baseline_labels
    metrics = result["metrics"]
    statuses = result["statuses"]

    print("\n" + "=" * 78)
    print(name.upper())
    print("=" * 78)

    print_label_summary(labels)

    print("\nLATENT MEMORY DIAGNOSTICS")
    print("-" * 78)
    print(f"Frames output as X while team is still remembered : "
          f"{metrics['latent_x_frames']:,}")
    print(f"Same-team reactivations                           : "
          f"{metrics['same_team_reactivations']:,}")
    print(f"Opposite-team reactivations                       : "
          f"{metrics['opposite_team_reactivations']:,}")
    print(f"Reentries after memory expired                    : "
          f"{metrics['expired_memory_reentries']:,}")
    print(f"Memory-based switches confirmed                   : "
          f"{metrics['memory_confirmed_switches']:,}")

    print("\nCHANGES VS BASELINE")
    print("-" * 78)
    print(f"Changed frames : {int(changed.sum()):,}")
    print(f"X -> H         : {int(np.sum((baseline_labels == 'X') & (labels == 'H'))):,}")
    print(f"X -> A         : {int(np.sum((baseline_labels == 'X') & (labels == 'A'))):,}")
    print(f"H -> A         : {int(np.sum((baseline_labels == 'H') & (labels == 'A'))):,}")
    print(f"A -> H         : {int(np.sum((baseline_labels == 'A') & (labels == 'H'))):,}")
    print(f"H/A -> X       : {int(np.sum(np.isin(baseline_labels, ['H', 'A']) & (labels == 'X'))):,}")

    print("\nCHANGED FRAMES BY TRACKING CONDITION")
    print("-" * 78)
    for status in (
        "candidate",
        "no_candidate",
        "no_evidence",
        "missing_ball",
        "outside_boundary",
    ):
        count = int(np.sum(changed & (statuses == status)))
        print(f"{status:20}: {count:>9,}")

    episodes = changed_episodes(changed)

    print("\nCONTIGUOUS CHANGE EPISODES")
    print("-" * 78)
    print(f"Changed episodes : {len(episodes):,}")

    if episodes:
        durations = np.array(
            [(end - start) / 25.0 for start, end in episodes],
            dtype=float,
        )
        print(f"Median duration  : {np.median(durations):.3f} s")
        print(f"P95 duration     : {np.percentile(durations, 95):.3f} s")
        print(f"Maximum duration : {np.max(durations):.3f} s")

        print("\nLONGEST 10 CHANGE EPISODES")
        print("-" * 78)
        print(
            f"{'FRAMES':<19} {'DUR(s)':>8} "
            f"{'BASE H/A/X':>15} {'NEW H/A/X':>15} "
            f"{'CANDIDATE %':>13}"
        )

        longest = sorted(
            episodes,
            key=lambda item: item[1] - item[0],
            reverse=True,
        )[:10]

        for start, end in longest:
            old = baseline_labels[start:end]
            new = labels[start:end]
            condition = statuses[start:end]

            old_counts = (
                int(np.sum(old == "H")),
                int(np.sum(old == "A")),
                int(np.sum(old == "X")),
            )
            new_counts = (
                int(np.sum(new == "H")),
                int(np.sum(new == "A")),
                int(np.sum(new == "X")),
            )
            candidate_pct = 100 * np.mean(condition == "candidate")

            print(
                f"{start:>7}-{end - 1:<7} "
                f"{(end - start) / 25.0:>8.2f} "
                f"{str(old_counts):>15} "
                f"{str(new_counts):>15} "
                f"{candidate_pct:>12.1f}%"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Audit separate output hold and latent memory durations."
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

    if any(memory < args.output_hold_frames for _, memory in SCENARIOS):
        parser.error("Every memory duration must be >= output hold duration")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)

    print("\nLATENT POSSESSION MEMORY AUDIT")
    print("=" * 78)
    print(f"Total frames             : {data.n_frames:,}")
    print(f"Output hold              : {args.output_hold_frames} frames")
    print(f"Switch confirmation      : {args.switch_frames} frames")
    print(f"Missing-ball hold        : {args.missing_ball_hold_frames} frames")
    print(f"No-evidence hold         : {args.no_evidence_hold_frames} frames")
    print(f"Control radius           : {args.control_radius:.2f} m")
    print(f"Minimum distance gap     : {args.distance_advantage:.2f} m")

    reference = estimate_possession(
        data,
        features,
        control_radius=args.control_radius,
        min_distance_advantage=args.distance_advantage,
        switch_frames=args.switch_frames,
        missing_ball_hold_frames=args.missing_ball_hold_frames,
        no_evidence_hold_frames=args.no_evidence_hold_frames,
        no_candidate_hold_frames=args.output_hold_frames,
    )

    results = {}

    for name, memory_frames in SCENARIOS:
        print(f"\nRunning {name} ({memory_frames} frames)...")

        result = replay_with_latent_memory(
            data,
            features,
            output_hold_frames=args.output_hold_frames,
            memory_frames=memory_frames,
            control_radius=args.control_radius,
            min_distance_advantage=args.distance_advantage,
            switch_frames=args.switch_frames,
            missing_ball_hold_frames=args.missing_ball_hold_frames,
            no_evidence_hold_frames=args.no_evidence_hold_frames,
        )

        results[memory_frames] = result

    # Safety check: baseline latent replay must exactly match production.
    baseline_result = results[args.output_hold_frames]

    np.testing.assert_array_equal(
        baseline_result["labels"],
        reference.labels,
        err_msg="Baseline labels do not match production estimator.",
    )
    np.testing.assert_array_equal(
        baseline_result["confidence"],
        reference.confidence,
        err_msg="Baseline confidence does not match production estimator.",
    )

    print("\nSAFETY CHECK PASSED")
    print("25-frame latent replay exactly matches production labels and confidence.")

    baseline_labels = reference.labels

    print("\nSCENARIO SUMMARY")
    print("=" * 78)
    print(
        f"{'Scenario':<30} {'H':>9} {'A':>9} {'X':>9} "
        f"{'Transitions':>13} {'Changed':>10}"
    )

    for name, memory_frames in SCENARIOS:
        labels = results[memory_frames]["labels"]

        print(
            f"{name:<30} "
            f"{np.sum(labels == 'H'):>9,} "
            f"{np.sum(labels == 'A'):>9,} "
            f"{np.sum(labels == 'X'):>9,} "
            f"{count_transitions(labels):>13,} "
            f"{np.sum(labels != baseline_labels):>10,}"
        )

    for name, memory_frames in SCENARIOS[1:]:
        print_comparison(
            name,
            results[memory_frames],
            baseline_labels,
        )

    print("\nIMPORTANT")
    print("- Output hold remains unchanged across all scenarios.")
    print("- Only latent memory duration changes.")
    print("- Missing-ball, no-evidence, and boundary behavior remain unchanged.")
    print("- No ground truth was loaded or used.")
    print("- No production source code or submission CSV was modified.")


if __name__ == "__main__":
    main()
