"""Audit stale latent possession state under intermittent opposing candidates.

Compares:
1) Production baseline.
2) Earlier latent-memory policy (any candidate resets no-candidate age).
3) Safer diagnostic policy: only a same-team candidate refreshes memory age;
   opposing candidates count as unresolved time, and output abstains after the
   output hold while the old state is still retained internally.

This is a diagnostic experiment only. No ground truth or CSV is used.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession
from audit_latent_possession_memory import replay_with_latent_memory


def classify_frames(features, control_radius, min_distance_advantage):
    n = len(features.ball_visible)
    statuses = np.full(n, "no_candidate", dtype="<U20")
    candidates = np.full(n, "", dtype="<U1")
    advantages = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        if not features.ball_visible[i]:
            statuses[i] = "missing_ball"
            continue
        if features.ball_outside_boundary[i]:
            statuses[i] = "outside_boundary"
            continue

        home = features.home_nearest_distance[i]
        away = features.away_nearest_distance[i]
        if not np.isfinite(home) or not np.isfinite(away):
            statuses[i] = "no_evidence"
            continue

        h_adv = float(away - home)
        a_adv = float(home - away)
        if home <= control_radius and h_adv > 0 and h_adv >= min_distance_advantage:
            statuses[i] = "candidate"
            candidates[i] = "H"
            advantages[i] = h_adv
        elif away <= control_radius and a_adv > 0 and a_adv >= min_distance_advantage:
            statuses[i] = "candidate"
            candidates[i] = "A"
            advantages[i] = a_adv

    return statuses, candidates, advantages


def replay_confirmed_support_clock(
    features,
    statuses,
    candidates,
    output_hold_frames,
    memory_frames,
    switch_frames,
    missing_ball_hold_frames,
    no_evidence_hold_frames,
    candidate_confidences,
):
    """Replay with a timer refreshed only by positive support for current team."""
    n = len(statuses)
    labels = np.full(n, "X", dtype="<U1")
    confidence = np.zeros(n, dtype=np.float32)

    current_team = None
    pending_team = None
    pending_count = 0
    support_age = 0
    missing_count = 0
    no_evidence_count = 0

    metrics = {
        "remembered_but_abstained": 0,
        "opposing_candidate_frames_while_remembered": 0,
        "opposing_candidate_runs": 0,
        "opposing_candidate_runs_confirmed": 0,
        "opposing_candidate_runs_unconfirmed": 0,
        "memory_expirations": 0,
        "confirmed_switches": 0,
    }

    in_opposing_run = False
    opposing_run_confirmed = False

    for i in range(n):
        status = statuses[i]
        candidate = candidates[i]

        if status == "missing_ball":
            missing_count += 1
            no_evidence_count = 0
            pending_team = None
            pending_count = 0
            in_opposing_run = False
            opposing_run_confirmed = False

            if current_team is not None and missing_count <= missing_ball_hold_frames:
                labels[i] = current_team
                confidence[i] = 0.25
            else:
                if current_team is not None and missing_count > missing_ball_hold_frames:
                    current_team = None
                    support_age = 0
                labels[i] = "X"
            continue

        missing_count = 0

        if status == "outside_boundary":
            current_team = None
            pending_team = None
            pending_count = 0
            support_age = 0
            no_evidence_count = 0
            in_opposing_run = False
            opposing_run_confirmed = False
            labels[i] = "X"
            continue

        if status == "no_evidence":
            no_evidence_count += 1
            pending_team = None
            pending_count = 0
            in_opposing_run = False
            opposing_run_confirmed = False
            if current_team is not None and no_evidence_count <= no_evidence_hold_frames:
                labels[i] = current_team
                confidence[i] = 0.20
            else:
                labels[i] = "X"
                if current_team is not None and no_evidence_count > no_evidence_hold_frames:
                    current_team = None
                    support_age = 0
            continue

        no_evidence_count = 0

        if status == "no_candidate":
            support_age += 1
            pending_team = None
            pending_count = 0
            if in_opposing_run:
                if opposing_run_confirmed:
                    metrics["opposing_candidate_runs_confirmed"] += 1
                else:
                    metrics["opposing_candidate_runs_unconfirmed"] += 1
                in_opposing_run = False
                opposing_run_confirmed = False

            if current_team is not None and support_age <= output_hold_frames:
                labels[i] = current_team
                confidence[i] = 0.25
            else:
                labels[i] = "X"
                if current_team is not None:
                    metrics["remembered_but_abstained"] += 1
                if current_team is not None and support_age > memory_frames:
                    current_team = None
                    support_age = 0
                    metrics["memory_expirations"] += 1
            continue

        # Clear candidate frame.
        if current_team is None:
            current_team = candidate
            pending_team = None
            pending_count = 0
            support_age = 0
            labels[i] = current_team
            confidence[i] = candidate_confidences[i]
            in_opposing_run = False
            opposing_run_confirmed = False
            continue

        if candidate == current_team:
            support_age = 0
            pending_team = None
            pending_count = 0
            if in_opposing_run:
                if opposing_run_confirmed:
                    metrics["opposing_candidate_runs_confirmed"] += 1
                else:
                    metrics["opposing_candidate_runs_unconfirmed"] += 1
                in_opposing_run = False
                opposing_run_confirmed = False
            labels[i] = current_team
            confidence[i] = candidate_confidences[i]
            continue

        # Opposing candidate is not positive evidence for the remembered team.
        support_age += 1
        metrics["opposing_candidate_frames_while_remembered"] += 1
        if not in_opposing_run:
            metrics["opposing_candidate_runs"] += 1
            in_opposing_run = True
            opposing_run_confirmed = False

        if candidate == pending_team:
            pending_count += 1
        else:
            pending_team = candidate
            pending_count = 1

        # If the old memory has expired, initialize from the current candidate.
        if support_age > memory_frames:
            current_team = candidate
            support_age = 0
            pending_team = None
            pending_count = 0
            labels[i] = current_team
            confidence[i] = candidate_confidences[i]
            metrics["memory_expirations"] += 1
            if in_opposing_run:
                opposing_run_confirmed = True
            continue

        if pending_count >= switch_frames:
            current_team = candidate
            support_age = 0
            pending_team = None
            pending_count = 0
            labels[i] = current_team
            confidence[i] = candidate_confidences[i]
            metrics["confirmed_switches"] += 1
            opposing_run_confirmed = True
            continue

        # Old team has no current positive support after output hold: abstain.
        if support_age <= output_hold_frames:
            labels[i] = current_team
            confidence[i] = 0.20
        else:
            labels[i] = "X"
            metrics["remembered_but_abstained"] += 1

    if in_opposing_run:
        if opposing_run_confirmed:
            metrics["opposing_candidate_runs_confirmed"] += 1
        else:
            metrics["opposing_candidate_runs_unconfirmed"] += 1

    return labels, confidence, metrics


def transitions(labels):
    return int(np.sum(labels[1:] != labels[:-1]))


def summarize(name, labels, baseline, metrics=None):
    n = len(labels)
    h = int(np.sum(labels == "H"))
    a = int(np.sum(labels == "A"))
    x = int(np.sum(labels == "X"))
    changed = labels != baseline
    print(f"\n{name}")
    print("-" * 78)
    print(f"H : {h:>9,} ({h / n * 100:6.2f}%)")
    print(f"A : {a:>9,} ({a / n * 100:6.2f}%)")
    print(f"X : {x:>9,} ({x / n * 100:6.2f}%)")
    print(f"Transitions         : {transitions(labels):,}")
    print(f"Changed vs baseline : {int(changed.sum()):,}")
    print(f"X -> H              : {int(np.sum((baseline == 'X') & (labels == 'H'))):,}")
    print(f"X -> A              : {int(np.sum((baseline == 'X') & (labels == 'A'))):,}")
    print(f"H -> A              : {int(np.sum((baseline == 'H') & (labels == 'A'))):,}")
    print(f"A -> H              : {int(np.sum((baseline == 'A') & (labels == 'H'))):,}")
    print(f"H/A -> X            : {int(np.sum(np.isin(baseline, ['H', 'A']) & (labels == 'X'))):,}")
    if metrics:
        print("\nSTATE DIAGNOSTICS")
        for key, value in metrics.items():
            print(f"{key:48}: {value:,}")


def main():
    parser = argparse.ArgumentParser(description="Audit latent-memory expiry and stale-state behavior.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument("--switch-frames", type=int, default=8)
    parser.add_argument("--missing-ball-hold-frames", type=int, default=25)
    parser.add_argument("--no-evidence-hold-frames", type=int, default=25)
    parser.add_argument("--output-hold-frames", type=int, default=25)
    args = parser.parse_args()

    if args.control_radius <= 0 or args.distance_advantage < 0:
        parser.error("Invalid distance thresholds")
    if args.switch_frames < 1 or args.output_hold_frames < 0:
        parser.error("Invalid frame thresholds")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)
    print("Extracting features...")
    features = extract_features(data)

    common = dict(
        control_radius=args.control_radius,
        min_distance_advantage=args.distance_advantage,
        switch_frames=args.switch_frames,
        missing_ball_hold_frames=args.missing_ball_hold_frames,
        no_evidence_hold_frames=args.no_evidence_hold_frames,
    )

    print("Running production baseline...")
    baseline_result = estimate_possession(
        data, features,
        no_candidate_hold_frames=args.output_hold_frames,
        **common,
    )
    baseline = baseline_result.labels

    print("Running 1-second latent replay for baseline safety check...")
    latent_baseline = replay_with_latent_memory(
        data, features,
        output_hold_frames=args.output_hold_frames,
        memory_frames=args.output_hold_frames,
        **common,
    )

    print("Running earlier any-candidate-refresh policy (2-second memory)...")
    older = replay_with_latent_memory(
        data, features,
        output_hold_frames=args.output_hold_frames,
        memory_frames=50,
        **common,
    )

    statuses, candidates, advantages = classify_frames(
        features, args.control_radius, args.distance_advantage
    )
    candidate_confidences = np.zeros(len(statuses), dtype=np.float32)
    valid = statuses == "candidate"
    candidate_confidences[valid] = np.minimum(
        1.0, advantages[valid] / max(args.distance_advantage * 3, 1.0)
    )

    print("Running confirmed-support-clock policy...")
    safer_labels, safer_confidence, metrics = replay_confirmed_support_clock(
        features,
        statuses,
        candidates,
        output_hold_frames=args.output_hold_frames,
        memory_frames=50,
        switch_frames=args.switch_frames,
        missing_ball_hold_frames=args.missing_ball_hold_frames,
        no_evidence_hold_frames=args.no_evidence_hold_frames,
        candidate_confidences=candidate_confidences,
    )

    np.testing.assert_array_equal(
        latent_baseline["labels"], baseline,
        err_msg="1-second latent replay must match production labels.",
    )
    np.testing.assert_array_equal(
        latent_baseline["confidence"], baseline_result.confidence,
        err_msg="1-second latent replay must match production confidence.",
    )

    print(
        "\nSAFETY CHECK PASSED: 1-second latent replay exactly matches "
        "production labels and confidence."
    )
    summarize("PRODUCTION BASELINE — 1 SECOND", baseline, baseline)
    summarize(
        "EARLIER POLICY — 2-SECOND MEMORY / ANY CANDIDATE RESETS TIMER",
        older["labels"], baseline, older["metrics"],
    )
    summarize("ALTERNATIVE — 2-SECOND MEMORY / ONLY SAME-TEAM SUPPORT RESETS TIMER", safer_labels, baseline, metrics)

    print("\nIMPORTANT")
    print("- Alternative policy is diagnostic only; it is not production code.")
    print("- Opposing candidate frames do not refresh old-team memory in the alternative.")
    print("- Output becomes X after output_hold_frames when old-team support is absent.")
    print("- Missing-ball, no-evidence, and outside-boundary handling follow production rules.")
    print("- No ground truth or submission CSV was loaded or created.")


if __name__ == "__main__":
    main()
