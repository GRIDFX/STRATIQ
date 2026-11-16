"""Compare latent-memory policies with reconciled event accounting."""
import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


def classify(f, i, radius, gap):
    if not f.ball_visible[i]:
        return "missing", None, 0.0
    if f.ball_outside_boundary[i]:
        return "outside", None, 0.0
    h, a = f.home_nearest_distance[i], f.away_nearest_distance[i]
    if not np.isfinite(h) or not np.isfinite(a):
        return "no_evidence", None, 0.0
    hg, ag = a - h, h - a
    if h <= radius and hg > 0 and hg >= gap:
        return "candidate", "H", min(1.0, hg / max(gap * 3, 1.0))
    if a <= radius and ag > 0 and ag >= gap:
        return "candidate", "A", min(1.0, ag / max(gap * 3, 1.0))
    return "no_candidate", None, 0.0


def replay_legacy_any(data, features, output_hold, memory_frames,
                      radius, gap, switch_frames, missing_hold, evidence_hold):
    """Earlier policy: every clear candidate refreshes the memory timer."""
    n = data.n_frames
    labels = np.full(n, "X", dtype="<U1")
    conf = np.zeros(n, dtype=np.float32)
    current = pending = None
    pending_count = missing_count = evidence_count = age = 0

    for i in range(n):
        status, candidate, candidate_conf = classify(features, i, radius, gap)

        if status == "missing":
            missing_count += 1
            evidence_count = age = 0
            pending, pending_count = None, 0
            if current is not None and missing_count <= missing_hold:
                labels[i], conf[i] = current, 0.25
            elif missing_count > missing_hold:
                current = None
            continue

        missing_count = 0

        if status == "outside":
            current, pending = None, None
            pending_count = evidence_count = age = 0
            continue

        if status == "no_evidence":
            evidence_count += 1
            age = 0
            pending, pending_count = None, 0
            if current is not None and evidence_count <= evidence_hold:
                labels[i], conf[i] = current, 0.20
            elif evidence_count > evidence_hold:
                current = None
            continue

        evidence_count = 0

        if status == "no_candidate":
            age += 1
            pending, pending_count = None, 0
            if current is not None and age <= output_hold:
                labels[i], conf[i] = current, 0.25
            elif current is not None and age > memory_frames:
                current = None
            continue

        # Any clear candidate refreshes the old-team memory.
        age = 0

        if current is None:
            current = candidate
            labels[i], conf[i] = current, candidate_conf
        elif candidate == current:
            pending, pending_count = None, 0
            labels[i], conf[i] = current, candidate_conf
        else:
            if candidate == pending:
                pending_count += 1
            else:
                pending, pending_count = candidate, 1

            if pending_count >= switch_frames:
                current, pending, pending_count = candidate, None, 0

            labels[i] = current
            conf[i] = candidate_conf if current == candidate else 0.2

    return labels, conf


def replay_confirmed_support(data, features, output_hold, memory_frames,
                             radius, gap, switch_frames,
                             missing_hold, evidence_hold):
    """Only support from the remembered team refreshes its memory age."""
    n = data.n_frames
    labels = np.full(n, "X", dtype="<U1")
    conf = np.zeros(n, dtype=np.float32)

    current = pending = None
    pending_count = missing_count = evidence_count = age = 0
    active_run = 0
    run_lengths = []

    metrics = {
        "remembered_but_abstained": 0,
        "opposing_candidate_frames": 0,
        "opposing_runs": 0,
        "confirmed_runs": 0,
        "unconfirmed_runs": 0,
        "interrupted_unconfirmed_runs": 0,
        "confirmed_switches": 0,
        "memory_expirations": 0,
    }

    def close_run(interrupted=False):
        nonlocal active_run
        if active_run == 0:
            return

        run_lengths.append(active_run)
        metrics["opposing_runs"] += 1

        if active_run >= switch_frames:
            metrics["confirmed_runs"] += 1
        else:
            metrics["unconfirmed_runs"] += 1
            if interrupted:
                metrics["interrupted_unconfirmed_runs"] += 1

        active_run = 0

    for i in range(n):
        status, candidate, candidate_conf = classify(
            features, i, radius, gap
        )

        if status == "missing":
            close_run(True)
            missing_count += 1
            evidence_count = age = 0
            pending, pending_count = None, 0

            if current is not None and missing_count <= missing_hold:
                labels[i], conf[i] = current, 0.25
            elif missing_count > missing_hold:
                current = None
            continue

        missing_count = 0

        if status == "outside":
            close_run(True)
            current, pending = None, None
            pending_count = evidence_count = age = 0
            continue

        if status == "no_evidence":
            close_run(True)
            evidence_count += 1
            age = 0
            pending, pending_count = None, 0

            if current is not None and evidence_count <= evidence_hold:
                labels[i], conf[i] = current, 0.20
            elif evidence_count > evidence_hold:
                current = None
            continue

        evidence_count = 0

        if status == "no_candidate":
            close_run(True)
            age += 1
            pending, pending_count = None, 0

            if current is not None and age <= output_hold:
                labels[i], conf[i] = current, 0.25
            elif current is not None:
                metrics["remembered_but_abstained"] += 1

            if current is not None and age > memory_frames:
                current = None
                metrics["memory_expirations"] += 1
            continue

        # A candidate appears after an uncertain period.
        if current is None:
            close_run(True)
            age = 0
            current = candidate
            pending, pending_count = None, 0
            labels[i], conf[i] = current, candidate_conf
            continue

        # Same-team support resets the latent memory clock.
        if candidate == current:
            close_run(False)
            age = 0
            pending, pending_count = None, 0
            labels[i], conf[i] = current, candidate_conf
            continue

        # Opposing candidate does not refresh old-team support.
        age += 1
        active_run += 1
        metrics["opposing_candidate_frames"] += 1

        if candidate == pending:
            pending_count += 1
        else:
            pending, pending_count = candidate, 1

        if pending_count >= switch_frames:
            current = candidate
            labels[i], conf[i] = current, candidate_conf
            metrics["confirmed_switches"] += 1
            close_run(False)
            pending, pending_count = None, 0
            age = 0
            continue

        if age <= output_hold:
            labels[i], conf[i] = current, 0.20
        else:
            labels[i], conf[i] = "X", 0.0
            metrics["remembered_but_abstained"] += 1

        if age > memory_frames:
            current, pending = None, None
            pending_count = 0
            close_run(True)
            metrics["memory_expirations"] += 1

    close_run(True)
    return labels, conf, metrics, run_lengths



def contiguous_run_lengths(labels, target):
    """Return lengths of all contiguous runs for one label."""
    mask = labels == target
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return ends - starts


def summary(title, labels, baseline):
    print(f"\n{title}\n{'-' * 76}")
    total = len(labels)

    for lab, name in (
        ("H", "Barcelona"),
        ("A", "Athletic Club"),
        ("X", "Abstain"),
        ("D", "Dead ball"),
    ):
        count = int(np.sum(labels == lab))
        print(f"{name:15}: {count:>9,} ({count / total * 100:6.2f}%)")

    print("\\nSPELL DIAGNOSTICS — CONTIGUOUS LABEL SEGMENTS")
    print("-" * 76)
    for lab, name in (("H", "Barcelona"), ("A", "Athletic Club"), ("X", "Abstain")):
        lengths = contiguous_run_lengths(labels, lab)
        if len(lengths) == 0:
            print(f"{name:15}: no runs")
            continue

        seconds = lengths / 25.0
        print(
            f"{name:15}: runs={len(lengths):>5,} | "
            f"median={np.median(seconds):>6.2f}s | "
            f"P90={np.percentile(seconds, 90):>6.2f}s | "
            f"max={np.max(seconds):>6.2f}s"
        )

    print(f"Transitions         : {int(np.sum(labels[1:] != labels[:-1])):,}")
    print(f"Changed vs baseline : {int(np.sum(labels != baseline)):,}")

    comparisons = (
        ("X -> H", (baseline == "X") & (labels == "H")),
        ("X -> A", (baseline == "X") & (labels == "A")),
        ("H -> A", (baseline == "H") & (labels == "A")),
        ("A -> H", (baseline == "A") & (labels == "H")),
        ("H/A -> X", np.isin(baseline, ["H", "A"]) & (labels == "X")),
    )

    for name, mask in comparisons:
        print(f"{name:19}: {int(mask.sum()):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument("--switch-frames", type=int, default=8)
    parser.add_argument("--missing-ball-hold-frames", type=int, default=25)
    parser.add_argument("--no-evidence-hold-frames", type=int, default=25)
    parser.add_argument("--output-hold-frames", type=int, default=25)
    parser.add_argument(
        "--memory-frames", type=int, nargs="+", default=[50, 125]
    )
    args = parser.parse_args()

    if args.control_radius <= 0 or args.distance_advantage < 0:
        parser.error("Invalid distance parameters")
    if args.switch_frames < 1 or args.output_hold_frames < 0:
        parser.error("Invalid frame parameters")
    if any(m < args.output_hold_frames for m in args.memory_frames):
        parser.error("Every memory duration must be >= output hold")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)

    params = (
        args.control_radius,
        args.distance_advantage,
        args.switch_frames,
        args.missing_ball_hold_frames,
        args.no_evidence_hold_frames,
    )

    production = estimate_possession(
        data,
        features,
        control_radius=args.control_radius,
        min_distance_advantage=args.distance_advantage,
        switch_frames=args.switch_frames,
        missing_ball_hold_frames=args.missing_ball_hold_frames,
        no_evidence_hold_frames=args.no_evidence_hold_frames,
        no_candidate_hold_frames=args.output_hold_frames,
    )
    baseline = production.labels

    summary("PRODUCTION BASELINE", baseline, baseline)

    # Safety check: one-second legacy replay must match production exactly.
    base_replay, base_conf = replay_legacy_any(
        data, features,
        args.output_hold_frames,
        args.output_hold_frames,
        *params,
    )
    np.testing.assert_array_equal(base_replay, baseline)
    np.testing.assert_array_equal(base_conf, production.confidence)

    print(
        "\nSAFETY CHECK PASSED: 1-second replay matches production "
        "labels and confidence."
    )

    for memory in args.memory_frames:
        print(f"\n{'=' * 76}")
        print(f"MEMORY: {memory} frames ({memory / 25:.2f} seconds)")
        print("=" * 76)

        old, _ = replay_legacy_any(
            data, features,
            args.output_hold_frames,
            memory,
            *params,
        )
        summary("EARLIER POLICY — ANY CANDIDATE RESETS TIMER", old, baseline)

        alt, _, metrics, runs = replay_confirmed_support(
            data, features,
            args.output_hold_frames,
            memory,
            *params,
        )
        summary(
            "ALTERNATIVE — ONLY SAME-TEAM SUPPORT RESETS TIMER",
            alt,
            baseline,
        )

        print("\nRECONCILED OPPOSING-CANDIDATE RUNS")
        print("-" * 76)
        total_runs = metrics["opposing_runs"]
        accounted = metrics["confirmed_runs"] + metrics["unconfirmed_runs"]

        print(f"Total opposing candidate runs : {total_runs:,}")
        print(
            f"Confirmed (>= {args.switch_frames} frames) : "
            f"{metrics['confirmed_runs']:,}"
        )
        print(
            f"Unconfirmed (< {args.switch_frames} frames) : "
            f"{metrics['unconfirmed_runs']:,}"
        )
        print(
            "  Of which interrupted         : "
            f"{metrics['interrupted_unconfirmed_runs']:,}"
        )
        print(f"Accounting check               : {accounted:,} / {total_runs:,}")

        if accounted != total_runs:
            raise RuntimeError("Event accounting does not reconcile")

        print("\nSTATE DIAGNOSTICS")
        for key, value in metrics.items():
            print(f"{key:42}: {value:>9,}")

        if runs:
            print(
                "Median opposing run length (frames): "
                f"{np.median(runs):.1f}"
            )

    print("\nIMPORTANT")
    print("- The one-second replay exactly matched production.")
    print("- Every opposing-candidate run is counted once.")
    print("- Eight-frame persistence is tracking evidence, not possession truth.")
    print("- No ground truth or submission CSV was loaded or created.")
    print("- No production source file was modified.")


if __name__ == "__main__":
    main()
