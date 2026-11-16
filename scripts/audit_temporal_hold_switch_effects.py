"""Inspect label changes caused by longer no-candidate temporal holds.

Full estimator replay is used. No ground truth is loaded and no
production parameters or submission labels are modified.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


SCENARIOS = [
    ("2 seconds", 50),
    ("3 seconds", 75),
    ("5 seconds", 125),
]


def contiguous_runs(mask):
    edges = np.diff(
        np.concatenate(([False], mask, [False])).astype(np.int8)
    )
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def no_candidate_mask(features):
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

    return valid & ~home_candidate & ~away_candidate


def pair_count(old, new, old_label, new_label):
    return int(np.sum((old == old_label) & (new == new_label)))


def analyze_scenario(name, hold_frames, baseline, replay, no_candidate, fps):
    changed = baseline != replay
    runs = contiguous_runs(changed)

    x_to_h = pair_count(baseline, replay, "X", "H")
    x_to_a = pair_count(baseline, replay, "X", "A")
    h_to_a = pair_count(baseline, replay, "H", "A")
    a_to_h = pair_count(baseline, replay, "A", "H")

    print(f"\n{'=' * 100}")
    print(f"{name.upper()} — {hold_frames} FRAMES")
    print("=" * 100)

    print("\nFRAME-LEVEL LABEL CHANGES")
    print("-" * 100)
    print(f"Changed frames       : {int(changed.sum()):,}")
    print(f"X -> H               : {x_to_h:,}")
    print(f"X -> A               : {x_to_a:,}")
    print(f"H -> A               : {h_to_a:,}")
    print(f"A -> H               : {a_to_h:,}")
    print(f"Other label changes  : {int(changed.sum()) - x_to_h - x_to_a - h_to_a - a_to_h:,}")

    print("\nCONTIGUOUS CHANGE EPISODES")
    print("-" * 100)

    lengths = np.asarray([end - start for start, end in runs])

    print(f"Changed episodes : {len(runs):,}")

    if len(lengths):
        print(f"Median duration  : {np.median(lengths) / fps:.3f} s")
        print(f"P95 duration     : {np.percentile(lengths, 95) / fps:.3f} s")
        print(f"Maximum duration : {np.max(lengths) / fps:.3f} s")

    print("\nLONGEST 15 CHANGE EPISODES")
    print("-" * 100)
    print(
        "FRAMES             DUR(s)   BASELINE H/A/X   REPLAY H/A/X   "
        "NO-CANDIDATE %   FIRST CHANGE"
    )

    for start, end in sorted(
        runs,
        key=lambda item: item[1] - item[0],
        reverse=True,
    )[:15]:
        old = baseline[start:end]
        new = replay[start:end]

        old_counts = tuple(int(np.sum(old == label)) for label in ("H", "A", "X"))
        new_counts = tuple(int(np.sum(new == label)) for label in ("H", "A", "X"))

        nc_pct = 100 * float(no_candidate[start:end].mean())

        first_change = f"{baseline[start]}->{replay[start]}"

        print(
            f"{start:>7}-{end:<7} "
            f"{(end-start)/fps:>7.2f} "
            f"{str(old_counts):>16} "
            f"{str(new_counts):>15} "
            f"{nc_pct:>15.1f}% "
            f"{first_change:>14}"
        )

    # Inspect whether opposite-team changes happen in the immediate
    # neighborhood of a no-candidate interval.
    opposite = (
        ((baseline == "H") & (replay == "A"))
        | ((baseline == "A") & (replay == "H"))
    )

    print("\nOPPOSITE-TEAM CHANGE CONTEXT")
    print("-" * 100)
    print(f"Opposite-team changed frames : {int(opposite.sum()):,}")
    print(
        "Inside no-candidate frames  : "
        f"{int(np.sum(opposite & no_candidate)):,}"
    )
    print(
        "Outside no-candidate frames : "
        f"{int(np.sum(opposite & ~no_candidate)):,}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    parser.add_argument("--switch-frames", type=int, default=8)
    parser.add_argument("--missing-ball-hold-frames", type=int, default=25)
    parser.add_argument("--no-evidence-hold-frames", type=int, default=25)
    args = parser.parse_args()

    if args.fps <= 0 or args.control_radius <= 0:
        parser.error("fps and control-radius must be positive")
    if args.distance_advantage < 0 or args.switch_frames < 1:
        parser.error("Invalid distance advantage or switch frames")

    print("Loading tracking...")
    data = load_tracking(args.data_dir)

    print("Extracting features...")
    features = extract_features(data)

    no_candidate = no_candidate_mask(features)

    common = dict(
        control_radius=args.control_radius,
        min_distance_advantage=args.distance_advantage,
        switch_frames=args.switch_frames,
        missing_ball_hold_frames=args.missing_ball_hold_frames,
        no_evidence_hold_frames=args.no_evidence_hold_frames,
    )

    print("Running 1-second baseline replay...")
    baseline = estimate_possession(
        data,
        features,
        no_candidate_hold_frames=25,
        **common,
    ).labels

    for name, hold_frames in SCENARIOS:
        print(f"Running {name} replay...")
        replay = estimate_possession(
            data,
            features,
            no_candidate_hold_frames=hold_frames,
            **common,
        ).labels

        analyze_scenario(
            name,
            hold_frames,
            baseline,
            replay,
            no_candidate,
            args.fps,
        )

    print("\nNo ground truth was loaded.")
    print("No production source code or submission labels were modified.")


if __name__ == "__main__":
    main()
