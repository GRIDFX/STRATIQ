"""Audit possession candidate eligibility without changing the estimator."""

import argparse

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


def report(name, mask, denominator):
    count = int(mask.sum())
    percentage = count / denominator * 100 if denominator else 0.0
    print(f"{name:<43}: {count:>9,} ({percentage:6.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="../more-than-a-hack-2027/data",
    )
    parser.add_argument("--control-radius", type=float, default=2.5)
    parser.add_argument("--distance-advantage", type=float, default=1.0)
    args = parser.parse_args()

    data = load_tracking(args.data_dir)
    features = extract_features(data)
    result = estimate_possession(
        data,
        features,
        control_radius=args.control_radius,
        min_distance_advantage=args.distance_advantage,
    )

    home = np.asarray(features.home_nearest_distance)
    away = np.asarray(features.away_nearest_distance)
    visible = np.asarray(features.ball_visible, dtype=bool)
    outside = np.asarray(features.ball_outside_boundary, dtype=bool)
    labels = result.labels

    valid = (
        visible
        & ~outside
        & np.isfinite(home)
        & np.isfinite(away)
    )

    home_advantage = away - home
    away_advantage = home - away

    home_candidate = (
        valid
        & (home <= args.control_radius)
        & (home_advantage > 0)
        & (home_advantage >= args.distance_advantage)
    )

    away_candidate = (
        valid
        & (away <= args.control_radius)
        & (away_advantage > 0)
        & (away_advantage >= args.distance_advantage)
    )

    no_candidate = valid & ~home_candidate & ~away_candidate

    both_far = (
        no_candidate
        & (home > args.control_radius)
        & (away > args.control_radius)
    )

    both_near_no_advantage = (
        no_candidate
        & (home <= args.control_radius)
        & (away <= args.control_radius)
    )

    one_near_but_not_clear = (
        no_candidate
        & ~both_far
        & ~both_near_no_advantage
    )

    x_mask = labels == "X"
    x_valid = x_mask & valid

    print("\n" + "=" * 65)
    print("TACTIQ AI - POSSESSION CANDIDATE AUDIT")
    print("=" * 65)
    print(f"Control radius       : {args.control_radius:.2f} m")
    print(f"Distance advantage   : {args.distance_advantage:.2f} m")
    print(f"Total frames         : {len(labels):,}")
    print(f"Valid visible frames : {valid.sum():,}")
    print(f"X with valid evidence: {x_valid.sum():,}")

    print("\nCANDIDATE ELIGIBILITY - ALL VALID FRAMES")
    report("Barcelona candidate", home_candidate, valid.sum())
    report("Athletic Club candidate", away_candidate, valid.sum())
    report("No eligible candidate", no_candidate, valid.sum())

    print("\nREASONS FOR NO CANDIDATE - ALL VALID FRAMES")
    report("Both teams farther than control radius", both_far, valid.sum())
    report("Both near, advantage insufficient", both_near_no_advantage, valid.sum())
    report("Only one near, but not clear enough", one_near_but_not_clear, valid.sum())

    print("\nCANDIDATE ELIGIBILITY - X WITH VALID EVIDENCE")
    denominator = int(x_valid.sum())
    report("Barcelona candidate", x_valid & home_candidate, denominator)
    report("Athletic Club candidate", x_valid & away_candidate, denominator)
    report("No eligible candidate", x_valid & no_candidate, denominator)

    print("\nREASONS FOR NO CANDIDATE - X WITH VALID EVIDENCE")
    report("Both teams farther than control radius", x_valid & both_far, denominator)
    report("Both near, advantage insufficient", x_valid & both_near_no_advantage, denominator)
    report("Only one near, but not clear enough", x_valid & one_near_but_not_clear, denominator)

    print("\nDISTANCE SUMMARY - X WITH VALID EVIDENCE")
    if denominator:
        print(f"Home distance mean   : {home[x_valid].mean():.2f} m")
        print(f"Away distance mean   : {away[x_valid].mean():.2f} m")
        print(
            "Absolute distance gap: "
            f"{np.abs(home[x_valid] - away[x_valid]).mean():.2f} m"
        )

    print("\nIMPORTANT")
    print("- This audit measures candidate eligibility, not accuracy.")
    print("- It does not change estimator parameters or labels.")
    print("- A valid candidate during X may indicate a lost temporal state.")
    print("- Do not interpret proximity alone as confirmed possession.")


if __name__ == "__main__":
    main()
