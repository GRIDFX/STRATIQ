"""Descriptive analysis of tracking and Smart Tagging feature interactions.

This module does not use ground truth and does not tune possession parameters.
Team-tag versus nearest-player comparisons are descriptive only.
"""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.smart_tagging_features import extract_smart_tagging_features


def percentage(part, total):
    return 100.0 * part / total if total else 0.0


def analyze(data_dir, xml_path, fps=25):
    tracking = load_tracking(data_dir)
    tracking_features = extract_features(tracking)

    smart = extract_smart_tagging_features(
        xml_path,
        n_frames=tracking.n_frames,
        fps=fps,
    )

    if smart.n_frames != tracking.n_frames:
        raise ValueError("Tracking and Smart Tagging frame counts differ.")

    ball_visible = tracking_features.ball_visible
    tagged = smart.category_active.any(axis=1)
    team_tagged = smart.team_active.any(axis=1)
    restart_tagged = smart.restart_active.any(axis=1)

    visible_count = int(ball_visible.sum())
    missing_count = int((~ball_visible).sum())

    print("\nSTRATIQ - FEATURE INTERACTION ANALYSIS")
    print("=" * 65)
    print(f"Total frames                 : {tracking.n_frames:,}")
    print(f"Frames with visible ball     : {visible_count:,}")
    print(f"Frames with missing ball     : {missing_count:,}")

    print("\n1. SMART TAGGING COVERAGE")
    print("-" * 65)
    print(
        f"Tagged frames                : {tagged.sum():,} "
        f"({percentage(tagged.sum(), tracking.n_frames):.2f}%)"
    )
    print(
        f"Tagged + visible ball       : "
        f"{(tagged & ball_visible).sum():,}"
    )
    print(
        f"Tagged + missing ball       : "
        f"{(tagged & ~ball_visible).sum():,}"
    )
    print(
        f"Visible ball + no category  : "
        f"{(ball_visible & ~tagged).sum():,}"
    )

    print("\n2. CATEGORY OVERLAP")
    print("-" * 65)
    active_categories = smart.category_active.sum(axis=1)
    for count in range(1, int(active_categories.max()) + 1):
        frames = int((active_categories == count).sum())
        if frames:
            print(f"Frames with {count:>2} active categories : {frames:,}")

    print(
        f"Frames with multiple categories : "
        f"{(active_categories > 1).sum():,}"
    )

    print("\n3. TEAM TAG VS NEAREST PLAYER")
    print("-" * 65)

    home_distance = tracking_features.home_nearest_distance
    away_distance = tracking_features.away_nearest_distance

    valid = (
        ball_visible
        & team_tagged
        & np.isfinite(home_distance)
        & np.isfinite(away_distance)
    )

    home_tag = smart.team_active[:, 0]
    away_tag = smart.team_active[:, 1]

    exactly_one_team = home_tag ^ away_tag
    comparable = valid & exactly_one_team
    home_nearest = home_distance < away_distance
    away_nearest = away_distance < home_distance
    non_tie = comparable & (home_nearest | away_nearest)

    matches = (
        (home_tag & home_nearest)
        | (away_tag & away_nearest)
    ) & non_tie

    print(f"Frames with visible ball + team tag : {valid.sum():,}")
    print(f"Frames with exactly one team tag    : {comparable.sum():,}")
    print(f"Comparable frames excluding ties    : {non_tie.sum():,}")
    print(f"Same team as nearest player         : {matches.sum():,}")
    print(
        f"Descriptive agreement               : "
        f"{percentage(matches.sum(), non_tie.sum()):.2f}%"
    )
    print(
        "Note: this is NOT possession accuracy. Team tags describe "
        "tactical context, while nearest distance is geometric proximity."
    )

    print("\n4. RESTART CONTEXT")
    print("-" * 65)
    print(f"Frames with restart annotation : {restart_tagged.sum():,}")
    print(
        f"Restart + visible ball         : "
        f"{(restart_tagged & ball_visible).sum():,}"
    )
    print(
        f"Restart + missing ball         : "
        f"{(restart_tagged & ~ball_visible).sum():,}"
    )

    print("\n5. HALF COVERAGE")
    print("-" * 65)
    for index, half in enumerate(smart.half_names):
        half_mask = smart.half_active[:, index]
        print(
            f"{half:<15} frames={half_mask.sum():>8,} "
            f"visible_ball={(half_mask & ball_visible).sum():>8,} "
            f"tagged={(half_mask & tagged).sum():>8,}"
        )

    print("\n6. TEAM TAG COVERAGE")
    print("-" * 65)
    for index, team in enumerate(smart.team_names):
        mask = smart.team_active[:, index]
        print(
            f"{team:<20} frames={mask.sum():>8,} "
            f"visible_ball={(mask & ball_visible).sum():>8,}"
        )

    print("\nAnalysis completed.")
    print("No ground truth was loaded or used.")
    print("No possession labels or model parameters were changed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="../more-than-a-hack-2027/data",
    )
    parser.add_argument(
        "--xml",
        default=(
            "../more-than-a-hack-2027/data/tactical/"
            "BAR-ATH_tactical_smart_tagging.xml"
        ),
    )
    parser.add_argument("--fps", type=float, default=25)
    args = parser.parse_args()

    analyze(args.data_dir, args.xml, args.fps)


if __name__ == "__main__":
    main()
