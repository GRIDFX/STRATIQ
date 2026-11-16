"""Descriptive audit of possible dead-ball signals.

This script does not generate possession or dead-ball labels.
It only compares tracking observations with Smart Tagging restart intervals.
"""

import argparse
from pathlib import Path

import numpy as np

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.smart_tagging import load_smart_tagging
from tactiq.smart_tagging_features import extract_smart_tagging_features
from tactiq.tracking_dynamics import extract_tracking_dynamics


def report_mask(name, mask, features, ball_speed):
    count = int(mask.sum())
    total = len(mask)

    visible = mask & features.ball_visible
    outside = mask & features.ball_outside_boundary
    inside = mask & features.ball_inside_boundary
    missing = mask & ~features.ball_visible

    speed_valid = mask & np.isfinite(ball_speed)
    speed_values = ball_speed[speed_valid]

    print(f"\n{name}")
    print("-" * 60)
    print(f"Frames                    : {count:,} ({count / total:.2%})")
    print(f"Ball visible              : {visible.sum():,}")
    print(f"Ball missing              : {missing.sum():,}")
    print(f"Ball inside boundary      : {inside.sum():,}")
    print(f"Ball outside boundary     : {outside.sum():,}")
    print(f"Ball speed available      : {speed_valid.sum():,}")

    if len(speed_values):
        print(f"Mean ball speed           : {np.mean(speed_values):.3f} m/s")
        print(f"Median ball speed         : {np.median(speed_values):.3f} m/s")
        print(f"95th percentile           : {np.percentile(speed_values, 95):.3f} m/s")


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
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    xml_path = Path(args.xml)

    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    print("Loading tracking data...")
    data = load_tracking(data_dir)

    print("Extracting tracking features...")
    features = extract_features(data)

    print("Extracting Smart Tagging features...")
    smart = extract_smart_tagging_features(
        xml_path,
        n_frames=data.n_frames,
        fps=args.fps,
    )

    print("Extracting reported ball speed...")
    dynamics = extract_tracking_dynamics(data)
    ball_speed = dynamics.ball_speed

    if smart.n_frames != data.n_frames:
        raise ValueError("Smart Tagging and tracking frame counts differ.")

    restart_mask = smart.restart_active.any(axis=1)
    set_pieces_index = smart.category_names.index("SET PIECES")
    set_pieces_mask = smart.category_active[:, set_pieces_index]

    outside_mask = features.ball_outside_boundary
    visible_mask = features.ball_visible
    missing_mask = ~visible_mask

    print("\n" + "=" * 65)
    print("STRATIQ - DEAD BALL CANDIDATE AUDIT")
    print("=" * 65)
    print(f"Total tracking frames       : {data.n_frames:,}")
    print(f"Smart Tagging frame count   : {smart.n_frames:,}")
    print(f"Restart categories          : {len(smart.restart_names)}")

    print("\nSMART TAGGING RESTART FRAME COUNTS")
    print("-" * 60)

    for index, name in enumerate(smart.restart_names):
        mask = smart.restart_active[:, index]
        print(f"{name:<30} {mask.sum():>10,} frames")

    print("\nOVERALL FRAME RELATIONSHIPS")
    print("-" * 60)
    print(f"Frames with SET PIECES      : {set_pieces_mask.sum():,}")
    print(f"Frames with restart type    : {restart_mask.sum():,}")
    print(f"Frames with visible ball    : {visible_mask.sum():,}")
    print(f"Frames with missing ball    : {missing_mask.sum():,}")
    print(f"Frames outside boundary     : {outside_mask.sum():,}")

    print("\nINTERSECTIONS")
    print("-" * 60)

    intersections = [
        ("Restart AND ball visible", restart_mask & visible_mask),
        ("Restart AND ball missing", restart_mask & missing_mask),
        ("Restart AND ball outside", restart_mask & outside_mask),
        ("SET PIECES AND ball visible", set_pieces_mask & visible_mask),
        ("SET PIECES AND ball missing", set_pieces_mask & missing_mask),
        ("Outside boundary AND visible", outside_mask & visible_mask),
        ("Outside boundary AND restart", outside_mask & restart_mask),
    ]

    for name, mask in intersections:
        print(f"{name:<35} {mask.sum():>10,}")

    report_mask(
        "ALL RESTART-TAGGED FRAMES",
        restart_mask,
        features,
        ball_speed,
    )

    report_mask(
        "ALL SET PIECES FRAMES",
        set_pieces_mask,
        features,
        ball_speed,
    )

    report_mask(
        "ALL OUTSIDE-BOUNDARY FRAMES",
        outside_mask,
        features,
        ball_speed,
    )

    print("\nRESTART TYPE DETAILS")
    print("=" * 65)

    for index, name in enumerate(smart.restart_names):
        mask = smart.restart_active[:, index]
        if mask.any():
            report_mask(name, mask, features, ball_speed)

    print("\nINTERPRETATION LIMITS")
    print("-" * 60)
    print("Restart annotations are contextual features, not D labels.")
    print("A missing ball does not prove that the ball is out of play.")
    print("An outside-boundary coordinate is not automatically a dead ball.")
    print("Ball speed alone does not establish whether play has stopped.")
    print("No ground-truth labels or estimator parameters were used.")
    print("No possession or dead-ball labels were generated.")


if __name__ == "__main__":
    main()
