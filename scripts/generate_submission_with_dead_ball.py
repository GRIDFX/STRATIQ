"""Generate a local H/A/D/X submission using the conservative D state layer."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from tactiq.data_loader import load_tracking
from tactiq.features import extract_features
from tactiq.possession import estimate_possession
from tactiq.dead_ball import extract_dead_ball_evidence
from tactiq.dead_ball_state import apply_dead_ball_state
from tactiq.smart_tagging_features import extract_smart_tagging_features
from tactiq.tracking_dynamics import extract_tracking_dynamics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible local TACTIQ AI H/A/D/X CSV."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing the tactical/ folder")
    parser.add_argument("--output", default="outputs/submission_with_dead_ball.csv")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--release-confirm-frames", type=int, default=3)
    parser.add_argument("--restart-release-speed-mps", type=float, default=1.0)
    parser.add_argument(
        "--smart-tagging-xml", default=None,
        help="Optional Smart Tagging XML; defaults to the expected file under data-dir/tactical",
    )
    parser.add_argument(
        "--no-smart-tagging", action="store_true",
        help="Do not use Smart Tagging restart intervals",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output).resolve()

    if args.fps <= 0:
        raise SystemExit("--fps must be positive")

    data = load_tracking(data_dir)
    features = extract_features(data)
    possession = estimate_possession(data, features)
    dynamics = extract_tracking_dynamics(data)
    evidence = extract_dead_ball_evidence(features, dynamics=dynamics)

    smart_tagging = None
    if not args.no_smart_tagging:
        xml_path = (
            Path(args.smart_tagging_xml).resolve()
            if args.smart_tagging_xml
            else data_dir / "tactical" / "BAR-ATH_tactical_smart_tagging.xml"
        )
        if xml_path.is_file():
            smart_tagging = extract_smart_tagging_features(
                xml_path, n_frames=data.n_frames, fps=args.fps
            )
        elif args.smart_tagging_xml:
            raise FileNotFoundError(f"Smart Tagging XML not found: {xml_path}")
        else:
            print("Note: Smart Tagging XML not found; restart-interval context is disabled.")

    final = apply_dead_ball_state(
        possession,
        features,
        evidence,
        smart_tagging=smart_tagging,
        dynamics=dynamics,
        release_confirm_frames=args.release_confirm_frames,
        restart_release_speed_mps=args.restart_release_speed_mps,
    )

    if len(final.labels) != data.n_frames:
        raise RuntimeError("Generated label count does not match tracking frames")
    if not set(final.labels.tolist()).issubset({"H", "A", "D", "X"}):
        raise RuntimeError("Generated unsupported labels")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["frame", "label"])
            writer.writerows(enumerate(final.labels.tolist()))
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"Submission generated: {output_path}")
    print(f"Rows written: {len(final.labels)}")
    print("Columns: frame,label")
    print("Labels supported: H/A/D/X")
    print("No ground-truth labels were read by this generator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
