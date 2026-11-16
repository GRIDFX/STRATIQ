"""Generate a frame-level possession CSV from local Metrica tracking data.

This entry point intentionally uses the base H/A/X possession estimator.
Use ``generate_submission_with_dead_ball.py`` for the separate D state layer.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

EXPECTED_FRAMES = 199_319

ALLOWED_LABELS = frozenset({"H", "A", "D", "X"})


def write_submission(
    labels: Sequence[str] | np.ndarray,
    output_path: str | Path,
    expected_frames: int = EXPECTED_FRAMES,
) -> Path:
    """Write and structurally validate a frame,label CSV atomically."""
    values = np.asarray(labels)

    if values.ndim != 1:
        raise ValueError("labels must be a one-dimensional sequence")
    if len(values) != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} labels, received {len(values)}"
        )

    normalized = [str(value) for value in values.tolist()]
    invalid = sorted(set(normalized) - ALLOWED_LABELS)
    if invalid:
        raise ValueError(f"Unsupported labels: {', '.join(invalid)}")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")

    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("frame", "label"))
            writer.writerows(enumerate(normalized))
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination


def _is_inside(child: Path, parent: Path) -> bool:
    """Return whether child is parent or one of its descendants."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def build_submission(data_dir: str | Path, output_path: str | Path) -> Path:
    """Run the current production path and write the generated CSV."""
    source_dir = Path(data_dir).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {source_dir}")
    if _is_inside(destination, source_dir):
        raise ValueError(
            "Output must be outside the competition data directory; "
            "keep match assets local and separate from submission code."
        )

    # Import the project pipeline lazily so CSV writer tests can run without
    # loading the match-processing package or any match data.
    from tactiq.data_loader import N_FRAMES, load_tracking
    from tactiq.features import extract_features
    from tactiq.possession import estimate_possession

    tracking = load_tracking(source_dir)
    if tracking.n_frames != N_FRAMES or N_FRAMES != EXPECTED_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_FRAMES} tracking frames, got {tracking.n_frames}"
        )

    features = extract_features(tracking)
    possession = estimate_possession(tracking, features)

    # estimate_possession currently produces H/A/X. D is an allowed CSV label,
    # but dead_ball.py only provides candidate evidence and is not promoted here.
    return write_submission(
        possession.labels,
        destination,
        expected_frames=EXPECTED_FRAMES,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate STRATIQ possession submission from local data."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the directory containing the tactical/ folder.",
    )
    parser.add_argument(
        "--output",
        default="outputs/submission.csv",
        help="Output CSV path (default: outputs/submission.csv).",
    )
    args = parser.parse_args(argv)

    try:
        output = build_submission(args.data_dir, args.output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(f"Submission generated: {output}")
    print(f"Rows written: {EXPECTED_FRAMES}")
    print("Columns: frame,label")
    print("Labels emitted by current estimator: H/A/X")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
