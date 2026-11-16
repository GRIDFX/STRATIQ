#!/usr/bin/env python3
"""Validate the structure of a STRATIQ submission CSV locally.

This tool reads only the final frame,label CSV supplied by the user. It does
not open match videos, tracking files, ground truth, or other competition data.
It prints structural pass/fail messages, not label distributions or row values.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

ALLOWED_LABELS = {"H", "A", "D", "X"}
EXPECTED_HEADER = ["frame", "label"]
DEFAULT_EXPECTED_ROWS = 199_319


def validate_submission(csv_path: Path, expected_rows: int = DEFAULT_EXPECTED_ROWS) -> list[str]:
    """Return generic validation errors; never echo row contents or labels."""
    errors: list[str] = []
    if expected_rows < 1:
        return ["Expected row count must be a positive integer."]
    if not csv_path.is_file():
        return ["Submission CSV does not exist or is not a regular file."]

    row_count = 0
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return ["Submission CSV is empty."]

            if header != EXPECTED_HEADER:
                errors.append("Header must be exactly: frame,label.")

            for row in reader:
                if not row:
                    errors.append("Submission contains a blank row.")
                    continue
                if len(row) != 2:
                    errors.append("Every data row must contain exactly two columns.")
                    continue

                frame_text, label = row
                try:
                    frame = int(frame_text)
                except ValueError:
                    errors.append("Frame column contains a non-integer value.")
                    continue

                expected_frame = row_count
                if frame != expected_frame:
                    errors.append("Frame values must be consecutive and start at zero.")

                if label not in ALLOWED_LABELS:
                    errors.append("Label column contains a value outside H, A, D, X.")

                row_count += 1

                # Stop accumulating repetitive diagnostics on malformed files.
                if len(errors) >= 10:
                    break

    except (OSError, UnicodeError, csv.Error):
        return ["Submission CSV could not be read as a valid UTF-8 CSV file."]

    if row_count != expected_rows:
        errors.append("Submission has an incorrect number of data rows.")

    return list(dict.fromkeys(errors))


def compare_identical(first: Path, second: Path) -> bool:
    """Check byte-for-byte reproducibility without displaying file contents."""
    try:
        if not first.is_file() or not second.is_file():
            return False
        with first.open("rb") as left, second.open("rb") as right:
            while True:
                left_chunk = left.read(1024 * 1024)
                right_chunk = right.read(1024 * 1024)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    except OSError:
        return False


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="path to the final submission CSV")
    parser.add_argument(
        "--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS,
        help=f"expected number of data rows (default: {DEFAULT_EXPECTED_ROWS})",
    )
    parser.add_argument(
        "--compare", type=Path, default=None,
        help="optional second generated CSV; require byte-for-byte identity",
    )
    args = parser.parse_args(argv)

    errors = validate_submission(args.csv, args.expected_rows)
    if errors:
        print("FAIL: submission structure check did not pass.")
        for error in errors:
            print(f"- {error}")
        return 1

    if args.compare is not None and not compare_identical(args.csv, args.compare):
        print("FAIL: the two generated CSV files are not byte-for-byte identical.")
        return 1

    print("PASS: submission structure is valid.")
    if args.compare is not None:
        print("PASS: both generated CSV files are byte-for-byte identical.")
    print("Note: this checks format and reproducibility only, not possession accuracy or rule compliance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
