import csv
from pathlib import Path

from scripts.validate_submission_local import compare_identical, validate_submission


def write_csv(path: Path, rows, header=("frame", "label")):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_accepts_valid_csv(tmp_path):
    path = tmp_path / "valid.csv"
    write_csv(path, [(0, "H"), (1, "A"), (2, "D"), (3, "X")])
    assert validate_submission(path, expected_rows=4) == []


def test_rejects_wrong_header(tmp_path):
    path = tmp_path / "bad_header.csv"
    write_csv(path, [(0, "H")], header=("index", "team"))
    assert any("Header" in error for error in validate_submission(path, 1))


def test_rejects_wrong_row_count(tmp_path):
    path = tmp_path / "short.csv"
    write_csv(path, [(0, "H")])
    assert any("number of data rows" in error for error in validate_submission(path, 2))


def test_rejects_invalid_label(tmp_path):
    path = tmp_path / "bad_label.csv"
    write_csv(path, [(0, "HOME")])
    assert any("Label column" in error for error in validate_submission(path, 1))


def test_rejects_nonconsecutive_frames(tmp_path):
    path = tmp_path / "bad_frames.csv"
    write_csv(path, [(0, "H"), (2, "A")])
    assert any("consecutive" in error for error in validate_submission(path, 2))


def test_byte_comparison(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    write_csv(first, [(0, "H")])
    write_csv(second, [(0, "H")])
    assert compare_identical(first, second)
    second.write_text("frame,label\n0,A\n", encoding="utf-8")
    assert not compare_identical(first, second)
