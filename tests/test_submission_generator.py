import csv

import pytest

from scripts.generate_submission import write_submission


def test_writes_header_and_sequential_frame_ids(tmp_path):
    target = tmp_path / "nested" / "submission.csv"
    result = write_submission(["H", "X", "A"], target, expected_frames=3)

    assert result == target.resolve()
    with result.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows == [
        ["frame", "label"],
        ["0", "H"],
        ["1", "X"],
        ["2", "A"],
    ]


def test_accepts_all_official_label_symbols(tmp_path):
    target = tmp_path / "all_labels.csv"
    write_submission(["H", "A", "D", "X"], target, expected_frames=4)
    assert target.is_file()


def test_rejects_wrong_frame_count(tmp_path):
    with pytest.raises(ValueError, match="Expected 3 labels"):
        write_submission(["H", "A"], tmp_path / "bad.csv", expected_frames=3)


def test_rejects_non_1d_labels(tmp_path):
    with pytest.raises(ValueError, match="one-dimensional"):
        write_submission([["H", "A"]], tmp_path / "bad.csv", expected_frames=1)


def test_rejects_unknown_label(tmp_path):
    with pytest.raises(ValueError, match="Unsupported labels"):
        write_submission(["H", "Z"], tmp_path / "bad.csv", expected_frames=2)


def test_does_not_leave_partial_file_on_invalid_labels(tmp_path):
    target = tmp_path / "bad.csv"
    with pytest.raises(ValueError):
        write_submission(["H", "BAD"], target, expected_frames=2)
    assert not target.exists()
