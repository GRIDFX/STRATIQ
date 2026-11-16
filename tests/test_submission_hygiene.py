from scripts.check_submission_hygiene import classify_path


def test_tracking_file_in_data_directory_is_flagged():
    risks = classify_path("data/tactical/BAR-ATH_tactical_atd.txt")
    assert risks


def test_video_is_flagged_even_if_renamed_folder():
    risks = classify_path("assets/source_clip.mp4")
    assert "video/tracking/serialized-data file type" in risks


def test_ground_truth_directory_is_flagged():
    risks = classify_path("evaluation/ground_truth/reference.csv")
    assert risks


def test_secret_file_is_flagged():
    risks = classify_path(".env")
    assert "possible credential/secret file" in risks


def test_normal_source_file_is_not_flagged():
    assert classify_path("src/tactiq/possession.py") == []


def test_normal_documentation_is_not_flagged():
    assert classify_path("docs/ALGORITHM.md") == []
