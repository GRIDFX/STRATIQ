#!/usr/bin/env python3
"""Path-only preflight for keeping restricted match assets out of a public repo.

This script inspects file *paths only*. It does not open, parse, hash, summarize,
or print the contents of match data. It is a conservative warning gate, not a
legal/compliance certification and not a substitute for reviewing the event's
Terms and Conditions.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Directory names that should never appear in the public project repository
# when they contain competition-match material.
RESTRICTED_DIR_NAMES = {
    "match_data",
    "raw_data",
    "private_data",
    "datasets",
    "ground_truth",
    "groundtruth",
    "private_gt",
}

# Extensions strongly associated with licensed video/tracking or serialized
# match-derived arrays. CSV/XML are handled contextually below to reduce noise.
RESTRICTED_SUFFIXES = {
    ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm",
    ".atd", ".parquet", ".feather", ".npz", ".npy", ".pkl", ".pickle",
}

RESTRICTED_NAME_PARTS = (
    "ground_truth", "groundtruth", "private_gt", "full_match_gt",
    "possession_submission", "match_tracking", "match_video",
)

SECRET_NAMES = {
    ".env", ".env.local", ".env.production", "secrets.json",
    "credentials.json", "service-account.json", "id_rsa", "id_ed25519",
}

# Common local-only folders are excluded by Git when using the fallback walker.
WALK_IGNORES = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    "node_modules", ".mypy_cache", ".ruff_cache",
}


def classify_path(relative_path: str | Path) -> list[str]:
    """Return conservative risk labels based only on a relative path."""
    path = Path(relative_path)
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    suffix = path.suffix.lower()
    risks: list[str] = []

    if name in SECRET_NAMES:
        risks.append("possible credential/secret file")

    if any(part in RESTRICTED_DIR_NAMES for part in parts[:-1]):
        risks.append("file is inside a match-data/ground-truth directory")

    # `data/` is common for many harmless projects, so only flag it when the
    # path also looks like a competition asset or a data serialization.
    looks_like_tracking_name = any(
        token in name for token in ("_atd", "homography", "tactical", "broadcast")
    )
    if "data" in parts[:-1] and (
        suffix in RESTRICTED_SUFFIXES
        or suffix in {".csv", ".xml", ".json"}
        or looks_like_tracking_name
    ):
        risks.append("data file under a data/ directory")

    if suffix in RESTRICTED_SUFFIXES:
        risks.append("video/tracking/serialized-data file type")

    if suffix in {".csv", ".xml", ".json"} and any(
        part in {"match_data", "raw_data", "private_data", "datasets", "ground_truth", "groundtruth"}
        for part in parts[:-1]
    ):
        risks.append("tabular/structured file in restricted-data directory")

    if any(token in name for token in RESTRICTED_NAME_PARTS):
        risks.append("filename suggests private or match-derived output")

    # Keep labels unique while retaining order.
    return list(dict.fromkeys(risks))


def _git_file_list(root: Path, args: list[str]) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", *args, "-z"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [item for item in result.stdout.split("\0") if item]


def discover_repo_files(root: Path) -> tuple[list[str], str]:
    """Find tracked + non-ignored untracked paths, without reading file bytes."""
    tracked = _git_file_list(root, [])
    untracked = _git_file_list(root, ["--others", "--exclude-standard"])
    if tracked is not None and untracked is not None:
        paths = sorted(set(tracked) | set(untracked))
        return paths, "Git tracked + non-ignored untracked files"

    paths: list[str] = []
    for current, dirs, files in __import__("os").walk(root):
        dirs[:] = [d for d in dirs if d not in WALK_IGNORES]
        current_path = Path(current)
        for filename in files:
            absolute = current_path / filename
            paths.append(absolute.relative_to(root).as_posix())
    return sorted(paths), "filesystem fallback (Git metadata unavailable)"


def run_check(root: Path) -> int:
    root = root.resolve()
    if not root.is_dir():
        print(f"ERROR: repository path does not exist: {root}")
        return 2

    paths, source = discover_repo_files(root)
    findings: list[tuple[str, list[str]]] = []
    for relative in paths:
        risks = classify_path(relative)
        if risks:
            findings.append((relative, risks))

    print("STRATIQ — SUBMISSION HYGIENE PREFLIGHT")
    print("=" * 48)
    print(f"Repository : {root}")
    print(f"Scan mode  : {source}")
    print(f"Paths seen : {len(paths)}")
    print("File contents read: NO")
    print()

    if findings:
        print(f"REVIEW REQUIRED: {len(findings)} path(s) need inspection")
        for relative, risks in findings:
            print(f"  - {relative}")
            for risk in risks:
                print(f"      reason: {risk}")
        print()
        print("Do not publish flagged competition assets or derived extracts.")
        print("Move local-only files outside the public repository and add suitable")
        print("ignore rules. Re-run this check before creating a public release.")
        return 1

    print("PASS: no obvious restricted paths detected.")
    print("LIMIT: this checks names/paths only. It cannot detect renamed data,")
    print("embedded extracts, hard-coded match-derived values, or rule violations")
    print("in the workflow. Review the Terms and the human-readable checklist too.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project repository root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)
    return run_check(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
