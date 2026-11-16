"""Convert Smart Tagging XML intervals into frame-level context features.

These features are automatic tactical annotations, not possession ground truth.
Intervals use the official half-open convention [start_frame, end_frame).
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


CATEGORIES = (
    "ATTACKING TRANSITION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "BUILD UP",
    "CREATING CHANCES",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENSIVE TRANSITION",
    "GOALS",
    "LONG BALL",
    "PLAYERS IN THE BOX",
    "PROGRESSION",
    "SET PIECES",
)

TEAMS = ("Barcelona", "Atletic Club")

RESTART_TYPES = (
    "Throw-in",
    "Corner Kick",
    "Goal Kick",
    "Free Kick",
    "Kick Off (After Goal)",
    "Kick Off (Start)",
)

SIDES = ("Central", "Left", "Right")
DIRECTIONS = ("Horizontal", "Vertical", "Diagonal")
HALVES = ("1st Half", "2nd Half")


@dataclass
class SmartTaggingFeatures:
    category_names: tuple
    team_names: tuple
    restart_names: tuple
    side_names: tuple
    direction_names: tuple
    half_names: tuple

    category_active: np.ndarray
    team_active: np.ndarray
    restart_active: np.ndarray
    max_players_in_box: np.ndarray
    side_active: np.ndarray
    direction_active: np.ndarray
    half_active: np.ndarray

    @property
    def n_frames(self):
        return len(self.max_players_in_box)


def _read_events(xml_path):
    root = ET.parse(xml_path).getroot()
    events = []

    for instance in root.iter("instance"):
        code = (instance.findtext("code") or "").strip()
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))

        tags = {}
        for label in instance.findall("label"):
            group = (label.findtext("group") or "").strip()
            value = (label.findtext("text") or "").strip()
            if group and value:
                tags[group] = value

        events.append((code, start_s, end_s, tags))

    return events


def _activate(matrix, column_names, value, start, end):
    if value in column_names and end > start:
        matrix[start:end, column_names.index(value)] = True


def extract_smart_tagging_features(
    xml_path,
    n_frames,
    fps=25,
):
    """Create frame-level features from Smart Tagging XML."""

    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")

    events = _read_events(Path(xml_path))

    n_categories = len(CATEGORIES)
    n_teams = len(TEAMS)
    n_restarts = len(RESTART_TYPES)
    n_sides = len(SIDES)
    n_directions = len(DIRECTIONS)
    n_halves = len(HALVES)

    category_active = np.zeros((n_frames, n_categories), dtype=bool)
    team_active = np.zeros((n_frames, n_teams), dtype=bool)
    restart_active = np.zeros((n_frames, n_restarts), dtype=bool)
    side_active = np.zeros((n_frames, n_sides), dtype=bool)
    direction_active = np.zeros((n_frames, n_directions), dtype=bool)
    half_active = np.zeros((n_frames, n_halves), dtype=bool)

    max_players_in_box = np.full(n_frames, np.nan, dtype=float)

    for code, start_s, end_s, tags in events:
        if start_s < 0 or end_s <= start_s:
            raise ValueError(
                f"Invalid interval for {code}: {start_s} - {end_s}"
            )

        start = int(round(start_s * fps))
        end = int(round(end_s * fps))

        # Clip intervals to the available tactical frame range.
        start = max(0, min(start, n_frames))
        end = max(0, min(end, n_frames))

        if end <= start:
            continue

        _activate(category_active, CATEGORIES, code, start, end)

        team = tags.get("Team")
        _activate(team_active, TEAMS, team, start, end)

        if code == "SET PIECES":
            _activate(
                restart_active,
                RESTART_TYPES,
                tags.get("Type"),
                start,
                end,
            )

        _activate(side_active, SIDES, tags.get("Side"), start, end)
        _activate(
            direction_active,
            DIRECTIONS,
            tags.get("Direction of ball entry"),
            start,
            end,
        )
        _activate(half_active, HALVES, tags.get("Half"), start, end)

        raw_count = tags.get("Max Players in the box")
        if raw_count:
            count = 7 if raw_count == "7+" else int(raw_count)
            current = max_players_in_box[start:end]
            max_players_in_box[start:end] = np.where(
                np.isnan(current),
                count,
                np.maximum(current, count),
            )

    return SmartTaggingFeatures(
        category_names=CATEGORIES,
        team_names=TEAMS,
        restart_names=RESTART_TYPES,
        side_names=SIDES,
        direction_names=DIRECTIONS,
        half_names=HALVES,
        category_active=category_active,
        team_active=team_active,
        restart_active=restart_active,
        max_players_in_box=max_players_in_box,
        side_active=side_active,
        direction_active=direction_active,
        half_active=half_active,
    )


def print_summary(features):
    print("\nTACTIQ AI - SMART TAGGING FEATURES")
    print("=" * 55)
    print(f"Frames                    : {features.n_frames:,}")
    print(
        "Frames with category     : "
        f"{features.category_active.any(axis=1).sum():,}"
    )
    print(
        "Frames with team context : "
        f"{features.team_active.any(axis=1).sum():,}"
    )
    print(
        "Frames with restart type : "
        f"{features.restart_active.any(axis=1).sum():,}"
    )
    print(
        "Frames with box context  : "
        f"{np.isfinite(features.max_players_in_box).sum():,}"
    )
    print(
        "Frames with side context : "
        f"{features.side_active.any(axis=1).sum():,}"
    )
    print(
        "Frames with direction    : "
        f"{features.direction_active.any(axis=1).sum():,}"
    )
    print(
        "Frames with half context : "
        f"{features.half_active.any(axis=1).sum():,}"
    )

    print("\nCATEGORY FRAME COUNTS")
    for index, name in enumerate(features.category_names):
        count = features.category_active[:, index].sum()
        print(f"{name:<35} {count:>8,}")

    print("\nNo possession labels were generated.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default=(
            "../more-than-a-hack-2027/data/tactical/"
            "BAR-ATH_tactical_smart_tagging.xml"
        ),
    )
    parser.add_argument("--frames", type=int, default=199319)
    parser.add_argument("--fps", type=float, default=25)
    args = parser.parse_args()

    features = extract_smart_tagging_features(
        args.xml,
        n_frames=args.frames,
        fps=args.fps,
    )
    print_summary(features)


if __name__ == "__main__":
    main()
