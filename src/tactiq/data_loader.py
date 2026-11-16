from dataclasses import dataclass
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

import numpy as np


N_FRAMES = 199_319
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0


@dataclass
class TrackingData:
    home_name: str
    away_name: str
    team_codes: np.ndarray
    players: np.ndarray
    ball: np.ndarray

    @property
    def n_frames(self):
        return self.players.shape[0]

    @property
    def n_tracks(self):
        return self.players.shape[1]


def load_team_metadata(xml_path):
    """Read team names and map each ATD track slot to H=0 or A=1."""
    root = ET.parse(xml_path).getroot()

    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"Expected 2 teams, found {len(teams)}")

    home_id = teams[0].get("id")
    home_name = teams[0].findtext("Name", default="Home")
    away_name = teams[1].findtext("Name", default="Away")

    players = list(root.iter("Player"))
    if not players:
        raise ValueError("No Player entries found in metadata")

    slots = sorted(players, key=lambda item: int(item.get("id")))

    expected_ids = list(range(1, len(slots) + 1))
    actual_ids = [int(item.get("id")) for item in slots]

    if actual_ids != expected_ids:
        raise ValueError("Player track IDs are not sequential from 1")

    team_codes = []
    for player in slots:
        team_id = player.get("teamId")

        if team_id == home_id:
            team_codes.append(0)
        elif team_id == teams[1].get("id"):
            team_codes.append(1)
        else:
            raise ValueError(
                f"Unknown teamId for track {player.get('id')}: {team_id}"
            )

    team_codes = np.asarray(team_codes, dtype=np.int8)
    if not np.any(team_codes == 0) or not np.any(team_codes == 1):
        raise ValueError("Metadata must contain at least one track per team")

    return home_name, away_name, team_codes


def parse_object(raw_object):
    """Parse x,y,speed. Missing observations become NaN."""
    values = raw_object.split(",")

    if len(values) != 3:
        raise ValueError(f"Invalid tracking object: {raw_object!r}")

    parsed = [float(value) for value in values]
    if any(np.isinf(value) for value in parsed):
        raise ValueError(f"Tracking object contains infinity: {raw_object!r}")
    return parsed


def load_tracking(data_dir, n_frames=N_FRAMES):
    """Load tactical ATD into NumPy arrays.

    players: (frames, tracks, 3), containing x, y, speed
    ball:    (frames, 3), containing x, y, speed

    Coordinates are normalized to 0..1. Missing values are NaN.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, (int, np.integer)):
        raise ValueError("n_frames must be a positive integer")
    if n_frames < 1:
        raise ValueError("n_frames must be a positive integer")

    data_dir = Path(data_dir)
    tactical_dir = data_dir / "tactical"

    xml_path = tactical_dir / "BAR-ATH_tactical_atd.xml"
    raw_path = tactical_dir / "BAR-ATH_tactical_atd.txt"

    if not xml_path.is_file():
        raise FileNotFoundError(f"Metadata not found: {xml_path}")

    if not raw_path.is_file():
        raise FileNotFoundError(f"ATD file not found: {raw_path}")

    home_name, away_name, team_codes = load_team_metadata(xml_path)

    n_tracks = len(team_codes)

    players = np.full(
        (n_frames, n_tracks, 3), np.nan, dtype=np.float32
    )
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)

    line_number = 0
    with raw_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split(":")

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid ATD format at line {line_number}"
                )

            frame_id = int(parts[0])

            if frame_id != line_number:
                raise ValueError(
                    f"Frame mismatch: line {line_number}, "
                    f"frame_id {frame_id}"
                )

            if frame_id > n_frames:
                raise ValueError(
                    f"ATD contains more than {n_frames} frames"
                )

            player_groups = parts[1].split(";")

            if len(player_groups) != n_tracks:
                raise ValueError(
                    f"Frame {frame_id}: expected {n_tracks} tracks, "
                    f"found {len(player_groups)}"
                )

            frame_index = frame_id - 1

            for slot, group in enumerate(player_groups):
                players[frame_index, slot] = parse_object(group)

            ball[frame_index] = parse_object(parts[2])

    if line_number != n_frames:
        raise ValueError(
            f"Expected {n_frames} frames, found {line_number}"
        )

    return TrackingData(
        home_name=home_name,
        away_name=away_name,
        team_codes=team_codes,
        players=players,
        ball=ball,
    )


def print_summary(data):
    """Print aggregate diagnostics without exporting match data."""
    player_visible = np.isfinite(data.players[:, :, 0]) & np.isfinite(
        data.players[:, :, 1]
    )
    ball_visible = np.isfinite(data.ball[:, 0]) & np.isfinite(
        data.ball[:, 1]
    )

    home_tracks = int(np.sum(data.team_codes == 0))
    away_tracks = int(np.sum(data.team_codes == 1))

    print("\nSTRATIQ - DATA LOADER REPORT")
    print("=" * 40)
    print(f"Home team       : {data.home_name}")
    print(f"Away team       : {data.away_name}")
    print(f"Frames          : {data.n_frames:,}")
    print(f"Track slots     : {data.n_tracks}")
    print(f"Home track slots: {home_tracks}")
    print(f"Away track slots: {away_tracks}")
    print(f"Player array    : {data.players.shape}")
    print(f"Ball array      : {data.ball.shape}")
    print()
    print(
        f"Ball visible    : {ball_visible.sum():,} "
        f"({ball_visible.mean() * 100:.2f}%)"
    )
    print(
        f"Ball missing    : {(~ball_visible).sum():,} "
        f"({(~ball_visible).mean() * 100:.2f}%)"
    )
    print(
        f"Player positions: {player_visible.sum():,} "
        f"of {data.n_frames * data.n_tracks:,} observations"
    )
    print("\nData loading completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect tactical ATD data"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to the competition data directory",
    )
    args = parser.parse_args()

    data = load_tracking(args.data_dir)
    print_summary(data)


if __name__ == "__main__":
    main()