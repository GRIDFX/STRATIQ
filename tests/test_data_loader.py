from pathlib import Path

import numpy as np
import pytest

from tactiq.data_loader import load_team_metadata, load_tracking, parse_object


def write_fixture(tmp_path: Path, *, players_xml: str, atd: str):
    tactical = tmp_path / "tactical"
    tactical.mkdir()
    (tactical / "BAR-ATH_tactical_atd.xml").write_text(players_xml, encoding="utf-8")
    (tactical / "BAR-ATH_tactical_atd.txt").write_text(atd, encoding="utf-8")
    return tmp_path


XML = """<root>
<Team id="home"><Name>Home</Name></Team>
<Team id="away"><Name>Away</Name></Team>
<Player id="1" teamId="home" />
<Player id="2" teamId="away" />
</root>"""


def test_parse_object_rejects_infinity():
    with pytest.raises(ValueError, match="infinity"):
        parse_object("inf,0,0")


def test_empty_atd_is_rejected(tmp_path):
    write_fixture(tmp_path, players_xml=XML, atd="")
    with pytest.raises(ValueError, match="Expected 2 frames, found 0"):
        load_tracking(tmp_path, n_frames=2)


def test_n_frames_must_be_positive_integer(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        load_tracking(tmp_path, n_frames=0)
    with pytest.raises(ValueError, match="positive integer"):
        load_tracking(tmp_path, n_frames=True)


def test_loader_preserves_synthetic_frame_alignment(tmp_path):
    atd = "1:0.1,0.2,1;0.3,0.4,2:0.5,0.6,0\n2:0.2,0.3,1;0.4,0.5,2:0.6,0.7,0\n"
    data = load_tracking(write_fixture(tmp_path, players_xml=XML, atd=atd), n_frames=2)
    assert data.players.shape == (2, 2, 3)
    assert data.ball.shape == (2, 3)
    np.testing.assert_array_equal(data.team_codes, [0, 1])
