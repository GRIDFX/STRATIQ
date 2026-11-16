import tempfile
import unittest
from pathlib import Path

from tactiq.smart_tagging_features import (
    extract_smart_tagging_features,
)


class TestSmartTaggingFeatures(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.xml_path = Path(self.temp_dir.name) / "sample.xml"

        self.xml_path.write_text(
            """<?xml version="1.0"?>
            <file>
              <instance>
                <ID>1</ID>
                <start>0.5</start>
                <end>2.5</end>
                <code>SET PIECES</code>
                <label>
                  <group>Team</group>
                  <text>N/A</text>
                </label>
                <label>
                  <group>Type</group>
                  <text>Corner Kick</text>
                </label>
                <label>
                  <group>Half</group>
                  <text>1st Half</text>
                </label>
              </instance>

              <instance>
                <ID>2</ID>
                <start>1.0</start>
                <end>2.0</end>
                <code>PLAYERS IN THE BOX</code>
                <label>
                  <group>Team</group>
                  <text>Barcelona</text>
                </label>
                <label>
                  <group>Max Players in the box</group>
                  <text>3</text>
                </label>
              </instance>

              <instance>
                <ID>3</ID>
                <start>1.5</start>
                <end>2.5</end>
                <code>PLAYERS IN THE BOX</code>
                <label>
                  <group>Team</group>
                  <text>Atletic Club</text>
                </label>
                <label>
                  <group>Max Players in the box</group>
                  <text>5</text>
                </label>
              </instance>

              <instance>
                <ID>4</ID>
                <start>4.0</start>
                <end>6.0</end>
                <code>BUILD UP</code>
                <label>
                  <group>Team</group>
                  <text>Barcelona</text>
                </label>
              </instance>
            </file>
            """,
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def extract(self, n_frames=10, fps=2):
        return extract_smart_tagging_features(
            self.xml_path,
            n_frames=n_frames,
            fps=fps,
        )

    def test_seconds_are_converted_to_frames(self):
        features = self.extract()

        # 0.5 * 2 = frame 1; 2.5 * 2 = frame 5.
        # The interval is [1, 5).
        active = features.restart_active[:, 1]
        self.assertEqual(
            active.tolist(),
            [False, True, True, True, True,
             False, False, False, False, False],
        )

    def test_end_frame_is_exclusive(self):
        features = self.extract()

        self.assertTrue(features.category_active[4, 13])
        self.assertFalse(features.category_active[5, 13])

    def test_overlapping_box_counts_use_maximum(self):
        features = self.extract()

        # 3 players: frames [2, 4)
        # 5 players: frames [3, 5)
        expected = [float("nan"), float("nan"), 3, 5, 5]
        actual = features.max_players_in_box[:5].tolist()

        self.assertEqual(actual[2:], expected[2:])

    def test_na_team_is_not_mapped_to_either_team(self):
        features = self.extract()

        # Frame 1 hanya memiliki SET PIECES dengan Team = N/A.
        self.assertFalse(features.team_active[1].any())

        # Frame 2 memiliki anotasi Barcelona.
        self.assertTrue(features.team_active[2, 0])

        # Frame 2 tidak memiliki anotasi Atletic Club.
        self.assertFalse(features.team_active[2, 1])

    def test_interval_is_clipped_to_frame_limit(self):
        features = self.extract(n_frames=10, fps=2)

        # BUILD UP: 4.0-6.0 seconds -> [8, 12), clipped to [8, 10).
        self.assertTrue(features.category_active[8, 3])
        self.assertTrue(features.category_active[9, 3])
        self.assertEqual(features.n_frames, 10)


if __name__ == "__main__":
    unittest.main()
