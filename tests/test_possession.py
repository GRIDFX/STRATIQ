import unittest

import numpy as np

from tactiq.data_loader import TrackingData
from tactiq.features import extract_features
from tactiq.possession import estimate_possession


class TestPossession(unittest.TestCase):

    def make_data(self, ball, home_x, away_x):
        n = len(ball)

        players = np.full((n, 2, 3), np.nan, dtype=np.float32)
        players[:, 0, 0] = home_x
        players[:, 0, 1] = 0.5
        players[:, 0, 2] = 1.0

        players[:, 1, 0] = away_x
        players[:, 1, 1] = 0.5
        players[:, 1, 2] = 1.0

        return TrackingData(
            home_name="Barcelona",
            away_name="Athletic Club",
            team_codes=np.array([0, 1], dtype=np.int8),
            players=players,
            ball=np.asarray(ball, dtype=np.float32),
        )

    def test_initial_home_control(self):
        ball = [[0.5, 0.5, 0.0]] * 5
        data = self.make_data(
            ball,
            home_x=[0.5] * 5,
            away_x=[0.7] * 5,
        )

        result = estimate_possession(data, extract_features(data))

        np.testing.assert_array_equal(result.labels, ["H"] * 5)

    def test_switch_requires_confirmation(self):
        ball = [[0.5, 0.5, 0.0]] * 6
        data = self.make_data(
            ball,
            home_x=[0.5, 0.5, 0.5, 0.7, 0.7, 0.7],
            away_x=[0.7, 0.7, 0.7, 0.5, 0.5, 0.5],
        )

        result = estimate_possession(
            data,
            extract_features(data),
            switch_frames=3,
        )

        self.assertEqual(result.labels[0], "H")
        self.assertEqual(result.labels[2], "H")
        self.assertEqual(result.labels[-1], "A")

    def test_contested_ball_keeps_previous_team(self):
        ball = [[0.5, 0.5, 0.0]] * 4
        data = self.make_data(
            ball,
            home_x=[0.5, 0.5, 0.5, 0.5],
            away_x=[0.7, 0.7, 0.51, 0.51],
        )

        result = estimate_possession(data, extract_features(data))

        np.testing.assert_array_equal(result.labels, ["H"] * 4)

    def test_missing_ball_holds_then_abstains(self):
        ball = [
            [0.5, 0.5, 0.0],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ]
        data = self.make_data(
            ball,
            home_x=[0.5] * 4,
            away_x=[0.7] * 4,
        )

        result = estimate_possession(
            data,
            extract_features(data),
            missing_ball_hold_frames=2,
        )

        np.testing.assert_array_equal(
            result.labels,
            ["H", "H", "H", "X"],
        )

    def test_confidence_is_bounded(self):
        ball = [[0.5, 0.5, 0.0]] * 4
        data = self.make_data(
            ball,
            home_x=[0.5] * 4,
            away_x=[0.7] * 4,
        )

        result = estimate_possession(data, extract_features(data))

        self.assertTrue(np.all(result.confidence >= 0))
        self.assertTrue(np.all(result.confidence <= 1))

    def test_outside_boundary_abstains_and_resets_state(self):
        ball = [
            [0.5, 0.5, 0.0],
            [1.1, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
        data = self.make_data(
            ball,
            home_x=[0.5, 0.7, 0.7],
            away_x=[0.7, 0.5, 0.5],
        )

        result = estimate_possession(
            data,
            extract_features(data),
            switch_frames=3,
        )

        np.testing.assert_array_equal(result.labels, ["H", "X", "A"])

    def test_no_player_evidence_holds_then_abstains(self):
        ball = [[0.5, 0.5, 0.0]] * 4
        data = self.make_data(
            ball,
            home_x=[0.5, np.nan, np.nan, np.nan],
            away_x=[0.7, np.nan, np.nan, np.nan],
        )

        result = estimate_possession(
            data,
            extract_features(data),
            no_evidence_hold_frames=2,
        )

        np.testing.assert_array_equal(
            result.labels,
            ["H", "H", "H", "X"],
        )

    def test_zero_distance_advantage_does_not_assign_tie_to_home(self):
        ball = [[0.5, 0.5, 0.0]]
        data = self.make_data(
            ball,
            home_x=[0.5],
            away_x=[0.5],
        )

        result = estimate_possession(
            data,
            extract_features(data),
            min_distance_advantage=0,
        )

        self.assertEqual(result.labels[0], "X")


    def test_no_candidate_holds_then_abstains(self):
        ball = [[0.5, 0.5, 0.0]] * 4
        data = self.make_data(
            ball,
            home_x=[0.5, 0.7, 0.7, 0.7],
            away_x=[0.7, 0.8, 0.8, 0.8],
        )

        result = estimate_possession(
            data,
            extract_features(data),
            no_candidate_hold_frames=2,
        )

        np.testing.assert_array_equal(
            result.labels,
            ["H", "H", "H", "X"],
        )

    def test_valid_candidate_resets_no_candidate_timer(self):
        ball = [[0.5, 0.5, 0.0]] * 7
        data = self.make_data(
            ball,
            home_x=[0.5, 0.7, 0.7, 0.5, 0.7, 0.7, 0.7],
            away_x=[0.7, 0.8, 0.8, 0.7, 0.8, 0.8, 0.8],
        )

        result = estimate_possession(
            data,
            extract_features(data),
            no_candidate_hold_frames=2,
        )

        np.testing.assert_array_equal(
            result.labels,
            ["H", "H", "H", "H", "H", "H", "X"],
        )

    def test_new_candidate_initializes_after_abstention(self):
        ball = [[0.5, 0.5, 0.0]] * 5
        data = self.make_data(
            ball,
            home_x=[0.5, 0.7, 0.7, 0.7, 0.7],
            away_x=[0.7, 0.8, 0.8, 0.5, 0.5],
        )

        result = estimate_possession(
            data,
            extract_features(data),
            no_candidate_hold_frames=1,
        )

        np.testing.assert_array_equal(
            result.labels,
            ["H", "H", "X", "A", "A"],
        )

    def test_negative_no_candidate_hold_frames_is_rejected(self):
        ball = [[0.5, 0.5, 0.0]]
        data = self.make_data(
            ball,
            home_x=[0.5],
            away_x=[0.7],
        )

        with self.assertRaises(ValueError):
            estimate_possession(
                data,
                extract_features(data),
                no_candidate_hold_frames=-1,
            )


if __name__ == "__main__":
    unittest.main()
