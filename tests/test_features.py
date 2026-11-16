import unittest

import numpy as np

from tactiq.data_loader import TrackingData, PITCH_LENGTH, PITCH_WIDTH
from tactiq.features import extract_features


class TestFeatureExtraction(unittest.TestCase):

    def setUp(self):
        # Dua frame, dua track:
        # Track 0 = Barcelona
        # Track 1 = Athletic Club
        self.data = TrackingData(
            home_name="Barcelona",
            away_name="Athletic Club",
            team_codes=np.array([0, 1], dtype=np.int8),
            players=np.array([
                [
                    [0.5, 0.5, 1.0],
                    [0.6, 0.5, 2.0],
                ],
                [
                    [0.5, 0.5, 1.0],
                    [0.6, 0.5, 2.0],
                ],
            ], dtype=np.float32),
            ball=np.array([
                [0.5, 0.5, 0.0],
                [np.nan, np.nan, np.nan],
            ], dtype=np.float32),
        )

    def test_ball_visibility(self):
        features = extract_features(self.data)
        np.testing.assert_array_equal(
            features.ball_visible,
            [True, False],
        )

    def test_nearest_distances(self):
        features = extract_features(self.data)

        self.assertAlmostEqual(
            features.home_nearest_distance[0],
            0.0,
            places=4,
        )
        self.assertAlmostEqual(
            features.away_nearest_distance[0],
            10.53,
            places=2,
        )

    def test_nearest_track(self):
        features = extract_features(self.data)

        self.assertEqual(features.home_nearest_track[0], 0)
        self.assertEqual(features.away_nearest_track[0], 1)

    def test_nearby_player_count(self):
        features = extract_features(self.data, radius=5.0)

        self.assertEqual(features.home_players_nearby[0], 1)
        self.assertEqual(features.away_players_nearby[0], 0)

    def test_missing_ball_produces_nan(self):
        features = extract_features(self.data)

        self.assertTrue(
            np.isnan(features.home_nearest_distance[1])
        )
        self.assertTrue(
            np.isnan(features.away_nearest_distance[1])
        )
        self.assertTrue(
            np.isnan(features.home_players_nearby[1])
        )
        self.assertEqual(features.home_nearest_track[1], -1)
        self.assertTrue(np.isnan(features.home_nearest_speed[1]))
        self.assertTrue(np.isnan(features.away_nearest_speed[1]))

    def test_nearest_speed_comes_from_nearest_track(self):
        features = extract_features(self.data)

        self.assertAlmostEqual(features.home_nearest_speed[0], 1.0)
        self.assertAlmostEqual(features.away_nearest_speed[0], 2.0)

    def test_missing_reported_speed_remains_nan(self):
        self.data.players[0, 0, 2] = np.nan

        features = extract_features(self.data)

        self.assertEqual(features.home_nearest_track[0], 0)
        self.assertTrue(np.isnan(features.home_nearest_speed[0]))

    def test_ball_inside_boundary(self):
        features = extract_features(self.data)

        self.assertTrue(features.ball_inside_boundary[0])
        self.assertFalse(features.ball_outside_boundary[0])
        self.assertAlmostEqual(
            features.ball_outside_distance[0],
            0.0,
            places=6,
        )

    def test_ball_outside_x_boundary(self):
        self.data.ball[0, 0] = 1.1

        features = extract_features(self.data)

        self.assertFalse(features.ball_inside_boundary[0])
        self.assertTrue(features.ball_outside_boundary[0])
        self.assertAlmostEqual(
            features.ball_outside_distance[0],
            0.1 * PITCH_LENGTH,
            places=4,
        )

    def test_ball_outside_y_boundary(self):
        self.data.ball[0, 1] = -0.1

        features = extract_features(self.data)

        self.assertFalse(features.ball_inside_boundary[0])
        self.assertTrue(features.ball_outside_boundary[0])
        self.assertAlmostEqual(
            features.ball_outside_distance[0],
            0.1 * PITCH_WIDTH,
            places=4,
        )

    def test_missing_ball_has_no_boundary_classification(self):
        features = extract_features(self.data)

        self.assertFalse(features.ball_inside_boundary[1])
        self.assertFalse(features.ball_outside_boundary[1])
        self.assertTrue(np.isnan(features.ball_outside_distance[1]))

    def test_radius_must_be_finite_and_positive(self):
        for radius in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                extract_features(self.data, radius=radius)

    def test_empty_team_is_rejected(self):
        data = TrackingData(
            home_name="Home",
            away_name="Away",
            team_codes=np.array([0, 0], dtype=np.int8),
            players=self.data.players,
            ball=self.data.ball,
        )
        with self.assertRaisesRegex(ValueError, "Both teams"):
            extract_features(data)


if __name__ == "__main__":
    unittest.main()
