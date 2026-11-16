import unittest
from types import SimpleNamespace

import numpy as np

from tactiq.dead_ball import extract_dead_ball_evidence


def make_inputs(n_frames=6):
    features = SimpleNamespace(
        ball_visible=np.ones(n_frames, dtype=bool),
        ball_outside_boundary=np.zeros(n_frames, dtype=bool),
        ball_outside_distance=np.zeros(n_frames, dtype=np.float32),
    )

    smart_tagging = SimpleNamespace(
        restart_active=np.zeros((n_frames, 6), dtype=bool),
    )

    dynamics = SimpleNamespace(
        ball_speed=np.ones(n_frames, dtype=np.float32),
    )

    return features, smart_tagging, dynamics


class TestDeadBallEvidence(unittest.TestCase):

    def test_persistent_outside_boundary_is_candidate(self):
        features, smart, dynamics = make_inputs()

        features.ball_outside_boundary[2:4] = True
        features.ball_outside_distance[2:4] = 0.5

        result = extract_dead_ball_evidence(features, smart, dynamics)

        self.assertTrue(result.boundary_persistent[2])
        self.assertTrue(result.boundary_persistent[3])
        self.assertTrue(result.candidate[2])
        self.assertTrue(result.candidate[3])
        self.assertEqual(result.level[2], "strong")

    def test_single_outside_frame_is_not_candidate_by_default(self):
        features, smart, dynamics = make_inputs()

        features.ball_outside_boundary[2] = True
        features.ball_outside_distance[2] = 0.5

        result = extract_dead_ball_evidence(features, smart, dynamics)

        self.assertTrue(result.boundary_outside[2])
        self.assertFalse(result.boundary_persistent[2])
        self.assertFalse(result.candidate[2])
        self.assertAlmostEqual(float(result.score[2]), 0.50)

    def test_outside_distance_within_margin_is_not_evidence(self):
        features, smart, dynamics = make_inputs()

        features.ball_outside_boundary[2:4] = True
        features.ball_outside_distance[2:4] = 0.10

        result = extract_dead_ball_evidence(features, smart, dynamics)

        self.assertFalse(result.boundary_outside.any())
        self.assertFalse(result.candidate.any())

    def test_restart_context_alone_is_not_candidate(self):
        features, smart, dynamics = make_inputs()

        smart.restart_active[2:4, 1] = True

        result = extract_dead_ball_evidence(features, smart, dynamics)

        self.assertTrue(result.restart_context[2])
        self.assertFalse(result.candidate.any())

    def test_low_speed_alone_is_not_candidate(self):
        features, smart, dynamics = make_inputs()

        dynamics.ball_speed[:] = 0.1

        result = extract_dead_ball_evidence(features, smart, dynamics)

        self.assertTrue(result.ball_stationary.all())
        self.assertFalse(result.candidate.any())
        self.assertTrue(np.all(result.score == 0))

    def test_restart_and_low_speed_do_not_add_speed_score(self):
        features, smart, dynamics = make_inputs()

        smart.restart_active[2, 0] = True
        dynamics.ball_speed[2] = 0.1

        result = extract_dead_ball_evidence(features, smart, dynamics)

        self.assertAlmostEqual(float(result.score[2]), 0.10)
        self.assertFalse(result.candidate[2])

    def test_speed_does_not_change_evidence_score(self):
        features, smart, dynamics = make_inputs()

        features.ball_outside_boundary[2] = True
        features.ball_outside_distance[2] = 0.5
        smart.restart_active[2, 0] = True

        with_speed = extract_dead_ball_evidence(
            features, smart, dynamics
        )
        without_speed = extract_dead_ball_evidence(
            features, smart, dynamics=None
        )

        np.testing.assert_array_equal(
            with_speed.score,
            without_speed.score,
        )
        np.testing.assert_array_equal(
            with_speed.candidate,
            without_speed.candidate,
        )

    def test_missing_ball_is_not_interpreted_as_dead_ball(self):
        features, smart, dynamics = make_inputs()

        features.ball_visible[2] = False
        smart.restart_active[2, 0] = True
        dynamics.ball_speed[2] = 0.1

        result = extract_dead_ball_evidence(features, smart, dynamics)

        self.assertFalse(result.boundary_outside[2])
        self.assertFalse(result.ball_stationary[2])
        self.assertFalse(result.candidate[2])

    def test_optional_context_inputs_can_be_omitted(self):
        features, _, _ = make_inputs()

        result = extract_dead_ball_evidence(features)

        self.assertFalse(result.restart_context.any())
        self.assertFalse(result.ball_speed_available.any())
        self.assertFalse(result.candidate.any())

    def test_smart_tagging_frame_mismatch_is_rejected(self):
        features, smart, dynamics = make_inputs()
        smart.restart_active = np.zeros((5, 6), dtype=bool)

        with self.assertRaises(ValueError):
            extract_dead_ball_evidence(features, smart, dynamics)

    def test_invalid_candidate_threshold_is_rejected(self):
        features, smart, dynamics = make_inputs()

        with self.assertRaises(ValueError):
            extract_dead_ball_evidence(
                features,
                smart,
                dynamics,
                candidate_threshold=0.2,
            )

    def test_nan_boolean_inputs_are_rejected(self):
        features, smart, dynamics = make_inputs()
        features.ball_visible = np.array([True, False, np.nan, True, True, True])
        with self.assertRaisesRegex(ValueError, "boolean"):
            extract_dead_ball_evidence(features, smart, dynamics)

    def test_nonfinite_threshold_is_rejected(self):
        features, smart, dynamics = make_inputs()
        with self.assertRaises(ValueError):
            extract_dead_ball_evidence(
                features, smart, dynamics, candidate_threshold=np.nan
            )


if __name__ == "__main__":
    unittest.main()
