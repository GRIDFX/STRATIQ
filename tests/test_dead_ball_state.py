from types import SimpleNamespace

import numpy as np
import pytest
from types import SimpleNamespace

from tactiq.dead_ball_state import apply_dead_ball_state


def objects(labels, inside, candidate, restart=None, speed=None):
    n = len(labels)
    possession = SimpleNamespace(
        labels=np.asarray(list(labels) if isinstance(labels, str) else labels, dtype="<U1"),
        confidence=np.full(n, 0.5, dtype=np.float32),
    )
    features = SimpleNamespace(ball_inside_boundary=np.asarray(inside, dtype=bool))
    evidence = SimpleNamespace(candidate=np.asarray(candidate, dtype=bool))
    smart = None
    if restart is not None:
        smart = SimpleNamespace(restart_active=np.asarray(restart, dtype=bool))
    dynamics = None
    if speed is not None:
        dynamics = SimpleNamespace(ball_speed=np.asarray(speed, dtype=np.float32))
    return possession, features, evidence, smart, dynamics


def run_case(labels, inside, candidate, restart=None, speed=None, **kwargs):
    args = objects(labels, inside, candidate, restart, speed)
    return apply_dead_ball_state(*args[:3], smart_tagging=args[3], dynamics=args[4], **kwargs)


def test_no_dead_ball_evidence_preserves_base_labels():
    result = run_case("HHAXX", [1] * 5, [0] * 5)
    assert result.labels.tolist() == list("HHAXX")


def test_boundary_candidate_starts_dead_ball():
    result = run_case("XXHHH", [0, 0, 1, 1, 1], [1, 0, 0, 0, 0])
    assert result.labels.tolist()[:3] == ["D", "D", "D"]
    assert result.labels.tolist()[3:] == ["D", "H"]


def test_dead_ball_waits_for_confirmed_live_possession():
    result = run_case("XAAAAA", [0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0],
                      release_confirm_frames=3)
    assert result.labels.tolist() == ["D", "D", "D", "A", "A", "A"]


def test_restart_context_holds_dead_ball_until_context_ends():
    restart = np.array([[1], [1], [1], [0], [0], [0]], dtype=bool)
    result = run_case("HHHHHH", [1] * 6, [0] * 6, restart=restart,
                      release_confirm_frames=2)
    assert result.labels.tolist() == ["D", "D", "D", "D", "H", "H"]


def test_provider_speed_can_confirm_restart_release_during_tag():
    restart = np.array([[1], [1], [1], [1], [1]], dtype=bool)
    result = run_case("HHHHH", [1] * 5, [0] * 5, restart=restart,
                      speed=[0.0, 0.0, 1.2, 1.3, 1.4],
                      release_confirm_frames=2, restart_release_speed_mps=1.0)
    assert result.labels.tolist() == ["D", "D", "D", "H", "H"]


def test_shape_mismatch_is_rejected():
    possession, features, evidence, smart, dynamics = objects(
        "HHH", [1, 1, 1], [0, 0, 0]
    )
    features.ball_inside_boundary = np.array([1, 1], dtype=bool)
    with pytest.raises(ValueError, match="expected 3"):
        apply_dead_ball_state(possession, features, evidence)


def test_restart_matrix_must_match_frame_count():
    possession, features, evidence, _, _ = objects("HH", [1, 1], [0, 0])
    smart = SimpleNamespace(restart_active=np.zeros((3, 1), dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        apply_dead_ball_state(possession, features, evidence, smart_tagging=smart)


def test_invalid_label_is_rejected():
    possession, features, evidence, _, _ = objects("HZ", [1, 1], [0, 0])
    with pytest.raises(ValueError, match="unsupported label"):
        apply_dead_ball_state(possession, features, evidence)


def test_release_confirm_frames_must_be_positive():
    possession, features, evidence, _, _ = objects("H", [1], [0])
    with pytest.raises(ValueError, match="positive integer"):
        apply_dead_ball_state(possession, features, evidence, release_confirm_frames=0)
    with pytest.raises(ValueError, match="positive integer"):
        apply_dead_ball_state(possession, features, evidence, release_confirm_frames=1.5)


def test_non_boolean_state_inputs_are_rejected():
    possession, features, evidence, _, _ = objects("H", [1], [0])
    features.ball_inside_boundary = np.array([1], dtype=np.int8)
    with pytest.raises(ValueError, match="booleans"):
        apply_dead_ball_state(possession, features, evidence)


def test_negative_ball_speed_is_rejected():
    possession, features, evidence, _, dynamics = objects(
        "H", [1], [0], speed=[-1.0]
    )
    with pytest.raises(ValueError, match="non-negative"):
        apply_dead_ball_state(possession, features, evidence, dynamics=dynamics)

def test_long_unsupported_label_is_rejected():
    possession = SimpleNamespace(
        labels=np.array(["HOME"], dtype="<U4"),
        confidence=np.array([0.5], dtype=np.float32),
    )
    features = SimpleNamespace(ball_inside_boundary=np.array([True]))
    evidence = SimpleNamespace(candidate=np.array([False]))

    with pytest.raises(ValueError, match="unsupported label"):
        apply_dead_ball_state(possession, features, evidence)