import numpy as np
from types import SimpleNamespace

from tactiq.tracking_dynamics import extract_tracking_dynamics


def make_tracking_data():
    players = np.array(
        [
            [[0.1, 0.2, 1.5], [0.3, 0.4, 2.5]],
            [[0.2, 0.2, 1.7], [0.4, 0.4, np.nan]],
            [[0.3, 0.2, np.nan], [0.5, 0.4, 3.0]],
        ],
        dtype=np.float32,
    )

    ball = np.array(
        [
            [0.5, 0.5, 4.0],
            [0.6, 0.5, 4.5],
            [0.7, 0.5, np.nan],
        ],
        dtype=np.float32,
    )

    return SimpleNamespace(players=players, ball=ball)


def test_player_speed_uses_reported_speed_column():
    data = make_tracking_data()

    result = extract_tracking_dynamics(data)

    np.testing.assert_allclose(
        result.player_speed[0],
        np.array([1.5, 2.5]),
    )
    assert result.player_speed[1, 0] == 1.7


def test_ball_speed_uses_reported_speed_column():
    data = make_tracking_data()

    result = extract_tracking_dynamics(data)

    np.testing.assert_allclose(
        result.ball_speed[:2],
        np.array([4.0, 4.5]),
    )


def test_missing_speeds_remain_nan():
    data = make_tracking_data()

    result = extract_tracking_dynamics(data)

    assert np.isnan(result.player_speed[1, 1])
    assert np.isnan(result.player_speed[2, 0])
    assert np.isnan(result.ball_speed[2])


def test_speed_extraction_does_not_depend_on_position_differences():
    data = make_tracking_data()

    # Change coordinates while keeping reported speeds unchanged.
    data.players[1, 0, :2] = [0.99, 0.99]
    data.ball[1, :2] = [0.01, 0.01]

    result = extract_tracking_dynamics(data)

    assert result.player_speed[1, 0] == 1.7
    assert result.ball_speed[1] == 4.5
