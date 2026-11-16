import numpy as np

from tactiq.data_loader import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    load_tracking,
)

FPS = 25.0
TOP_N = 20

data = load_tracking("../more-than-a-hack-2027/data")


def inspect(name, positions, track_names=None):
    positions = np.asarray(positions, dtype=float).copy()

    # Convert normalized coordinates to meters.
    positions[..., 0] *= PITCH_LENGTH
    positions[..., 1] *= PITCH_WIDTH

    previous = positions[:-1]
    current = positions[1:]

    valid = (
        np.isfinite(previous).all(axis=-1)
        & np.isfinite(current).all(axis=-1)
    )

    displacement = np.linalg.norm(current - previous, axis=-1)
    speed = displacement * FPS
    speed[~valid] = np.nan

    print(f"\n{name}")
    print("=" * 90)
    print("Valid consecutive pairs:", np.isfinite(speed).sum())

    flat = speed.reshape(-1)
    valid_indices = np.flatnonzero(np.isfinite(flat))

    if not len(valid_indices):
        return

    order = valid_indices[
        np.argsort(flat[valid_indices])[::-1][:TOP_N]
    ]

    for rank, flat_index in enumerate(order, start=1):
        index = np.unravel_index(flat_index, speed.shape)
        frame = index[0] + 1

        if speed.ndim == 1:
            track = "BALL"
            prev_pos = previous[index]
            curr_pos = current[index]
        else:
            track_index = index[1]
            track = track_names[track_index]
            prev_pos = previous[index]
            curr_pos = current[index]

        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]

        print(
            f"{rank:02d}. frame={frame:6d} "
            f"track={track:12s} "
            f"speed={flat[flat_index]:9.3f} m/s "
            f"dx={dx:8.3f} m dy={dy:8.3f} m"
        )


track_names = [
    f"team{data.team_codes[i]}_track{i}"
    for i in range(len(data.team_codes))
]

print("TACTIQ AI - CORRECTED VELOCITY OUTLIER AUDIT")
print("=" * 90)

inspect(
    "PLAYER OUTLIERS",
    data.players[:, :, :2],
    track_names,
)

inspect(
    "BALL OUTLIERS",
    data.ball[:, :2],
)

print("\nAudit completed. No possession labels modified.")
