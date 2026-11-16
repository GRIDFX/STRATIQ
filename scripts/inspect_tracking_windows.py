import numpy as np

from tactiq.data_loader import load_tracking

data = load_tracking("../more-than-a-hack-2027/data")

WINDOWS = [
    ("PLAYER", 15951, 5, 23),
    ("PLAYER", 115353, 5, 23),
    ("BALL", 106716, 5, None),
    ("BALL", 105359, 5, None),
    ("BALL", 108965, 5, None),
    ("BALL", 179905, 5, None),
]

for kind, center, radius, track in WINDOWS:
    start = max(1, center - radius)
    end = min(data.n_frames, center + radius)

    print(f"\n{'=' * 85}")
    print(f"{kind} | CENTER FRAME {center} | RANGE {start}-{end}")
    print("=" * 85)

    for frame in range(start, end + 1):
        i = frame - 1

        if kind == "PLAYER":
            p = data.players[i, track]
            x, y, reported_speed = p
            print(
                f"frame={frame:6d} "
                f"x={x: .6f} y={y: .6f} "
                f"reported_speed={reported_speed:8.3f}"
            )
        else:
            x, y, reported_speed = data.ball[i]
            print(
                f"frame={frame:6d} "
                f"x={x: .6f} y={y: .6f} "
                f"reported_speed={reported_speed:8.3f}"
            )

print("\nInspection completed. No tracking data modified.")
