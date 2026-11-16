"""Compare coordinate-derived ball speed with provider-reported speed."""

import argparse
import numpy as np

from tactiq.data_loader import load_tracking, PITCH_LENGTH, PITCH_WIDTH
from tactiq.features import extract_features
from tactiq.tracking_dynamics import extract_tracking_dynamics


def describe(name, values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    print(f"\n{name}")
    print("-" * 70)

    if values.size == 0:
        print("No valid observations.")
        return

    print(f"Count   : {values.size:,}")
    print(f"Median  : {np.median(values):.3f}")
    print(f"P95     : {np.percentile(values, 95):.3f}")
    print(f"P99     : {np.percentile(values, 99):.3f}")
    print(f"Maximum : {np.max(values):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Audit reported and coordinate-derived ball speed"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    if args.fps <= 0 or args.top < 1:
        parser.error("fps and top must be positive")

    print("Loading tracking data...")
    data = load_tracking(args.data_dir)

    ball = np.asarray(data.ball)
    features = extract_features(data)
    dynamics = extract_tracking_dynamics(data)

    xy = ball[:, :2]
    visible = features.ball_visible

    valid_pairs = visible[:-1] & visible[1:]
    pair_indices = np.flatnonzero(valid_pairs)

    dx = np.diff(xy[:, 0]) * PITCH_LENGTH
    dy = np.diff(xy[:, 1]) * PITCH_WIDTH

    displacement = np.hypot(dx, dy)
    derived_speed = displacement * args.fps
    derived_speed = derived_speed[valid_pairs]

    reported = dynamics.ball_speed
    reported_from = reported[pair_indices]
    reported_to = reported[pair_indices + 1]

    both_reported = (
        np.isfinite(reported_from)
        & np.isfinite(reported_to)
        & (reported_from >= 0)
        & (reported_to >= 0)
    )

    mean_reported = np.full(len(pair_indices), np.nan)
    mean_reported[both_reported] = (
        reported_from[both_reported] + reported_to[both_reported]
    ) / 2.0

    one_reported = (
        np.isfinite(reported_from) | np.isfinite(reported_to)
    )

    boundary_outside = features.ball_outside_boundary

    from_outside = boundary_outside[pair_indices]
    to_outside = boundary_outside[pair_indices + 1]

    both_inside = ~from_outside & ~to_outside
    both_outside = from_outside & to_outside
    crossing_boundary = from_outside ^ to_outside

    print("\nBALL SPEED AUDIT")
    print("=" * 70)
    print(f"Total frames                  : {data.n_frames:,}")
    print(f"Visible consecutive pairs     : {len(pair_indices):,}")
    print(f"Pairs with both speeds        : {both_reported.sum():,}")
    print(f"Pairs with at least one speed : {one_reported.sum():,}")

    describe("COORDINATE-DERIVED SPEED (m/s)", derived_speed)
    describe(
        "PROVIDER SPEED AT PAIR START (m/s)",
        reported_from,
    )
    describe(
        "PROVIDER SPEED AT PAIR END (m/s)",
        reported_to,
    )
    describe(
        "MEAN PROVIDER SPEED AT BOTH ENDS (m/s)",
        mean_reported,
    )

    print("\nDERIVED SPEED THRESHOLDS")
    print("-" * 70)

    for threshold in (10, 20, 30, 50, 75, 100):
        count = int(np.sum(derived_speed > threshold))
        print(
            f"Above {threshold:>3} m/s : "
            f"{count:>8,} ({count / max(len(derived_speed), 1) * 100:.3f}%)"
        )

    print("\nPAIR LOCATION")
    print("-" * 70)

    for name, mask in (
        ("Both frames inside", both_inside),
        ("Both frames outside", both_outside),
        ("Crossing boundary", crossing_boundary),
    ):
        values = derived_speed[mask]
        high = int(np.sum(values > 30))
        print(
            f"{name:<22}: {len(values):>8,} pairs | "
            f"median={np.median(values) if len(values) else float('nan'):.2f} "
            f"m/s | above 30 m/s={high:,}"
        )

    comparable = (
        both_reported
        & np.isfinite(derived_speed)
        & np.isfinite(mean_reported)
    )

    print("\nDERIVED VS PROVIDER COMPARISON")
    print("-" * 70)

    if comparable.sum() >= 2:
        derived_values = derived_speed[comparable]
        reported_values = mean_reported[comparable]

        describe("Derived speed on comparable pairs", derived_values)
        describe("Mean provider speed on comparable pairs", reported_values)

        correlation = np.corrcoef(
            derived_values,
            reported_values,
        )[0, 1]

        print(f"\nPearson correlation : {correlation:.4f}")
        print(
            "Median absolute difference : "
            f"{np.median(np.abs(derived_values - reported_values)):.3f} m/s"
        )
    else:
        print("Not enough pairs with provider speed at both endpoints.")

    # Print the largest coordinate-derived jumps with both provider speeds.
    selected = np.argsort(derived_speed)[::-1][:args.top]

    print(f"\nTOP {min(args.top, len(selected))} DERIVED SPEED PAIRS")
    print("-" * 110)
    print(
        "FROM  TO     DERIVED    REPORTED_FROM  REPORTED_TO  "
        "FROM_STATE  TO_STATE"
    )

    for local_index in selected:
        frame = int(pair_indices[local_index])
        next_frame = frame + 1

        from_state = (
            "outside" if from_outside[local_index] else "inside"
        )
        to_state = (
            "outside" if to_outside[local_index] else "inside"
        )

        print(
            f"{frame:>6} {next_frame:>6} "
            f"{derived_speed[local_index]:>10.2f} "
            f"{reported_from[local_index]:>14.2f} "
            f"{reported_to[local_index]:>12.2f} "
            f"{from_state:>12} {to_state:>10}"
        )

    print("\nNo tracking data or possession labels were modified.")


if __name__ == "__main__":
    main()
