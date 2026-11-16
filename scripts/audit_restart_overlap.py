"""Audit overlaps and temporal gaps between SET PIECES events."""

import argparse
from pathlib import Path

from tactiq.smart_tagging_features import _read_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        default=(
            "../more-than-a-hack-2027/data/tactical/"
            "BAR-ATH_tactical_smart_tagging.xml"
        ),
    )
    args = parser.parse_args()

    events = _read_events(Path(args.xml))

    selected = []
    for code, start, end, tags in events:
        if code == "SET PIECES":
            selected.append({
                "start": start,
                "end": end,
                "type": tags.get("Type", "(unknown)"),
                "team": tags.get("Team", "(unknown)"),
            })

    selected.sort(key=lambda event: event["start"])

    overlaps = []
    gaps = []

    for previous, current in zip(selected, selected[1:]):
        gap = current["start"] - previous["end"]

        if gap < 0:
            overlaps.append((previous, current, abs(gap)))
        else:
            gaps.append((previous, current, gap))

    print("\nTACTIQ AI - RESTART OVERLAP AUDIT")
    print("=" * 75)
    print(f"Total events             : {len(selected)}")
    print(f"Overlapping adjacent pairs: {len(overlaps)}")
    print(f"Non-overlapping pairs    : {len(gaps)}")

    if overlaps:
        print("\nOVERLAPPING EVENTS")
        print("-" * 75)

        for previous, current, duration in overlaps:
            print(
                f"{previous['type']} ({previous['team']}) "
                f"[{previous['start']:.2f}, {previous['end']:.2f}]"
            )
            print(
                f"  overlaps {current['type']} ({current['team']}) "
                f"[{current['start']:.2f}, {current['end']:.2f}]"
            )
            print(f"  overlap duration: {duration:.2f}s\n")

    if gaps:
        gap_values = [item[2] for item in gaps]

        print("\nGAP STATISTICS")
        print("-" * 75)
        print(f"Minimum gap : {min(gap_values):.2f}s")
        print(f"Median gap  : {sorted(gap_values)[len(gap_values)//2]:.2f}s")
        print(f"Maximum gap : {max(gap_values):.2f}s")
        print(f"Gaps <= 5s  : {sum(g <= 5 for g in gap_values)}")
        print(f"Gaps <= 10s : {sum(g <= 10 for g in gap_values)}")

    print("\nNo possession or dead-ball labels were generated.")


if __name__ == "__main__":
    main()
