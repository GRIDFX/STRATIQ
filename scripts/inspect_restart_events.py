"""Inspect individual Smart Tagging SET PIECES event intervals."""

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
    selected = [
        (code, start, end, tags)
        for code, start, end, tags in events
        if code == "SET PIECES"
    ]

    print("\nTACTIQ AI - INDIVIDUAL RESTART EVENT INSPECTION")
    print("=" * 95)
    print(f"Total SET PIECES events: {len(selected)}")
    print(
        f"{'ID':>3} {'START':>10} {'END':>10} "
        f"{'DURATION':>10} {'TYPE':<25} {'TEAM':<20}"
    )
    print("-" * 95)

    for index, (code, start, end, tags) in enumerate(selected, 1):
        duration = end - start
        restart_type = tags.get("Type", "(unknown)")
        team = tags.get("Team", "(unknown)")

        print(
            f"{index:>3} "
            f"{start:>10.2f} "
            f"{end:>10.2f} "
            f"{duration:>9.2f}s "
            f"{restart_type:<25} "
            f"{team:<20}"
        )

    print("\nNo labels were generated or modified.")


if __name__ == "__main__":
    main()
