"""Read and validate Smart Tagging metadata."""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from collections import Counter


@dataclass
class TagEvent:
    event_id: int
    start: float
    end: float
    code: str
    half: str | None


def load_smart_tagging(xml_path):
    root = ET.parse(xml_path).getroot()
    events = []

    for instance in root.iter("instance"):
        values = {}

        for child in instance:
            if len(child) == 0:
                values[child.tag] = (child.text or "").strip()
            else:
                for sub in child:
                    if sub.tag == "text":
                        values[f"{child.tag}_{sub.tag}"] = (
                            sub.text or ""
                        ).strip()

        if not all(k in values for k in ("ID", "start", "end", "code")):
            continue

        events.append(
            TagEvent(
                event_id=int(values["ID"]),
                start=float(values["start"]),
                end=float(values["end"]),
                code=values["code"],
                half=values.get("label_text"),
            )
        )

    return events


def validate_events(events):
    invalid = [
        event for event in events
        if event.start < 0 or event.end <= event.start
    ]

    ids = [event.event_id for event in events]
    duplicate_ids = len(ids) - len(set(ids))

    return invalid, duplicate_ids


def print_report(events):
    counts = Counter(event.code for event in events)
    invalid, duplicate_ids = validate_events(events)

    print("\nTACTIQ AI - SMART TAGGING REPORT")
    print("=" * 55)
    print(f"Total events       : {len(events)}")
    print(f"Invalid intervals  : {len(invalid)}")
    print(f"Duplicate IDs      : {duplicate_ids}")

    print("\nEVENT COUNTS")
    for code, count in counts.most_common():
        print(f"{code:<35} {count:>5}")

    print("\nEVENT TIME RANGE")
    if events:
        print(f"Earliest start     : {min(e.start for e in events):.2f}s")
        print(f"Latest end         : {max(e.end for e in events):.2f}s")

    print("\nHALF DISTRIBUTION")
    halves = Counter(event.half or "(unknown)" for event in events)
    for half, count in halves.items():
        print(f"{half:<25} {count:>5}")

    print("\nCATEGORY DETAILS")
    for code in sorted(counts):
        selected = [e for e in events if e.code == code]
        durations = [e.end - e.start for e in selected]

        print(
            f"{code:<35} "
            f"count={len(selected):>4} "
            f"min={min(durations):.2f}s "
            f"max={max(durations):.2f}s"
        )

    print("\nNo possession labels were generated.")


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

    path = Path(args.xml)
    if not path.exists():
        raise FileNotFoundError(path)

    events = load_smart_tagging(path)
    print_report(events)


if __name__ == "__main__":
    main()
