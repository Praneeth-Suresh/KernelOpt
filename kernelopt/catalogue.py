"""Append-only benchmark evidence catalogue."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def append(source: Path, destination: Path, hardware: str, driver: str) -> int:
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            record["catalogued_at"] = datetime.now(timezone.utc).isoformat()
            record["hardware"] = hardware
            record["driver"] = driver
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    command = sub.add_parser("append")
    command.add_argument("source", type=Path)
    command.add_argument("--destination", type=Path, default=Path("results/catalogue.jsonl"))
    command.add_argument("--hardware", required=True)
    command.add_argument("--driver", required=True)
    args = parser.parse_args()
    print(f"appended {append(args.source, args.destination, args.hardware, args.driver)} records")


if __name__ == "__main__":
    main()
