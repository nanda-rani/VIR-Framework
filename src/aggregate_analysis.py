#!/usr/bin/env python3
"""Produce aggregate VIR distributions with small-cell suppression."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from vir_fields import ACTIONS, ASSISTANCE, ATTACK_VECTOR_NAMES, AWARENESS_NAMES, EMOTIONS


def distribution(rows, field, names=None):
    counts = Counter(int(row[field]) for row in rows)
    total = len(rows)
    return {
        (names[value] if names else str(value)): {
            "n": count,
            "proportion": count / total,
        }
        for value, count in sorted(counts.items())
    }


def indicator_rates(rows, fields):
    return {
        field: {"n": sum(int(row[field]) for row in rows), "proportion": sum(int(row[field]) for row in rows) / len(rows)}
        for field in fields
    }


def suppressed_rate(rows, field, minimum):
    positive = sum(int(row[field]) for row in rows)
    negative = len(rows) - positive
    if min(positive, negative) < minimum:
        return {"n": len(rows), "positive_n": "SUPPRESSED", "proportion": "SUPPRESSED"}
    return {"n": len(rows), "positive_n": positive, "proportion": positive / len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--min-cell-size", type=int, default=5)
    args = parser.parse_args()
    if args.min_cell_size < 1:
        raise SystemExit("--min-cell-size must be positive")

    with args.input_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Input contains no records")

    output = {
        "records": len(rows),
        "small_cell_threshold": args.min_cell_size,
        "attack_vector": distribution(rows, "attack_vector", ATTACK_VECTOR_NAMES),
        "attribution": distribution(rows, "attribution", {0: "Internal attribution", 1: "External attribution"}),
        "technical_awareness": distribution(rows, "awareness", AWARENESS_NAMES),
        "misconception": distribution(rows, "misconception", {0: "No identified misconception", 1: "Misconception present"}),
        "emotional_expression": indicator_rates(rows, EMOTIONS),
        "requested_assistance": indicator_rates(rows, ASSISTANCE),
        "post_incident_action": indicator_rates(rows, ACTIONS),
        "by_awareness": {},
    }

    for value, name in AWARENESS_NAMES.items():
        subset = [row for row in rows if int(row["awareness"]) == value]
        if len(subset) < args.min_cell_size:
            output["by_awareness"][name] = {"n": len(subset), "status": "SUPPRESSED"}
            continue
        output["by_awareness"][name] = {
            "n": len(subset),
            "misconception": suppressed_rate(subset, "misconception", args.min_cell_size),
            "emotional_expression": {field: suppressed_rate(subset, field, args.min_cell_size) for field in EMOTIONS},
            "requested_assistance": {field: suppressed_rate(subset, field, args.min_cell_size) for field in ASSISTANCE},
            "post_incident_action": {field: suppressed_rate(subset, field, args.min_cell_size) for field in ACTIONS},
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aggregate-only analysis for {len(rows)} records to {args.output_json}")


if __name__ == "__main__":
    main()
