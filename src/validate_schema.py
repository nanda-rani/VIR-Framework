#!/usr/bin/env python3
"""Validate a CSV against the public VIR field contract."""

import argparse
import csv
from pathlib import Path

from vir_fields import LABEL_FIELDS, VALUE_RANGES


REQUIRED = ["record_id", "is_synthetic", "split", "narrative"] + LABEL_FIELDS
SPLITS = {"train", "validation", "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    errors = []
    seen_ids = set()
    with args.csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in REQUIRED if field not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing required columns: {', '.join(missing)}")
        count = 0
        for line_number, row in enumerate(reader, start=2):
            count += 1
            record_id = row["record_id"]
            if not record_id:
                errors.append(f"line {line_number}: empty record_id")
            elif record_id in seen_ids:
                errors.append(f"line {line_number}: duplicate record_id {record_id}")
            seen_ids.add(record_id)
            if row["split"] not in SPLITS:
                errors.append(f"line {line_number}: invalid split {row['split']!r}")
            if not row["narrative"].strip():
                errors.append(f"line {line_number}: empty narrative")
            if row["is_synthetic"] not in {"0", "1"}:
                errors.append(f"line {line_number}: is_synthetic must be 0 or 1")
            for field in LABEL_FIELDS:
                try:
                    value = int(row[field])
                except ValueError:
                    errors.append(f"line {line_number}: {field} is not an integer")
                    continue
                if value not in VALUE_RANGES[field]:
                    errors.append(f"line {line_number}: {field}={value} is outside the schema")

    if errors:
        shown = "\n".join(errors[:30])
        suffix = f"\n... and {len(errors) - 30} more" if len(errors) > 30 else ""
        raise SystemExit(f"Schema validation failed:\n{shown}{suffix}")
    print(f"Schema validation passed for {count} records in {args.csv_path}")


if __name__ == "__main__":
    main()
