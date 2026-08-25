#!/usr/bin/env python3
"""Fail on common privacy and double-blind release hazards."""

import argparse
import csv
import re
from pathlib import Path


FORBIDDEN_FILENAMES = {
    ".DS_Store",
    "HUMAN_ANNOTATED_GOLD.csv",
    "LLM_ANNOTATED_GOLD.csv",
    "SILVER_LABELED_WITH_PROBS.csv",
}
FORBIDDEN_CSV_COLUMNS = {
    "crimeaditionalinfo",
    "__row_id",
    "llm_raw_json",
    "llm_input_tokens",
    "llm_output_tokens",
}
TEXT_SUFFIXES = {".md", ".txt", ".tex", ".json", ".yaml", ".yml", ".py", ".cff"}
GENERATED_OR_PRIVATE_DIRECTORIES = {".git", "__pycache__", "runs", "predictions", "checkpoints", "embeddings", "logs", "private"}
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
ABSOLUTE_USER_PATH = re.compile(r"/(?:Users|home)/[^/\s]+/")
IDENTIFYING_REPOSITORY = re.compile(r"https?://(?:www\.)?(?:github|gitlab)\.com/[^/\s]+/", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", nargs="?", type=Path, default=Path("."))
    parser.add_argument(
        "--private-markers",
        type=Path,
        help="Optional file outside the artifact containing one identifying string per line",
    )
    args = parser.parse_args()
    root = args.artifact_root.resolve()
    markers = []
    if args.private_markers:
        markers = [line.strip().casefold() for line in args.private_markers.read_text(encoding="utf-8").splitlines() if line.strip()]

    problems = []
    warnings = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in GENERATED_OR_PRIVATE_DIRECTORIES for part in relative.parts):
            if path.is_file():
                problems.append(f"private/generated metadata present: {relative}")
            continue
        if not path.is_file():
            continue
        if relative == Path("src/validate_release.py"):
            # The validator necessarily contains the patterns it searches for.
            continue
        if path.name in FORBIDDEN_FILENAMES:
            problems.append(f"forbidden private filename: {relative}")
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="", errors="replace") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames or []
                public_rows = list(reader)
            found = sorted(FORBIDDEN_CSV_COLUMNS & set(header))
            if found:
                problems.append(f"forbidden private columns in {relative}: {', '.join(found)}")
            if "narrative" in header:
                if "record_id" not in header or "is_synthetic" not in header:
                    problems.append(f"narrative-bearing CSV lacks synthetic record IDs: {relative}")
                for line_number, row in enumerate(public_rows, start=2):
                    if not row.get("record_id", "").startswith("SYN-"):
                        problems.append(f"non-synthetic record ID in public narrative CSV {relative}:{line_number}")
                    if row.get("is_synthetic") != "1":
                        problems.append(f"non-synthetic narrative row in {relative}:{line_number}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Makefile", "LICENSE-CODE", "LICENSE-DOCUMENTATION"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            # Documentation is allowed to name omitted fields while no CSV may contain them.
            if EMAIL.search(text):
                problems.append(f"email address in {relative}")
            if ABSOLUTE_USER_PATH.search(text):
                problems.append(f"absolute user path in {relative}")
            if IDENTIFYING_REPOSITORY.search(text):
                problems.append(f"potentially identifying repository URL in {relative}")
            folded = text.casefold()
            for marker in markers:
                if marker in folded:
                    problems.append(f"private identity marker {marker!r} in {relative}")

    if problems:
        raise SystemExit("Release validation FAILED:\n- " + "\n- ".join(sorted(set(problems))))
    print("Release validation passed: no prohibited files, columns, or direct identity patterns found.")
    for warning in sorted(set(warnings)):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
