"""CSV loading, deterministic splitting, tokenization, and VIR batching."""

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vir_fields import ACTIONS, ASSISTANCE, EMOTIONS


TASK_FIELDS = {
    "attack_vector": ["attack_vector"],
    "awareness": ["awareness"],
    "attribution": ["attribution"],
    "misconception": ["misconception"],
    "emotions": EMOTIONS,
    "assistance": ASSISTANCE,
    "actions": ACTIONS,
}


def load_rows(path: Path, text_column="narrative") -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No records found in {path}")
    required = {text_column} | {field for fields in TASK_FIELDS.values() for field in fields}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError("Missing model fields: " + ", ".join(missing))
    return rows


def deterministic_split(rows, seed, train_ratio, validation_ratio, stratification_column):
    named = {name: [row for row in rows if row.get("split") == name] for name in ("train", "validation", "test")}
    if all(named.values()):
        return named

    groups = defaultdict(list)
    fallback = "attack_vector"
    for row in rows:
        stratum = row.get(stratification_column) or row[fallback]
        groups[stratum].append(row)

    rng = random.Random(seed)
    output = {"train": [], "validation": [], "test": []}
    for stratum in sorted(groups):
        group = list(groups[stratum])
        rng.shuffle(group)
        n = len(group)
        train_end = max(1, round(n * train_ratio))
        validation_count = max(1, round(n * validation_ratio)) if n >= 3 else 0
        validation_end = min(n, train_end + validation_count)
        output["train"].extend(group[:train_end])
        output["validation"].extend(group[train_end:validation_end])
        output["test"].extend(group[validation_end:])

    for split in output.values():
        rng.shuffle(split)
    if not all(output.values()):
        raise ValueError("The deterministic split produced an empty partition; provide explicit train/validation/test values")
    return output


class VIRDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class VIRCollator:
    def __init__(self, tokenizer, maximum_sequence_length, text_column="narrative"):
        self.tokenizer = tokenizer
        self.maximum_sequence_length = maximum_sequence_length
        self.text_column = text_column

    def __call__(self, rows):
        encoded = self.tokenizer(
            [row[self.text_column] for row in rows],
            padding=True,
            truncation=True,
            max_length=self.maximum_sequence_length,
            return_tensors="pt",
        )
        labels = {
            "attack_vector": torch.tensor([int(row["attack_vector"]) for row in rows], dtype=torch.long),
            "awareness": torch.tensor([int(row["awareness"]) for row in rows], dtype=torch.long),
            "attribution": torch.tensor([[int(row["attribution"])] for row in rows], dtype=torch.float),
            "misconception": torch.tensor([[int(row["misconception"])] for row in rows], dtype=torch.float),
            "emotions": torch.tensor([[int(row[field]) for field in EMOTIONS] for row in rows], dtype=torch.float),
            "assistance": torch.tensor([[int(row[field]) for field in ASSISTANCE] for row in rows], dtype=torch.float),
            "actions": torch.tensor([[int(row[field]) for field in ACTIONS] for row in rows], dtype=torch.float),
        }
        return {"inputs": encoded, "labels": labels}
