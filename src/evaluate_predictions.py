#!/usr/bin/env python3
"""Compute aggregate VIR prediction metrics without external dependencies."""

import argparse
import csv
import json
import math
from pathlib import Path

from vir_fields import (
    ACTIONS,
    ASSISTANCE,
    ATTACK_VECTOR,
    ATTRIBUTION,
    AWARENESS,
    EMOTIONS,
    MISCONCEPTION,
)


def accuracy(gold, pred):
    return sum(a == b for a, b in zip(gold, pred)) / len(gold)


def f1_for_class(gold, pred, positive):
    tp = sum(a == positive and b == positive for a, b in zip(gold, pred))
    fp = sum(a != positive and b == positive for a, b in zip(gold, pred))
    fn = sum(a == positive and b != positive for a, b in zip(gold, pred))
    return 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def macro_f1(gold, pred, classes):
    return sum(f1_for_class(gold, pred, value) for value in classes) / len(classes)


def cohen_kappa(gold, pred):
    observed = accuracy(gold, pred)
    values = sorted(set(gold) | set(pred))
    expected = sum(
        (gold.count(value) / len(gold)) * (pred.count(value) / len(pred))
        for value in values
    )
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            out[index] = average_rank
        position = end
    return out


def pearson(left, right):
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((a - left_mean) ** 2 for a in left)
    right_ss = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_ss * right_ss)
    return 0.0 if denominator == 0 else numerator / denominator


def jaccard_for_group(row, fields):
    gold = {field for field in fields if int(row[field]) == 1}
    pred = {field for field in fields if int(row[f"pred_{field}"]) == 1}
    return 1.0 if not gold and not pred else len(gold & pred) / len(gold | pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    with args.input_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Input contains no records")

    required = [ATTACK_VECTOR, ATTRIBUTION, AWARENESS, MISCONCEPTION] + EMOTIONS + ASSISTANCE + ACTIONS
    missing = [field for field in required if field not in rows[0] or f"pred_{field}" not in rows[0]]
    if missing:
        raise SystemExit("Missing gold/prediction fields: " + ", ".join(missing))

    values = {
        field: ([int(row[field]) for row in rows], [int(row[f"pred_{field}"]) for row in rows])
        for field in required
    }
    attack_gold, attack_pred = values[ATTACK_VECTOR]
    aware_gold, aware_pred = values[AWARENESS]

    output = {
        "records": len(rows),
        "attack_vector": {
            "accuracy": accuracy(attack_gold, attack_pred),
            "macro_f1": macro_f1(attack_gold, attack_pred, range(6)),
            "confusion_matrix": [
                [sum(a == gold_class and b == pred_class for a, b in zip(attack_gold, attack_pred))
                 for pred_class in range(6)]
                for gold_class in range(6)
            ],
        },
        "technical_awareness": {
            "accuracy": accuracy(aware_gold, aware_pred),
            "within_one_level_accuracy": sum(abs(a - b) <= 1 for a, b in zip(aware_gold, aware_pred)) / len(rows),
            "macro_f1": macro_f1(aware_gold, aware_pred, range(3)),
            "spearman_rho": pearson(ranks(aware_gold), ranks(aware_pred)),
        },
        "attribution": {
            "accuracy": accuracy(*values[ATTRIBUTION]),
            "macro_f1": macro_f1(*values[ATTRIBUTION], classes=range(2)),
            "cohen_kappa": cohen_kappa(*values[ATTRIBUTION]),
        },
        "misconception": {
            "accuracy": accuracy(*values[MISCONCEPTION]),
            "f1": f1_for_class(*values[MISCONCEPTION], positive=1),
            "cohen_kappa": cohen_kappa(*values[MISCONCEPTION]),
        },
    }

    for name, fields in (("emotional_expression", EMOTIONS), ("requested_assistance", ASSISTANCE), ("post_incident_action", ACTIONS)):
        similarities = [jaccard_for_group(row, fields) for row in rows]
        output[name] = {
            "mean_jaccard": sum(similarities) / len(similarities),
            "per_label_f1": {
                field: f1_for_class(*values[field], positive=1) for field in fields
            },
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aggregate evaluation for {len(rows)} records to {args.output_json}")


if __name__ == "__main__":
    main()
