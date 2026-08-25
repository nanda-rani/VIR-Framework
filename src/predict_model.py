#!/usr/bin/env python3
"""Apply a trained VIR model without exporting narrative text."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ml_data import VIRCollator, VIRDataset, load_rows
from ml_metrics import sigmoid, softmax
from modeling import MultiTaskHingRoBERTa
from train_model import choose_device, move_batch
from vir_fields import ACTIONS, ASSISTANCE, EMOTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads((args.run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    thresholds = json.loads((args.run_dir / "thresholds.json").read_text(encoding="utf-8"))
    rows = load_rows(args.data, config["data"]["text_column"])
    tokenizer_path = args.run_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else config["base_model"])
    loader = DataLoader(
        VIRDataset(rows),
        batch_size=config["optimization"]["batch_size"],
        collate_fn=VIRCollator(tokenizer, config["data"]["maximum_sequence_length"], config["data"]["text_column"]),
    )
    device = choose_device(args.device)
    model = MultiTaskHingRoBERTa(config["base_model"], **config["architecture"]).to(device)
    model.load_state_dict(torch.load(args.run_dir / "model_state.pt", map_location=device, weights_only=True))
    model.eval()
    collected = {task: [] for task in ("attack_vector", "awareness", "attribution", "misconception", "emotions", "assistance", "actions")}
    with torch.no_grad():
        for batch in loader:
            inputs, _ = move_batch(batch, device)
            logits = model(**inputs)
            for task, values in logits.items():
                collected[task].append(values.detach().cpu().numpy())
    logits = {task: np.concatenate(values) for task, values in collected.items()}
    attack_probabilities = softmax(logits["attack_vector"])
    awareness_probabilities = softmax(logits["awareness"])
    binary_probabilities = {task: sigmoid(logits[task]) for task in ("attribution", "misconception", "emotions", "assistance", "actions")}

    output_rows = []
    for index, row in enumerate(rows):
        output = {
            "record_id": row.get("record_id", str(index + 1)),
            "attack_vector": int(np.argmax(attack_probabilities[index])),
            "attack_vector_confidence": float(np.max(attack_probabilities[index])),
            "awareness": int(np.argmax(awareness_probabilities[index])),
            "awareness_confidence": float(np.max(awareness_probabilities[index])),
        }
        for task in ("attribution", "misconception"):
            probability = float(binary_probabilities[task][index, 0])
            output[task] = int(probability >= thresholds[task][0])
            output[f"{task}_probability"] = probability
        for task, fields in (("emotions", EMOTIONS), ("assistance", ASSISTANCE), ("actions", ACTIONS)):
            for column, field in enumerate(fields):
                probability = float(binary_probabilities[task][index, column])
                output[field] = int(probability >= thresholds[task][column])
                output[f"{field}_probability"] = probability
        output_rows.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} prediction records without narrative text to {args.output}")


if __name__ == "__main__":
    main()
