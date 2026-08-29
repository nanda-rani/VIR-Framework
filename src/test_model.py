#!/usr/bin/env python3
"""Evaluate a saved VIR checkpoint on an explicitly marked test split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ml_data import VIRCollator, VIRDataset, load_rows
from ml_metrics import evaluate_logits
from modeling import MultiTaskHingRoBERTa, VIRLoss, compute_training_weights
from train_model import choose_device, collect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/model_test_metrics.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads((args.run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    thresholds = json.loads((args.run_dir / "thresholds.json").read_text(encoding="utf-8"))
    rows = [row for row in load_rows(args.data, config["data"]["text_column"]) if row.get("split") == "test"]
    if not rows:
        raise SystemExit("No rows with split=test")
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
    class_weights, positive_weights = compute_training_weights(rows)
    loss_function = VIRLoss(
        class_weights,
        positive_weights,
        config["loss"]["task_weights"],
        config["loss"]["misconception_focal_gamma"],
    ).to(device)
    loss, logits, labels = collect(model, loader, loss_function, device)
    metrics = evaluate_logits(logits, labels, thresholds)
    metrics["evaluation_split"] = "test"
    metrics["loss"] = loss
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote test metrics for {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
