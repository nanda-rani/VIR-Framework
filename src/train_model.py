#!/usr/bin/env python3
"""Train, validate, threshold-tune, and test the paper's VIR model."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ml_data import TASK_FIELDS, VIRCollator, VIRDataset, deterministic_split, load_rows
from ml_metrics import evaluate_logits, tune_thresholds
from modeling import MultiTaskHingRoBERTa, VIRLoss, compute_training_weights


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch, device):
    inputs = {key: value.to(device) for key, value in batch["inputs"].items()}
    labels = {key: value.to(device) for key, value in batch["labels"].items()}
    return inputs, labels


def collect(model, loader, loss_function, device):
    model.eval()
    losses = []
    logits = {task: [] for task in TASK_FIELDS}
    labels = {task: [] for task in TASK_FIELDS}
    with torch.no_grad():
        for batch in loader:
            inputs, batch_labels = move_batch(batch, device)
            batch_logits = model(**inputs)
            loss, _ = loss_function(batch_logits, batch_labels)
            losses.append(float(loss.detach().cpu()))
            for task in TASK_FIELDS:
                logits[task].append(batch_logits[task].detach().cpu().numpy())
                labels[task].append(batch_labels[task].detach().cpu().numpy())
    return float(np.mean(losses)), {k: np.concatenate(v) for k, v in logits.items()}, {k: np.concatenate(v) for k, v in labels.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/training_config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vir_model"))
    parser.add_argument("--base-model", help="Override the configured encoder, useful for a tiny smoke test")
    parser.add_argument("--epochs", type=int, help="Override maximum epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--max-length", type=int, help="Override tokenizer sequence length")
    parser.add_argument("--max-train-batches", type=int, help="Limit batches per epoch for a smoke test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.base_model:
        config["base_model"] = args.base_model
    if args.epochs:
        config["optimization"]["maximum_epochs"] = args.epochs
    if args.batch_size:
        config["optimization"]["batch_size"] = args.batch_size
    if args.max_length:
        config["data"]["maximum_sequence_length"] = args.max_length
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    optimization = config["optimization"]
    data_config = config["data"]
    architecture = config["architecture"]
    seed_everything(optimization["random_seed"])
    device = choose_device(args.device)

    rows = load_rows(args.data, data_config["text_column"])
    splits = deterministic_split(
        rows,
        optimization["random_seed"],
        data_config["train_ratio"],
        data_config["validation_ratio"],
        data_config["stratification_column"],
    )
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    collator = VIRCollator(tokenizer, data_config["maximum_sequence_length"], data_config["text_column"])
    loaders = {
        name: DataLoader(
            VIRDataset(split_rows),
            batch_size=optimization["batch_size"],
            shuffle=name == "train",
            collate_fn=collator,
        )
        for name, split_rows in splits.items()
    }

    model = MultiTaskHingRoBERTa(config["base_model"], **architecture).to(device)
    class_weights, positive_weights = compute_training_weights(splits["train"])
    loss_function = VIRLoss(
        class_weights,
        positive_weights,
        config["loss"]["task_weights"],
        config["loss"]["misconception_focal_gamma"],
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=optimization["learning_rate"], weight_decay=optimization["weight_decay"])
    accumulation = optimization["gradient_accumulation_steps"]
    batches_per_epoch = len(loaders["train"])
    if args.max_train_batches:
        batches_per_epoch = min(batches_per_epoch, args.max_train_batches)
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_updates = updates_per_epoch * optimization["maximum_epochs"]
    warmup_steps = round(total_updates * optimization["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    checkpoint = args.output_dir / "model_state.pt"
    for epoch in range(1, optimization["maximum_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        processed_batches = 0
        for batch_number, batch in enumerate(loaders["train"], start=1):
            if args.max_train_batches and batch_number > args.max_train_batches:
                break
            inputs, labels = move_batch(batch, device)
            logits = model(**inputs)
            loss, _ = loss_function(logits, labels)
            (loss / accumulation).backward()
            train_losses.append(float(loss.detach().cpu()))
            processed_batches += 1
            if batch_number % accumulation == 0 or batch_number == batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(trainable, optimization["maximum_gradient_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation_loss, _, _ = collect(model, loaders["validation"], loss_function, device)
        record = {"epoch": epoch, "training_loss": float(np.mean(train_losses)), "validation_loss": validation_loss}
        history.append(record)
        print(json.dumps(record))
        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= optimization["early_stopping_patience"]:
                break

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    validation_loss, validation_logits, validation_labels = collect(model, loaders["validation"], loss_function, device)
    threshold_config = config["threshold_tuning"]
    thresholds = tune_thresholds(
        validation_logits,
        validation_labels,
        threshold_config["minimum"],
        threshold_config["maximum"],
        threshold_config["step"],
    )
    validation_metrics = evaluate_logits(validation_logits, validation_labels, thresholds)
    test_loss, test_logits, test_labels = collect(model, loaders["test"], loss_function, device)
    test_metrics = evaluate_logits(test_logits, test_labels, thresholds)
    validation_metrics["loss"] = validation_loss
    test_metrics["loss"] = test_loss
    metadata = {
        "device": str(device),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    for name, value in (
        ("thresholds.json", thresholds),
        ("validation_metrics.json", validation_metrics),
        ("test_metrics.json", test_metrics),
        ("training_history.json", history),
        ("run_metadata.json", metadata),
    ):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
