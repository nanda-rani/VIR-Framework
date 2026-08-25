"""Threshold tuning and aggregate metrics for the seven VIR dimensions."""

import math

import numpy as np

from vir_fields import ACTIONS, ASSISTANCE, EMOTIONS


MULTILABEL_NAMES = {"emotions": EMOTIONS, "assistance": ASSISTANCE, "actions": ACTIONS}


def sigmoid(values):
    values = np.clip(values, -40, 40)
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values):
    shifted = values - values.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=1, keepdims=True)


def f1_for_class(gold, predicted, positive):
    gold = np.asarray(gold)
    predicted = np.asarray(predicted)
    tp = np.sum((gold == positive) & (predicted == positive))
    fp = np.sum((gold != positive) & (predicted == positive))
    fn = np.sum((gold == positive) & (predicted != positive))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else float(2 * tp / denominator)


def macro_f1(gold, predicted, classes):
    return float(np.mean([f1_for_class(gold, predicted, value) for value in classes]))


def accuracy(gold, predicted):
    return float(np.mean(np.asarray(gold) == np.asarray(predicted)))


def cohen_kappa(gold, predicted):
    gold = np.asarray(gold)
    predicted = np.asarray(predicted)
    observed = accuracy(gold, predicted)
    values = sorted(set(gold.tolist()) | set(predicted.tolist()))
    expected = sum(np.mean(gold == value) * np.mean(predicted == value) for value in values)
    return 1.0 if expected == 1.0 else float((observed - expected) / (1.0 - expected))


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_rho(gold, predicted):
    left = rankdata(gold)
    right = rankdata(predicted)
    left -= left.mean()
    right -= right.mean()
    denominator = math.sqrt(float(np.sum(left ** 2) * np.sum(right ** 2)))
    return 0.0 if denominator == 0 else float(np.sum(left * right) / denominator)


def tune_thresholds(logits, labels, minimum=0.05, maximum=0.95, step=0.05):
    candidates = np.arange(minimum, maximum + step / 2, step)
    thresholds = {}
    for task in ("attribution", "misconception", "emotions", "assistance", "actions"):
        probabilities = sigmoid(logits[task])
        gold = labels[task].astype(int)
        task_thresholds = []
        for column in range(probabilities.shape[1]):
            best_threshold = 0.5
            best_score = -1.0
            for threshold in candidates:
                predicted = (probabilities[:, column] >= threshold).astype(int)
                score = macro_f1(gold[:, column], predicted, (0, 1))
                if score > best_score:
                    best_score = score
                    best_threshold = float(round(threshold, 10))
            task_thresholds.append(best_threshold)
        thresholds[task] = task_thresholds
    return thresholds


def evaluate_logits(logits, labels, thresholds):
    attack_gold = labels["attack_vector"].astype(int).reshape(-1)
    awareness_gold = labels["awareness"].astype(int).reshape(-1)
    attack_pred = np.argmax(softmax(logits["attack_vector"]), axis=1)
    awareness_pred = np.argmax(softmax(logits["awareness"]), axis=1)

    results = {
        "records": int(len(attack_gold)),
        "attack_vector": {
            "accuracy": accuracy(attack_gold, attack_pred),
            "macro_f1": macro_f1(attack_gold, attack_pred, range(6)),
            "per_class_f1": [f1_for_class(attack_gold, attack_pred, value) for value in range(6)],
            "confusion_matrix": [
                [int(np.sum((attack_gold == gold_value) & (attack_pred == pred_value))) for pred_value in range(6)]
                for gold_value in range(6)
            ],
        },
        "technical_awareness": {
            "accuracy": accuracy(awareness_gold, awareness_pred),
            "within_one_level_accuracy": float(np.mean(np.abs(awareness_gold - awareness_pred) <= 1)),
            "macro_f1": macro_f1(awareness_gold, awareness_pred, range(3)),
            "spearman_rho": spearman_rho(awareness_gold, awareness_pred),
        },
    }

    for task in ("attribution", "misconception"):
        gold = labels[task].astype(int).reshape(-1)
        predicted = (sigmoid(logits[task]).reshape(-1) >= thresholds[task][0]).astype(int)
        results[task] = {
            "accuracy": accuracy(gold, predicted),
            "macro_f1": macro_f1(gold, predicted, (0, 1)),
            "cohen_kappa": cohen_kappa(gold, predicted),
            "threshold": thresholds[task][0],
        }

    for task, names in MULTILABEL_NAMES.items():
        gold = labels[task].astype(int)
        threshold_array = np.asarray(thresholds[task])[None, :]
        predicted = (sigmoid(logits[task]) >= threshold_array).astype(int)
        jaccard = []
        for gold_row, predicted_row in zip(gold, predicted):
            union = np.sum((gold_row == 1) | (predicted_row == 1))
            intersection = np.sum((gold_row == 1) & (predicted_row == 1))
            jaccard.append(1.0 if union == 0 else intersection / union)
        per_label = {
            name: f1_for_class(gold[:, column], predicted[:, column], 1)
            for column, name in enumerate(names)
        }
        results[task] = {
            "mean_jaccard": float(np.mean(jaccard)),
            "macro_f1": float(np.mean(list(per_label.values()))),
            "per_label_f1": per_label,
            "thresholds": {name: thresholds[task][column] for column, name in enumerate(names)},
        }
    return results
