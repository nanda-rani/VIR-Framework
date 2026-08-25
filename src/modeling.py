"""Shared-encoder, multi-head HingRoBERTa model described in the paper."""

import torch
from torch import nn
from transformers import AutoModel


OUTPUT_DIMENSIONS = {
    "attack_vector": 6,
    "awareness": 3,
    "attribution": 1,
    "misconception": 1,
    "emotions": 5,
    "assistance": 5,
    "actions": 4,
}


class MultiTaskHingRoBERTa(nn.Module):
    def __init__(self, base_model, projection_dimension=256, dropout=0.2, freeze_embeddings=True, freeze_lower_encoder_layers=6):
        super().__init__()
        self.base_model_name = base_model
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size
        self.projections = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(hidden_size, projection_dimension),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for task in OUTPUT_DIMENSIONS
            }
        )
        self.heads = nn.ModuleDict(
            {task: nn.Linear(projection_dimension, size) for task, size in OUTPUT_DIMENSIONS.items()}
        )
        self.freeze_encoder(freeze_embeddings, freeze_lower_encoder_layers)

    def freeze_encoder(self, freeze_embeddings, lower_layers):
        base = getattr(self.encoder, "roberta", self.encoder)
        if freeze_embeddings and hasattr(base, "embeddings"):
            for parameter in base.embeddings.parameters():
                parameter.requires_grad = False
        encoder = getattr(base, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            layers = getattr(encoder, "layers", [])
        for layer in list(layers)[:lower_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def forward(self, **inputs):
        encoded = self.encoder(**inputs)
        pooled = encoded.last_hidden_state[:, 0]
        return {task: self.heads[task](self.projections[task](pooled)) for task in OUTPUT_DIMENSIONS}


class VIRLoss(nn.Module):
    def __init__(self, class_weights, positive_weights, task_weights, focal_gamma=2.0):
        super().__init__()
        self.register_buffer("attack_weights", class_weights["attack_vector"])
        self.register_buffer("awareness_weights", class_weights["awareness"])
        for task in ("attribution", "misconception", "emotions", "assistance", "actions"):
            self.register_buffer(f"{task}_positive_weights", positive_weights[task])
        self.task_weights = task_weights
        self.focal_gamma = focal_gamma

    def forward(self, logits, labels):
        losses = {
            "attack_vector": nn.functional.cross_entropy(logits["attack_vector"], labels["attack_vector"], weight=self.attack_weights),
            "awareness": nn.functional.cross_entropy(logits["awareness"], labels["awareness"], weight=self.awareness_weights),
            "attribution": nn.functional.binary_cross_entropy_with_logits(
                logits["attribution"], labels["attribution"], pos_weight=self.attribution_positive_weights
            ),
            "emotions": nn.functional.binary_cross_entropy_with_logits(
                logits["emotions"], labels["emotions"], pos_weight=self.emotions_positive_weights
            ),
            "assistance": nn.functional.binary_cross_entropy_with_logits(
                logits["assistance"], labels["assistance"], pos_weight=self.assistance_positive_weights
            ),
            "actions": nn.functional.binary_cross_entropy_with_logits(
                logits["actions"], labels["actions"], pos_weight=self.actions_positive_weights
            ),
        }
        misconception_bce = nn.functional.binary_cross_entropy_with_logits(
            logits["misconception"],
            labels["misconception"],
            pos_weight=self.misconception_positive_weights,
            reduction="none",
        )
        misconception_probability = torch.sigmoid(logits["misconception"])
        correct_probability = torch.where(labels["misconception"] == 1, misconception_probability, 1 - misconception_probability)
        losses["misconception"] = (((1 - correct_probability) ** self.focal_gamma) * misconception_bce).mean()
        total_weight = sum(self.task_weights[task] for task in losses)
        total = sum(self.task_weights[task] * value for task, value in losses.items()) / total_weight
        return total, losses


def compute_training_weights(rows):
    def multiclass(field, classes):
        counts = torch.tensor([sum(int(row[field]) == value for row in rows) for value in range(classes)], dtype=torch.float)
        counts = torch.clamp(counts, min=1)
        weights = len(rows) / (classes * counts)
        return weights / weights.mean()

    def positive(fields):
        positives = torch.tensor([sum(int(row[field]) for row in rows) for field in fields], dtype=torch.float)
        positives = torch.clamp(positives, min=1)
        negatives = len(rows) - positives
        return torch.clamp(negatives / positives, min=0.1, max=20.0)

    from vir_fields import ACTIONS, ASSISTANCE, EMOTIONS

    class_weights = {"attack_vector": multiclass("attack_vector", 6), "awareness": multiclass("awareness", 3)}
    positive_weights = {
        "attribution": positive(["attribution"]),
        "misconception": positive(["misconception"]),
        "emotions": positive(EMOTIONS),
        "assistance": positive(ASSISTANCE),
        "actions": positive(ACTIONS),
    }
    return class_weights, positive_weights
