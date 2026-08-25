# Multi-Task HingRoBERTa Pipeline

## Architecture

The implementation loads `l3cube-pune/hing-roberta` as a shared text encoder.
The first-token representation is passed to seven independent task-specific
projection layers consisting of a linear transformation, GELU activation, and
dropout. Each projection feeds its own prediction head:

| Head | Output |
|---|---:|
| Attack vector | 6-class logits |
| Attribution | 1 binary logit |
| Technical awareness | 3 ordinal-class logits |
| Misconception | 1 binary logit |
| Emotional expression | 5 independent logits |
| Requested assistance | 5 independent logits |
| Post-incident action | 4 independent logits |

The embedding parameters and the first six transformer layers are frozen. All
remaining encoder layers, projections, and heads are optimized jointly.

## Objective

- Attack vector and awareness: inverse-frequency-weighted cross-entropy.
- Attribution, emotions, assistance, and actions: binary cross-entropy with a
  training-set positive-class weight for each output.
- Misconception: class-weighted focal binary cross-entropy with gamma 2.
- Multi-task objective: equally weighted mean of the seven task losses.

The optimizer is AdamW with learning rate `2e-5` and weight decay `0.01`.
Gradients are accumulated for two batches and clipped to norm 1.0. A linear
schedule uses a 10% warm-up. The deterministic seed is 42.

## Partitions and selection

If the CSV contains explicit `train`, `validation`, and `test` values in the
`split` column, the implementation preserves them. Otherwise, it performs a
deterministic 70/15/15 stratified split, using `category` when present and
attack vector as the fallback stratum.

Training stops when validation loss does not improve for three consecutive
epochs. The best validation-loss checkpoint is retained. Single-label heads
use argmax. Binary and multi-label thresholds are chosen on the validation set
from 0.05 through 0.95 in steps of 0.05 by maximizing macro-F1. These frozen
thresholds are then used for held-out testing and label propagation.

## Evaluation

- Attack vector: accuracy, macro-F1, per-class F1, and confusion matrix.
- Attribution and misconception: accuracy, macro-F1, and Cohen's kappa.
- Awareness: accuracy, within-one-level accuracy, macro-F1, and Spearman's rho.
- Multi-label outputs: mean per-record Jaccard similarity, macro-F1, and
  per-label F1.

## Privacy behavior

Checkpoints and complaint-level predictions are ignored by the public artifact.
The inference command writes record identifiers, labels, confidence values, and
probabilities; it never exports narrative text. Authorized data, checkpoints,
and predictions should remain outside the public artifact directory.
