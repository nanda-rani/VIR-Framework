# Annotation and Adjudication Protocol

## Annotator preparation

Annotators read the full VIR codebook, discuss invented practice examples, and
complete a pilot batch independently. The pilot is used to align interpretation
of definitions and decision boundaries; it is not used to alter labels merely
to maximize agreement.

## Independent annotation

1. Assign all seven VIR dimensions independently.
2. Use only evidence expressed in the narrative.
3. Apply multi-label fields independently.
4. Mark genuinely ambiguous decisions for later discussion.
5. Do not include verbatim complaint passages in notes or exported audit files.

The pre-adjudication annotations must be preserved separately from the final
consensus labels so that agreement is calculated before discussion.

## Pre-adjudication inter-annotator agreement

- Attack vector: accuracy and a class-wise confusion matrix.
- Attribution and misconception: accuracy and Cohen's kappa.
- Technical awareness: accuracy, within-one-level accuracy, and Spearman's rho.
- Emotion, requested assistance, and action: per-record Jaccard similarity and
  per-label measures where appropriate.

## Adjudication

After pre-adjudication metrics are fixed, annotators discuss disagreements by
referring to the codebook and the available narrative evidence. The objective
is a defensible consensus label, not agreement with a model. If the evidence
cannot support a specific interpretation, the least specific permissible label
is used. The final consensus record must not overwrite the original independent
annotations.

## Sampling and partitions reported in the manuscript

The manuscript reports one stratified gold sample of 1,998 complaints. After
adjudication, the complete gold sample is divided once into training,
validation, and held-out test partitions in a 70/15/15 ratio. The training
partition is used for parameter estimation, the validation partition for early
stopping and decision-threshold selection, and the test partition only for the
final predictive evaluation and Human-AI agreement analysis. There is no
separate 500-record agreement partition. Exact split membership is not public
because it is complaint-level information.

Pre-adjudication inter-annotator agreement and final Human-AI agreement are
different analyses. The former compares the two independent human annotations
before discussion. The latter compares model predictions with the final
adjudicated expert labels on the held-out test partition.
