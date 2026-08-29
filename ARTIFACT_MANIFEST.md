# Artifact Manifest and Claim Mapping

Status meanings:

- **Available**: included in the anonymous package and intended for permanent
  public release after acceptance.
- **Unavailable**: intentionally withheld for the reason stated.

| Paper component or claim | Artifact | Status | Location or explanation |
|---|---|---|---|
| Definition of the seven VIR dimensions | Annotation codebook | Available | `documentation/VIR_CODEBOOK.md` |
| Permitted values and field types | Machine-readable schema | Available | `schema/vir_schema.json` |
| Annotation and adjudication procedure | Protocol description | Available | `documentation/ANNOTATION_PROTOCOL.md` |
| Demonstration of the full file format | Synthetic labelled/predicted records | Available | `synthetic/vir_synthetic.csv` |
| Aggregate VIR distributions and cross-tabulations | Aggregate analysis code | Available | `src/aggregate_analysis.py` |
| Accuracy, macro-F1, Cohen's kappa, Spearman correlation, and Jaccard calculations | Evaluation code | Available | `src/evaluate_predictions.py` |
| Multi-head HingRoBERTa method | Model implementation | Available | `src/modeling.py` |
| Training, validation, threshold tuning, early stopping, and held-out testing | Experiment pipeline | Available | `src/train_model.py` and `src/test_model.py` |
| Model label propagation | Inference pipeline | Available | `src/predict_model.py` |
| Model, loss, optimizer, split, and threshold settings | Experiment configuration | Available | `config/training_config.json` |
| Model architecture, objective, selection, and metric documentation | Model pipeline description | Available | `documentation/MODEL_PIPELINE.md` |
| Numerical values appearing in the manuscript | Machine-readable aggregate record | Available | `results/reported_aggregate_results.json` |
| Raw corpus of 118,848 NCRP complaints | Complaint narratives | Unavailable | No redistribution authorization and material privacy risk |
| Gold corpus of 1,998 expert-annotated complaints, partitioned once into stratified training, validation, and held-out test sets (70/15/15) | Complaint-level annotations | Unavailable | Labels and split membership remain linked to restricted narratives |
| Silver corpus of 116,850 propagated labels | Complaint-level predictions | Unavailable | Sensitive, linkable derived records |
| Fine-tuned weights | Model checkpoint | Unavailable | Withheld because memorization of sensitive narratives has not been ruled out; the training source and configuration are provided |

## Artifact-to-paper navigation

- Framework and annotation sections: codebook, protocol, and schema.
- Human-AI pipeline: held-out-test evaluation code, run metadata, and reported aggregate results.
- Large-scale analysis: aggregate analysis code and reported aggregate results.

The synthetic dataset is solely a functional test fixture; it is not evidence
for any empirical claim.
