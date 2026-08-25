# VIR Cybercrime-Complaint Study

This artifact documents the Victim Interpretation and Response (VIR) framework and provides
code for validating VIR-formatted data, calculating aggregate statistics, and
evaluating predictions. It also implements the shared-encoder, multi-head
HingRoBERTa training, validation, threshold-tuning, testing, and inference
pipeline described in the paper. A wholly invented dataset permits an
end-to-end test without exposing any real complaint.

## Important data restriction

The original National Cybercrime Reporting Portal (NCRP) narratives are not
included. The authors do not own the data and do not have permission to
redistribute it. Even after removal/anonymization of direct identifiers, free-text victim
narratives may create residual disclosure and re-identification risks.

The following are deliberately absent:

- raw or processed complaint narratives;
- human-labelled or model-labelled complaint-level records;
- row identifiers, data splits, embeddings, token caches, and raw model output;
- trained weights fine-tuned on the complaints; and
- any intermediate representation from which complaint text might be inferred.

Please refer to `DATA_AVAILABILITY.md` for the complete rationale and the substitute
supplied for each unavailable artifact.

## Included materials

- `documentation/VIR_CODEBOOK.md`: all seven VIR dimensions and labels;
- `documentation/ANNOTATION_PROTOCOL.md`: annotation and adjudication process;
- `schema/vir_schema.json`: machine-readable field definitions;
- `synthetic/vir_synthetic.csv`: invented, non-representative test records;
- `src/aggregate_analysis.py`: aggregate analysis with small-cell suppression;
- `src/evaluate_predictions.py`: aggregate prediction-evaluation metrics;
- `src/modeling.py`: partially frozen shared HingRoBERTa encoder and seven
  task-specific projection/classification heads;
- `src/train_model.py`: stratified splitting, weighted multi-task training,
  early stopping, validation threshold tuning, and held-out testing;
- `src/test_model.py`: evaluation of a saved checkpoint;
- `src/predict_model.py`: label propagation without narrative export;
- `config/training_config.json`: complete artifact configuration;

`ARTIFACT_MANIFEST.md` maps claims to artifacts or omission explanations.

## Quick start

Requirements: Python 3.10 or newer. The schema, aggregate, and release
demonstration uses the standard library. Model training additionally uses the
versions in `requirements.txt`.

```bash
make demo
```

The expanded commands are:

```bash
python3 src/generate_synthetic_data.py
python3 src/validate_schema.py synthetic/vir_synthetic.csv
python3 src/evaluate_predictions.py \
  synthetic/vir_synthetic.csv results/demo_evaluation.json
python3 src/aggregate_analysis.py \
  synthetic/vir_synthetic.csv results/demo_aggregates.json --min-cell-size 5
```

The synthetic outputs demonstrate code execution only. They must not be
interpreted as approximations of the empirical findings.

## Model training, validation, and testing

Install the model dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 src/train_model.py \
  --data synthetic/vir_synthetic.csv \
  --config config/training_config.json \
  --output-dir runs/vir_model
python3 src/test_model.py \
  --data synthetic/vir_synthetic.csv \
  --run-dir runs/vir_model \
  --output results/model_test_metrics.json
```

The training pipeline uses `l3cube-pune/hing-roberta`, a shared encoder,
task-specific projection layers, seven task-specific output groups covering all
seven VIR dimensions, class-weighted cross-entropy and binary cross-entropy,
focal loss for misconception, partial encoder freezing, validation-loss early
stopping, and validation-set macro-F1 threshold selection. Single-label heads
use argmax decoding.

For label propagation, write outputs outside the public artifact directory:

```bash
python3 src/predict_model.py \
  --data /authorized/private/input.csv \
  --run-dir runs/vir_model \
  --output /authorized/private/predictions.csv
```

The prediction export contains IDs, labels, and probabilities but never copies
narrative text.

## License

Code is released under the MIT License. Documentation, the codebook, schema,
and synthetic example are released under CC BY 4.0. Neither license applies to
the unavailable NCRP dataset. `THIRD_PARTY_NOTICES.md` records the encoder's
separate attribution and license.
