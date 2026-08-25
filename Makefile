.PHONY: demo model-smoke validate clean

export PYTHONDONTWRITEBYTECODE=1
PYTHON ?= python3
SMOKE_DIR ?= /tmp/vir-artifact-smoke-run

demo:
	$(PYTHON) src/generate_synthetic_data.py
	$(PYTHON) src/validate_schema.py synthetic/vir_synthetic.csv
	$(PYTHON) src/evaluate_predictions.py synthetic/vir_synthetic.csv results/demo_evaluation.json
	$(PYTHON) src/aggregate_analysis.py synthetic/vir_synthetic.csv results/demo_aggregates.json --min-cell-size 5

validate:
	$(PYTHON) src/validate_release.py .

model-smoke:
	$(PYTHON) src/train_model.py --data synthetic/vir_synthetic.csv --config config/training_config.json --output-dir $(SMOKE_DIR) --base-model hf-internal-testing/tiny-xlm-roberta --epochs 1 --batch-size 4 --max-length 48 --max-train-batches 2 --device cpu
	$(PYTHON) src/test_model.py --data synthetic/vir_synthetic.csv --run-dir $(SMOKE_DIR) --output $(SMOKE_DIR)/test_metrics_repeat.json --device cpu

clean:
	rm -f results/demo_evaluation.json results/demo_aggregates.json
