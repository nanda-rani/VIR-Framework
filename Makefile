.PHONY: demo clean

export PYTHONDONTWRITEBYTECODE=1
PYTHON ?= python3

demo:
	$(PYTHON) src/validate_schema.py synthetic/vir_synthetic.csv
	$(PYTHON) src/evaluate_predictions.py synthetic/vir_synthetic.csv results/demo_test_evaluation.json --split test
	$(PYTHON) src/aggregate_analysis.py synthetic/vir_synthetic.csv results/demo_aggregates.json --min-cell-size 5

clean:
	rm -f results/demo_test_evaluation.json results/demo_aggregates.json
