PYTHON ?= python
.PHONY: data labels eval eval-dev test
data:
	$(PYTHON) scripts/prepare_squad.py
labels:
	$(PYTHON) -m src.evals.benchmark --split dev --prepare-labels
eval:
	$(PYTHON) -m src.evals.benchmark --split all
eval-dev:
	$(PYTHON) -m src.evals.benchmark --split dev
test:
	$(PYTHON) -m pytest tests -q
