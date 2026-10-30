.PHONY: setup train evaluate serve test lint docker

PYTHON ?= python3.12
VENV_PYTHON = .venv/bin/python

setup:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install -e '.[dev]'
train:
	PYTHONPATH=src $(VENV_PYTHON) scripts/run_pipeline.py
evaluate: train
serve:
	PYTHONPATH=src $(VENV_PYTHON) -m uvicorn api.main:app --reload
test:
	$(VENV_PYTHON) -m pytest
lint:
	$(VENV_PYTHON) -m ruff check .
docker:
	docker compose up --build
