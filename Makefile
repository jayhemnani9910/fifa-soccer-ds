PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
VENV ?= .venv
URL ?= 0
FRAMES_DIR ?= data/processed/sample
OUTPUT_DIR ?= outputs/pipeline_run

.PHONY: setup lint test smoke run run-detect run-live run-pipeline fmt lint-all export trt clean

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	pre-commit install

lint:
	ruff check src tests
	ruff format --check src tests
	isort --check-only src tests
	black --check src tests
	mypy src

fmt:
	ruff format src tests
	isort src tests
	black src tests

lint-all:
	pre-commit run --all-files

test:
	pytest

smoke:
	pytest -m smoke

run:
	@echo "TODO: wire live pipeline entrypoint (see src/live/run_live.py)."

run-detect:
	$(PYTHON) -m src.detect.infer --output outputs/detect

run-live:
	$(PYTHON) -m src.live.run_live --source "$(URL)"

run-pipeline:
	$(PYTHON) -m src.pipeline_full \
		--frames-dir "$(FRAMES_DIR)" \
		--output-dir "$(OUTPUT_DIR)" \
		--distance-threshold 120.0 \
		--min-confidence 0.35 \
		--max-age 20

export:
	@echo "TODO: export trained weights to ONNX/TorchScript."

trt:
	@echo "TODO: integrate TensorRT conversion pipeline."

clean:
	rm -rf build/ dist/ .pytest_cache/ .coverage outputs/
