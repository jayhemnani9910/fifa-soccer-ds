VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP ?= $(PYTHON) -m pip
URL ?= 0
FRAMES_DIR ?= data/processed/sample
OUTPUT_DIR ?= outputs/pipeline_run
CHECK_PATHS := src tests scripts

.PHONY: setup setup-all build check lint test test-cov smoke run run-detect run-live run-pipeline fmt lint-all export trt clean

setup:
	$(PIP) install --upgrade "pip==26.1.2" "setuptools==83.0.0" "wheel==0.47.0"
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m pre_commit install

setup-all:
	$(PIP) install -e ".[audio,data,dev,export,mlops]"

build:
	$(PYTHON) -I -m build

check: lint test

lint:
	$(PYTHON) -m ruff check $(CHECK_PATHS)
	$(PYTHON) -m ruff format --check $(CHECK_PATHS)
	$(PYTHON) -m mypy src

fmt:
	$(PYTHON) -m ruff check --fix $(CHECK_PATHS)
	$(PYTHON) -m ruff format $(CHECK_PATHS)

lint-all:
	$(PYTHON) -m pre_commit run --all-files

test:
	$(PYTHON) -m pytest

test-cov:
	$(PYTHON) -m pytest --cov=src --cov-branch --cov-report=term-missing

smoke:
	$(PYTHON) -m pytest -m smoke

run:
	$(PYTHON) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

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
	$(PYTHON) -m src.detect.export_onnx

trt:
	@echo "Use: python -m src.detect.export_trt --onnx MODEL.onnx --output MODEL.plan --fp16"

clean:
	@echo "Refusing to delete outputs automatically; remove generated paths explicitly after review."
