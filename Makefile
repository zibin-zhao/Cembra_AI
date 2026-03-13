.PHONY: help install test lint toy-data pipeline clean

PYTHON ?= python
PIP ?= pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode with all dependencies
	$(PIP) install -e ".[all]" --break-system-packages

install-cpu:  ## Install without GPU dependencies
	$(PIP) install -e ".[dev]" --break-system-packages

test:  ## Run unit tests
	pytest tests/unit -v --tb=short

test-integration:  ## Run integration tests (requires toy data)
	pytest tests/integration -v --tb=short -m "not gpu"

test-all:  ## Run all tests
	pytest tests/ -v --tb=short

lint:  ## Run linter
	ruff check src/ tests/
	mypy src/oa_prs/ --ignore-missing-imports

format:  ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

toy-data:  ## Generate synthetic toy data
	$(PYTHON) scripts/generate_toy_data.py --output-dir data/toy/

pipeline-toy:  ## Run full pipeline on toy data (local, no HPC)
	oa-prs run --config configs/config.yaml data.mode=toy

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
