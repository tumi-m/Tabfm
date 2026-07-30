.PHONY: help install install-tabfm test lint run run-baselines clean

help:
	@echo "install         Install the package and dev tools"
	@echo "install-tabfm   Add TabFM with the CPU build of torch"
	@echo "test            Run the unit tests (no network needed)"
	@echo "lint            Run ruff"
	@echo "run             Full benchmark: TabFM plus baselines"
	@echo "run-baselines   Benchmark without TabFM"
	@echo "clean           Remove caches, reports and build artefacts"

install:
	python -m pip install -e ".[dev]"

# CPU wheel first, otherwise pip pulls ~3 GB of CUDA libraries on a CPU-only box.
install-tabfm:
	python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	python -m pip install "tabfm[pytorch]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

run:
	python -m tabfm_lab.cli

run-baselines:
	python -m tabfm_lab.cli --no-tabfm

clean:
	rm -rf reports/*.json reports/*.md .pytest_cache .ruff_cache build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
