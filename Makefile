.PHONY: help install install-tabfm test lint run run-baselines train serve \
        docker-build docker-up k8s-apply k8s-delete clean

ARTIFACTS ?= artifacts
PORT ?= 8000
IMAGE ?= tabfm-lab-api:local

help:
	@echo "install         Install the package, API extras and dev tools"
	@echo "install-tabfm   Add TabFM with the CPU build of torch"
	@echo "test            Run the unit tests (no network needed)"
	@echo "lint            Run ruff"
	@echo "run             Full benchmark: TabFM plus baselines"
	@echo "run-baselines   Benchmark without TabFM"
	@echo "app             Run the Streamlit app (simplest; opens a browser)"
	@echo "train           Fit models and write pickled artifacts to $(ARTIFACTS)/"
	@echo "serve           Run the API and UI on http://localhost:$(PORT)"
	@echo "docker-build    Build the serving image ($(IMAGE))"
	@echo "docker-up       One command: build, train on first boot, serve"
	@echo "docker-retrain  Refresh the models in the compose volume"
	@echo "k8s-apply       Apply the Kubernetes manifests"
	@echo "clean           Remove caches, reports, artifacts and build output"

install:
	python -m pip install -e ".[api,dev]"

# CPU wheel first, otherwise pip pulls ~3 GB of CUDA libraries on a CPU-only box.
install-tabfm:
	python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	python -m pip install "tabfm[pytorch]"

test:
	python -m pytest

lint:
	python -m ruff check src tests streamlit_app.py

# Trains on first run if artifacts/ is empty, then opens in a browser.
app:
	python -m streamlit run streamlit_app.py

run:
	python -m tabfm_lab.cli

run-baselines:
	python -m tabfm_lab.cli --no-tabfm

train:
	python -m tabfm_lab.serving.train --output $(ARTIFACTS)

serve:
	TABFM_ARTIFACT_DIR=$(ARTIFACTS) python -m uvicorn tabfm_lab.serving.app:app \
		--host 127.0.0.1 --port $(PORT) --reload

docker-build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

# The entrypoint trains on first boot when the volume is empty, so this is the
# only command needed.
docker-up:
	docker compose -f docker/docker-compose.yml up

docker-retrain:
	docker compose -f docker/docker-compose.yml run --rm retrain

k8s-apply:
	kubectl apply -k k8s/

k8s-delete:
	kubectl delete -k k8s/ --ignore-not-found

clean:
	rm -rf reports/*.json reports/*.md artifacts/*.pkl .pytest_cache .ruff_cache build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
