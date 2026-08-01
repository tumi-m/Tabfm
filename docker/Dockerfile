# syntax=docker/dockerfile:1
#
# Serving image for the TabFM Lab API and its static UI.
#
# TabFM is OFF by default. The serving path only needs numpy/pandas/sklearn
# unless an artifact actually contains a TabFM model, and installing torch
# multiplies the image size by roughly ten. Build with --build-arg
# INSTALL_TABFM=true when you are serving TabFM artifacts; the CPU wheel index
# is used so a GPU-less node does not pull the CUDA libraries.

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- builder ---
FROM python:${PYTHON_VERSION}-slim AS builder

ARG INSTALL_TABFM=false

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN python -m venv /opt/venv

WORKDIR /build

# Dependency metadata first, so the dependency layer survives source edits.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install ".[api]" \
 && if [ "$INSTALL_TABFM" = "true" ]; then \
      python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && \
      python -m pip install "tabfm[pytorch]"; \
    fi

# ---------------------------------------------------------------- runtime ---
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="tabfm-lab-api" \
      org.opencontainers.image.description="TabFM and baseline models for e-commerce and sports-betting predictions" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/tumi-m/Tabfm"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    TABFM_ARTIFACT_DIR=/var/lib/tabfm/artifacts \
    TABFM_FRONTEND_DIR=/app/frontend \
    PORT=8000

# curl is used by the container HEALTHCHECK below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 tabfm

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY frontend/ ./frontend/

# Artifacts are supplied at runtime — a PVC, an init container, or a bind mount
# — so the image stays model-agnostic and does not need rebuilding to ship a
# retrained model.
RUN mkdir -p /var/lib/tabfm/artifacts && chown -R tabfm:tabfm /var/lib/tabfm /app

USER tabfm
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# Single worker by default: each worker unpickles its own copy of every model,
# so scale with replicas (see k8s/) rather than in-process workers.
CMD ["sh", "-c", "exec uvicorn tabfm_lab.serving.app:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}"]
