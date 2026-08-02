# syntax=docker/dockerfile:1
#
# Serving image for the TabFM Lab API and its static UI.
#
# The goal is that `docker run -p 8000:8000 <image>` is enough. Two build knobs
# shape how that happens:
#
#   BAKE_ARTIFACTS=true   Train during the build so the image ships with models
#                         and starts instantly. Needs network at build time.
#                         This is what CI publishes.
#   INSTALL_TABFM=true    Add TabFM itself. Off by default: serving only needs
#                         numpy/pandas/scikit-learn unless an artifact actually
#                         contains a TabFM model, and torch multiplies the image
#                         size by roughly ten. The CPU wheel index is used so a
#                         GPU-less node does not pull the CUDA libraries.
#
# With neither set, the entrypoint trains on first boot instead.

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- builder ---
FROM python:${PYTHON_VERSION}-slim AS builder

ARG INSTALL_TABFM=false
ARG BAKE_ARTIFACTS=false
ARG TRAIN_MODEL=auto

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

# Always create the directory so the runtime COPY below has a source whether or
# not artifacts were baked.
RUN mkdir -p /baked/artifacts \
 && if [ "$BAKE_ARTIFACTS" = "true" ]; then \
      tabfm-lab-train --output /baked/artifacts --data-dir /tmp/tabfm-data --model "$TRAIN_MODEL"; \
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
    TABFM_DATA_DIR=/var/lib/tabfm/data \
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
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Empty unless BAKE_ARTIFACTS was set, in which case the models ship inside.
COPY --from=builder --chown=tabfm:tabfm /baked/artifacts /var/lib/tabfm/artifacts

RUN mkdir -p /var/lib/tabfm/data && chown -R tabfm:tabfm /var/lib/tabfm /app

USER tabfm
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# Single worker by default: each worker unpickles its own copy of every model,
# so scale with replicas (see k8s/) rather than in-process workers.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
