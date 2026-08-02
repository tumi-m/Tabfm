#!/bin/sh
# Container entrypoint.
#
# Goal: `docker run -p 8000:8000 <image>` produces a working service with no
# second command. If the image was built with BAKE_ARTIFACTS=true the models are
# already inside and startup is instant; otherwise the first boot trains them.
#
# Explicit commands are passed straight through, so `docker run <image>
# tabfm-lab-train ...` and the Kubernetes training Job both still work and skip
# the auto-train branch entirely.

set -eu

ARTIFACT_DIR="${TABFM_ARTIFACT_DIR:-/var/lib/tabfm/artifacts}"
DATA_DIR="${TABFM_DATA_DIR:-/var/lib/tabfm/data}"

# Anything other than the `serve` keyword is a command to run instead.
if [ "$#" -gt 0 ] && [ "$1" != "serve" ]; then
  exec "$@"
fi

has_artifacts() {
  # A glob that matches nothing stays literal, so test for a real file.
  for candidate in "$ARTIFACT_DIR"/*.pkl; do
    [ -e "$candidate" ] && return 0
  done
  return 1
}

if has_artifacts; then
  echo "[entrypoint] Found existing artifacts in $ARTIFACT_DIR."
elif [ "${AUTO_TRAIN:-1}" = "1" ]; then
  echo "[entrypoint] No artifacts in $ARTIFACT_DIR — training on first boot."
  echo "[entrypoint] This downloads the public datasets and takes a few minutes."
  mkdir -p "$ARTIFACT_DIR" "$DATA_DIR"
  tabfm-lab-train \
    --output "$ARTIFACT_DIR" \
    --data-dir "$DATA_DIR" \
    --model "${TRAIN_MODEL:-auto}"
  echo "[entrypoint] Training complete."
else
  # Deliberate in Kubernetes: a Job populates a shared read-only volume, so a
  # pod that trained on its own would be both wasteful and wrong. Readiness
  # stays 503 until the volume is filled.
  echo "[entrypoint] No artifacts and AUTO_TRAIN=0; starting anyway."
  echo "[entrypoint] /api/ready will report 503 until artifacts appear."
fi

exec uvicorn tabfm_lab.serving.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers "${WEB_CONCURRENCY:-1}"
