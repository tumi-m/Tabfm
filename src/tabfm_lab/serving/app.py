"""FastAPI service exposing the pickled models.

The request path deliberately does no data loading and no fitting: artifacts are
unpickled once at startup and every prediction is a transform plus a forward
pass. That is what makes the container's readiness probe meaningful — once it
reports ready, it can serve.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..evaluation.betting import implied_probabilities, poisson_over_probability
from .artifacts import ArtifactRegistry, ModelArtifact
from .schemas import (
    FixtureRequest,
    FixtureResponse,
    HealthResponse,
    ModelDetail,
    ModelSummary,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

LOGGER = logging.getLogger(__name__)

RESULT_TASK = "sports-match-result"
GOALS_TASK = "sports-total-goals"
OVER_UNDER_LINE = 2.5
#: Minimum expected value per unit staked before a bet is worth recommending.
EDGE_THRESHOLD = 0.05

#: 1X2 class label -> field on the submitted odds object.
_ODDS_FIELDS = {"H": "home", "D": "draw", "A": "away"}


def artifact_dir() -> Path:
    return Path(os.environ.get("TABFM_ARTIFACT_DIR", "artifacts"))


def frontend_dir() -> Path:
    configured = os.environ.get("TABFM_FRONTEND_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    directory = artifact_dir()
    LOGGER.info("Loading artifacts from %s", directory.resolve())
    app.state.registry = ArtifactRegistry(directory).load()
    LOGGER.info("Serving %d model(s): %s", len(app.state.registry),
                ", ".join(app.state.registry.names()) or "none")
    yield


def _registry(request: Request) -> ArtifactRegistry:
    return request.app.state.registry


def _artifact(request: Request, task: str) -> ModelArtifact:
    try:
        return _registry(request).get(task)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _finite(value: Any) -> float | None:
    """JSON has no NaN or Infinity, so unrepresentable numbers become null."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _summary(artifact: ModelArtifact) -> dict[str, Any]:
    meta = artifact.metadata
    return {
        "task": meta.task,
        "industry": meta.industry,
        "task_type": meta.task_type,
        "target": meta.target,
        "model_name": meta.model_name,
        "description": meta.description,
        "n_train": meta.n_train,
        "n_test": meta.n_test,
        "metrics": {k: _finite(v) for k, v in meta.metrics.items()},
        "classes": [str(c) for c in meta.classes] if meta.classes is not None else None,
        "trained_at": meta.trained_at,
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="TabFM Lab API",
        version=__version__,
        description=(
            "Serves TabFM and baseline models for e-commerce conversion, session "
            "page value, football match results and total goals."
        ),
        lifespan=lifespan,
    )

    # The UI is served from the same origin in production; permissive CORS keeps
    # a separately-hosted frontend workable during development.
    origins = os.environ.get("TABFM_CORS_ORIGINS", "*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",")] if origins else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        # Bad feature names and malformed records are client errors, not 500s.
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # -- operational endpoints -------------------------------------------
    @app.get("/api/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness: the process is up. Never touches the models."""
        return {"status": "ok", "version": __version__}

    @app.get("/api/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready(request: Request) -> JSONResponse:
        """Readiness: at least one artifact loaded, so requests can be served."""
        registry = _registry(request)
        payload = {
            "status": "ready" if len(registry) else "no-models",
            "models_loaded": len(registry),
            "models": registry.names(),
            "errors": registry.errors,
        }
        return JSONResponse(status_code=200 if len(registry) else 503, content=payload)

    # -- model discovery --------------------------------------------------
    @app.get("/api/models", response_model=list[ModelSummary], tags=["models"])
    async def list_models(request: Request) -> list[dict[str, Any]]:
        return [_summary(a) for a in _registry(request).all()]

    @app.get("/api/models/{task}", response_model=ModelDetail, tags=["models"])
    async def model_detail(task: str, request: Request) -> dict[str, Any]:
        artifact = _artifact(request, task)
        meta = artifact.metadata
        return {
            **_summary(artifact),
            "feature_columns": meta.feature_columns,
            "categorical_features": meta.categorical_features,
            "fields": [f.to_dict() for f in meta.fields],
            "teams": meta.teams,
            "target_transform": meta.target_transform,
            "versions": meta.versions,
        }

    # -- prediction -------------------------------------------------------
    @app.post("/api/models/{task}/predict", response_model=PredictResponse, tags=["predict"])
    async def predict(task: str, payload: PredictRequest, request: Request) -> dict[str, Any]:
        artifact = _artifact(request, task)
        started = time.perf_counter()

        frame = artifact.frame_from_records(payload.records)
        predictions = artifact.predict(frame)
        probabilities = artifact.predict_proba(frame)
        classes = artifact.metadata.classes or []

        results: list[dict[str, Any]] = []
        for index, value in enumerate(predictions):
            entry: dict[str, Any] = {"prediction": _finite(value)
                                     if artifact.metadata.task_type == "regression"
                                     else str(value)}
            if probabilities is not None and len(classes):
                entry["probabilities"] = {
                    str(label): _finite(probabilities[index][position])
                    for position, label in enumerate(classes)
                }
            if artifact.metadata.target_transform == "log1p":
                entry["prediction_original_units"] = _finite(np.expm1(float(value)))
            results.append(entry)

        return {
            "task": task,
            "model_name": artifact.metadata.model_name,
            "task_type": artifact.metadata.task_type,
            "predictions": results,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    # -- football fixture convenience -------------------------------------
    @app.post("/api/fixtures/predict", response_model=FixtureResponse, tags=["predict"])
    async def predict_fixture(payload: FixtureRequest, request: Request) -> dict[str, Any]:
        """Score a fixture from two team names.

        The server rebuilds the pre-match features from the featuriser stored in
        the artifact, so the caller does not need to know — or be able to forge —
        the 35 engineered inputs.
        """
        registry = _registry(request)
        if RESULT_TASK not in registry and GOALS_TASK not in registry:
            raise HTTPException(status_code=503, detail="No football models are loaded.")

        primary = registry.get(RESULT_TASK if RESULT_TASK in registry else GOALS_TASK)
        featurizer = primary.featurizer
        if featurizer is None:
            raise HTTPException(
                status_code=503,
                detail="Football artifact has no featuriser; retrain to enable fixtures.",
            )

        known = set(featurizer.teams())
        unknown = {payload.home_team, payload.away_team} - known
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown team(s): {sorted(unknown)}. Known teams: {sorted(known)}",
            )

        date = pd.Timestamp(payload.date) if payload.date else featurizer.last_date
        features = featurizer.features_for(payload.home_team, payload.away_team, date)
        reliable = bool(features.pop("is_valid", True))

        notes: list[str] = []
        if not reliable:
            notes.append("One or both teams lack enough match history for reliable features.")

        response: dict[str, Any] = {
            "home_team": payload.home_team,
            "away_team": payload.away_team,
            "date": (date or pd.Timestamp.now()).isoformat(),
            "reliable": reliable,
            "notes": notes,
        }

        result_probabilities: dict[str, float] | None = None
        if RESULT_TASK in registry:
            artifact = registry.get(RESULT_TASK)
            frame = artifact.frame_from_records([_restrict(features, artifact)])
            probabilities = artifact.predict_proba(frame)
            classes = [str(c) for c in (artifact.metadata.classes or [])]
            if probabilities is not None and classes:
                result_probabilities = {
                    label: float(probabilities[0][position])
                    for position, label in enumerate(classes)
                }
                response["result_probabilities"] = {
                    k: _finite(v) for k, v in result_probabilities.items()
                }
                response["predicted_result"] = max(
                    result_probabilities, key=result_probabilities.get
                )

        if GOALS_TASK in registry:
            artifact = registry.get(GOALS_TASK)
            frame = artifact.frame_from_records([_restrict(features, artifact)])
            expected = float(artifact.predict(frame)[0])
            response["expected_total_goals"] = _finite(expected)
            response["over_2_5_probability"] = _finite(
                poisson_over_probability(np.array([expected]), OVER_UNDER_LINE)[0]
            )

        if payload.odds is not None and result_probabilities:
            response.update(_value_bets(result_probabilities, payload.odds))

        return response

    # -- static frontend --------------------------------------------------
    ui = frontend_dir()
    if ui.is_dir():
        # Mounted last so the /api routes above always win the match.
        app.mount("/", StaticFiles(directory=str(ui), html=True), name="frontend")
    else:
        LOGGER.warning("Frontend directory %s not found; serving API only", ui)

    return app


def _restrict(features: dict[str, Any], artifact: ModelArtifact) -> dict[str, Any]:
    """Keep only the columns this artifact was trained on.

    The two football models share a featuriser but may not share a feature list,
    and `frame_from_records` rejects unknown columns rather than ignoring them.
    """
    allowed = set(artifact.metadata.feature_columns)
    record = {k: v for k, v in features.items() if k in allowed}
    # `league` is categorical and comes from the artifact's own default.
    return record


def _value_bets(probabilities: dict[str, float], odds) -> dict[str, Any]:
    """Compare model probabilities against the quoted prices."""
    labels = [label for label in ("H", "D", "A") if label in probabilities]
    prices = np.array([[getattr(odds, _ODDS_FIELDS[label]) for label in labels]], dtype=float)
    market = implied_probabilities(prices)[0]
    raw_total = float(np.sum(1.0 / prices[0]))

    bets = []
    for position, label in enumerate(labels):
        model_probability = float(probabilities[label])
        price = float(prices[0][position])
        edge = model_probability * price - 1.0
        bets.append(
            {
                "outcome": label,
                "model_probability": _finite(model_probability),
                "market_probability": _finite(market[position]),
                "decimal_odds": price,
                "edge": _finite(edge),
                "recommended": bool(edge >= EDGE_THRESHOLD),
            }
        )

    return {"value_bets": bets, "market_overround": _finite(raw_total)}


app = create_app()
