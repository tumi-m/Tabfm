"""Run every model against every task and collect the results.

TabFM is treated as one candidate among several rather than the assumed winner:
if its weights cannot be loaded the run continues on the baselines and records
why it was skipped, so a restricted machine still produces a complete report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .data.base import TabularTask
from .evaluation import betting as betting_mod
from .evaluation.metrics import evaluate
from .models.baselines import build_baselines
from .models.tabfm_backend import TabFMModel, TabFMUnavailableError

LOGGER = logging.getLogger(__name__)

OVER_UNDER_LINE = 2.5


@dataclass
class ModelRun:
    """One model's result on one task."""

    task: str
    model: str
    status: str = "ok"
    metrics: dict[str, float] = field(default_factory=dict)
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    betting: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probabilities(model, X, task: TabularTask) -> tuple[np.ndarray | None, np.ndarray | None]:
    if task.task_type != "classification":
        return None, None
    try:
        probabilities = model.predict_proba(X)
    except (AttributeError, NotImplementedError):
        return None, None
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.unique(task.y_train)
    return np.asarray(probabilities), np.asarray(classes)


def _align_to_classes(
    probabilities: np.ndarray, classes: np.ndarray, wanted: list[str]
) -> np.ndarray | None:
    """Reorder probability columns to ``wanted``, or give up if any is missing."""
    lookup = {str(label): index for index, label in enumerate(classes)}
    if not all(label in lookup for label in wanted):
        return None
    return probabilities[:, [lookup[label] for label in wanted]]


def _result_betting(
    task: TabularTask, probabilities: np.ndarray | None, classes: np.ndarray | None
) -> list[dict[str, Any]]:
    """Value-betting simulation for the 1X2 market."""
    odds = task.extras.get("odds")
    if odds is None or probabilities is None or classes is None:
        return []

    wanted = ["H", "D", "A"]
    aligned = _align_to_classes(probabilities, classes, wanted)
    if aligned is None:
        LOGGER.warning("Model classes %s do not cover H/D/A; skipping betting", classes)
        return []

    odds_matrix = odds[[betting_mod.RESULT_ODDS_COLUMNS[label] for label in wanted]].to_numpy()
    truth = np.asarray(task.y_test)

    return [
        betting_mod.simulate(
            truth, aligned, odds_matrix, np.array(wanted), strategy=strategy
        ).as_dict()
        for strategy in ("flat", "kelly")
    ]


def _total_goals_betting(task: TabularTask, predictions: np.ndarray) -> list[dict[str, Any]]:
    """Convert a goals forecast into over/under 2.5 bets and settle them."""
    odds = task.extras.get("over_under_odds")
    if odds is None:
        return []

    over_probability = betting_mod.poisson_over_probability(predictions, OVER_UNDER_LINE)
    probabilities = np.column_stack([over_probability, 1.0 - over_probability])
    odds_matrix = odds[
        [betting_mod.OVER_UNDER_ODDS_COLUMNS[side] for side in ("over", "under")]
    ].to_numpy()
    truth = np.where(np.asarray(task.y_test) > OVER_UNDER_LINE, "over", "under")

    return [
        betting_mod.simulate(
            truth, probabilities, odds_matrix, np.array(["over", "under"]), strategy=strategy
        ).as_dict()
        for strategy in ("flat", "kelly")
    ]


def run_model(model, task: TabularTask) -> ModelRun:
    """Fit one model on the task's training split and score it on the test split."""
    run = ModelRun(task=task.name, model=model.name)

    started = time.perf_counter()
    try:
        model.fit(task.X_train, task.y_train, task.categorical_features)
    except TabFMUnavailableError as exc:
        run.status = "skipped"
        run.note = str(exc)
        LOGGER.warning("Skipping %s on %s: %s", model.name, task.name, exc)
        return run
    except Exception as exc:  # noqa: BLE001 - one model failing must not end the run
        run.status = "failed"
        run.note = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Model %s failed on %s", model.name, task.name)
        return run
    run.fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    try:
        predictions = model.predict(task.X_test)
        probabilities, classes = _probabilities(model, task.X_test, task)
    except Exception as exc:  # noqa: BLE001
        run.status = "failed"
        run.note = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Prediction failed for %s on %s", model.name, task.name)
        return run
    run.predict_seconds = time.perf_counter() - started

    run.metrics = evaluate(task, predictions, probabilities, classes)

    if task.industry == "sports-betting":
        if task.task_type == "classification":
            run.betting = _result_betting(task, probabilities, classes)
        else:
            run.betting = _total_goals_betting(task, np.asarray(predictions, dtype=float))

    return run


def run_task(
    task: TabularTask,
    *,
    include_tabfm: bool = True,
    include_baselines: bool = True,
    tabfm_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
) -> list[ModelRun]:
    LOGGER.info("Task %s", task.name)
    models: list[Any] = []
    if include_tabfm:
        models.append(TabFMModel(task.task_type, **(tabfm_kwargs or {})))
    if include_baselines:
        models.extend(build_baselines(task, seed=seed))

    return [run_model(model, task) for model in models]


def market_benchmark(task: TabularTask) -> dict[str, Any]:
    """Score the bookmaker's prices on the same test rows, where available."""
    odds = task.extras.get("odds")
    if odds is None or task.task_type != "classification":
        return {}
    wanted = ["H", "D", "A"]
    odds_matrix = odds[[betting_mod.RESULT_ODDS_COLUMNS[label] for label in wanted]].to_numpy()
    return betting_mod.market_reference(
        np.asarray(task.y_test), odds_matrix, np.array(wanted)
    )


def run_benchmark(
    tasks: list[TabularTask],
    *,
    include_tabfm: bool = True,
    include_baselines: bool = True,
    tabfm_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the full benchmark and return a serialisable report."""
    report: dict[str, Any] = {"tasks": [], "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    for task in tasks:
        runs = run_task(
            task,
            include_tabfm=include_tabfm,
            include_baselines=include_baselines,
            tabfm_kwargs=tabfm_kwargs,
            seed=seed,
        )
        entry: dict[str, Any] = {
            "name": task.name,
            "industry": task.industry,
            "task_type": task.task_type,
            "target": task.target,
            "description": task.description,
            "n_train": len(task.X_train),
            "n_test": len(task.X_test),
            "n_features": task.n_features,
            "runs": [run.to_dict() for run in runs],
        }
        benchmark = market_benchmark(task)
        if benchmark:
            entry["market_benchmark"] = benchmark
        report["tasks"].append(entry)

    return report
