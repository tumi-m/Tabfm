"""End-to-end cover for the betting path in the pipeline.

The public mirror of football-data.co.uk ships match statistics without odds, so
on many machines a real run never exercises the betting branch. These tests
inject synthetic odds so the wiring — column alignment, market benchmark,
over/under conversion — is still covered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabfm_lab.data.base import TabularTask
from tabfm_lab.pipeline import market_benchmark, run_model

RESULTS = ["H", "D", "A"]


class StubClassifier:
    """Returns a fixed probability per row, in a deliberately non-sorted order."""

    def __init__(self, probabilities: np.ndarray, classes: list[str]) -> None:
        self.name = "Stub"
        self._probabilities = probabilities
        self.classes_ = np.array(classes)

    def fit(self, X, y, categorical=None):
        return self

    def predict(self, X):
        return self.classes_[self._probabilities.argmax(axis=1)]

    def predict_proba(self, X):
        return self._probabilities


class StubRegressor:
    def __init__(self, predictions: np.ndarray) -> None:
        self.name = "StubReg"
        self._predictions = predictions

    def fit(self, X, y, categorical=None):
        return self

    def predict(self, X):
        return self._predictions


def _features(n: int) -> pd.DataFrame:
    return pd.DataFrame({"elo_diff": np.linspace(-100, 100, n)})


def _result_task(probabilities_classes: list[str]) -> tuple[TabularTask, np.ndarray]:
    n = 6
    truth = pd.Series(["H", "H", "D", "A", "H", "A"])
    odds = pd.DataFrame(
        {
            "B365H": [1.8] * n,
            "B365D": [3.6] * n,
            "B365A": [4.5] * n,
        }
    )
    # Strong, correct-leaning confidence on the home win so bets get placed.
    base = {"H": 0.70, "D": 0.18, "A": 0.12}
    probabilities = np.tile([base[c] for c in probabilities_classes], (n, 1))

    task = TabularTask(
        name="sports-test",
        industry="sports-betting",
        task_type="classification",
        target="FTR",
        X_train=_features(n),
        y_train=truth,
        X_test=_features(n),
        y_test=truth,
        extras={"odds": odds},
    )
    return task, probabilities


def test_betting_runs_and_reports_both_strategies() -> None:
    task, probabilities = _result_task(RESULTS)
    run = run_model(StubClassifier(probabilities, RESULTS), task)

    assert run.status == "ok"
    assert len(run.betting) == 2
    strategies = {entry["strategy"].split()[0] for entry in run.betting}
    assert strategies == {"flat", "kelly"}

    flat = next(e for e in run.betting if e["strategy"].startswith("flat"))
    # Edge on H is 0.70 * 1.8 - 1 = 0.26, so every match is backed.
    assert flat["n_bets"] == 6
    assert flat["hit_rate"] == pytest.approx(3 / 6)


def test_betting_is_invariant_to_model_class_order() -> None:
    """A model listing classes A/D/H must bet identically to one listing H/D/A."""
    task, probabilities = _result_task(RESULTS)
    natural = run_model(StubClassifier(probabilities, RESULTS), task)

    shuffled_classes = ["A", "D", "H"]
    _, shuffled_probabilities = _result_task(shuffled_classes)
    shuffled = run_model(StubClassifier(shuffled_probabilities, shuffled_classes), task)

    assert natural.betting[0]["profit"] == pytest.approx(shuffled.betting[0]["profit"])
    assert natural.metrics["log_loss"] == pytest.approx(shuffled.metrics["log_loss"])


def test_betting_skipped_when_odds_absent() -> None:
    task, probabilities = _result_task(RESULTS)
    task.extras.pop("odds")

    run = run_model(StubClassifier(probabilities, RESULTS), task)
    assert run.status == "ok"
    assert run.betting == []
    assert market_benchmark(task) == {}


def test_market_benchmark_scores_the_bookmaker() -> None:
    task, _ = _result_task(RESULTS)
    benchmark = market_benchmark(task)

    assert benchmark["n_matches"] == 6
    assert benchmark["mean_overround"] > 1.0
    assert benchmark["log_loss"] > 0.0


def test_total_goals_converts_to_over_under_bets() -> None:
    n = 6
    truth = pd.Series([3.0, 1.0, 4.0, 0.0, 2.0, 5.0])
    odds = pd.DataFrame({"B365>2.5": [1.9] * n, "B365<2.5": [1.95] * n})

    task = TabularTask(
        name="sports-goals-test",
        industry="sports-betting",
        task_type="regression",
        target="total_goals",
        X_train=_features(n),
        y_train=truth,
        X_test=_features(n),
        y_test=truth,
        extras={"over_under_odds": odds},
    )

    # A high goals forecast pushes P(over 2.5) well above the market's ~0.51.
    run = run_model(StubRegressor(np.full(n, 4.2)), task)

    assert run.status == "ok"
    assert len(run.betting) == 2
    flat = next(e for e in run.betting if e["strategy"].startswith("flat"))
    assert flat["n_bets"] > 0
    # Three of the six matches went over 2.5.
    assert flat["hit_rate"] == pytest.approx(3 / 6)


def test_failing_model_is_recorded_not_raised() -> None:
    """One broken model must not abort the whole benchmark."""

    class Broken:
        name = "Broken"

        def fit(self, X, y, categorical=None):
            raise RuntimeError("boom")

    task, _ = _result_task(RESULTS)
    run = run_model(Broken(), task)

    assert run.status == "failed"
    assert "boom" in run.note
