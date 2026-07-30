"""Tests for scoring, the task container and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabfm_lab.data.base import TabularTask
from tabfm_lab.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    multiclass_brier,
    regression_metrics,
)
from tabfm_lab.models.preprocess import as_sklearn_frame, ordinal_transformer

CLASSES = np.array(["A", "D", "H"])


def test_brier_rewards_confident_correct_predictions() -> None:
    truth = np.array(["H", "H"])
    perfect = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    hedged = np.array([[1 / 3, 1 / 3, 1 / 3]] * 2)

    assert multiclass_brier(truth, perfect, CLASSES) == pytest.approx(0.0)
    assert multiclass_brier(truth, hedged, CLASSES) > 0.0


def test_calibration_error_is_zero_for_a_calibrated_model() -> None:
    """A model that says 100% and is always right has no calibration error."""
    truth = np.array(["H"] * 50)
    probabilities = np.tile([0.0, 0.0, 1.0], (50, 1))
    assert expected_calibration_error(truth, probabilities, CLASSES) == pytest.approx(0.0)


def test_calibration_error_penalises_overconfidence() -> None:
    """Claiming 100% while being right half the time should score about 0.5."""
    truth = np.array(["H"] * 25 + ["A"] * 25)
    probabilities = np.tile([0.0, 0.0, 1.0], (50, 1))
    assert expected_calibration_error(truth, probabilities, CLASSES) == pytest.approx(0.5, abs=1e-6)


def test_classification_metrics_cover_probabilistic_scores() -> None:
    truth = np.array(["H", "A", "H", "D"])
    predicted = np.array(["H", "A", "D", "D"])
    probabilities = np.array(
        [[0.1, 0.2, 0.7], [0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.2, 0.6, 0.2]]
    )

    scores = classification_metrics(truth, predicted, probabilities, CLASSES)
    assert scores["accuracy"] == pytest.approx(0.75)
    assert "log_loss" in scores
    assert "roc_auc_ovr" in scores
    assert "calibration_error" in scores


def test_metrics_are_independent_of_class_order() -> None:
    """Passing columns as H/D/A must score the same as A/D/H.

    scikit-learn's log_loss ignores the ordering of its `labels` argument and
    assumes lexicographic columns, so an unsorted caller would otherwise be
    scored against the wrong outcome entirely.
    """
    truth = np.array(["H", "A", "D", "H"])
    lexicographic = np.array(["A", "D", "H"])
    probabilities = np.array(
        [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.3, 0.2, 0.5]]
    )

    reordered = np.array(["H", "D", "A"])
    permutation = [2, 1, 0]

    sorted_scores = classification_metrics(
        truth, np.array(["H", "A", "D", "H"]), probabilities, lexicographic
    )
    shuffled_scores = classification_metrics(
        truth, np.array(["H", "A", "D", "H"]), probabilities[:, permutation], reordered
    )

    assert sorted_scores["log_loss"] == pytest.approx(shuffled_scores["log_loss"])
    assert sorted_scores["brier"] == pytest.approx(shuffled_scores["brier"])
    assert sorted_scores["roc_auc_ovr"] == pytest.approx(shuffled_scores["roc_auc_ovr"])


def test_binary_metrics_include_pr_auc() -> None:
    truth = np.array([0, 1, 0, 1])
    predicted = np.array([0, 1, 0, 0])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4]])

    scores = classification_metrics(truth, predicted, probabilities, np.array([0, 1]))
    assert "pr_auc" in scores
    assert "roc_auc" in scores


def _regression_task(transform: str | None) -> TabularTask:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    target = pd.Series([0.0, 1.0, 2.0, 3.0])
    return TabularTask(
        name="t",
        industry="test",
        task_type="regression",
        target="y",
        X_train=frame,
        y_train=target,
        X_test=frame,
        y_test=target,
        target_transform=transform,
    )


def test_regression_metrics_report_original_units_when_transformed() -> None:
    task = _regression_task("log1p")
    truth = np.array([0.0, 1.0, 2.0])
    predictions = np.array([0.0, 1.0, 2.0])

    scores = regression_metrics(truth, predictions, task)
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["rmse_original_units"] == pytest.approx(0.0)


def test_inverse_transform_round_trips_log1p() -> None:
    task = _regression_task("log1p")
    original = np.array([0.0, 5.0, 250.0])
    assert np.allclose(task.inverse_transform_target(np.log1p(original)), original)


def test_inverse_transform_is_identity_without_a_transform() -> None:
    task = _regression_task(None)
    values = np.array([1.0, 2.0])
    assert np.allclose(task.inverse_transform_target(values), values)


def test_task_rejects_mismatched_columns() -> None:
    with pytest.raises(ValueError, match="feature columns differ"):
        TabularTask(
            name="t",
            industry="test",
            task_type="regression",
            target="y",
            X_train=pd.DataFrame({"a": [1.0]}),
            y_train=pd.Series([1.0]),
            X_test=pd.DataFrame({"b": [1.0]}),
            y_test=pd.Series([1.0]),
        )


def test_task_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        TabularTask(
            name="t",
            industry="test",
            task_type="regression",
            target="y",
            X_train=pd.DataFrame({"a": [1.0, 2.0]}),
            y_train=pd.Series([1.0]),
            X_test=pd.DataFrame({"a": [1.0]}),
            y_test=pd.Series([1.0]),
        )


def test_ordinal_transformer_handles_unseen_categories_and_nans() -> None:
    train = pd.DataFrame({"num": [1.0, 2.0, np.nan], "cat": ["a", "b", "a"]})
    test = pd.DataFrame({"num": [np.nan, 3.0], "cat": ["a", "totally-new"]})

    transformer = ordinal_transformer(train, ["cat"])
    transformer.fit(as_sklearn_frame(train, ["cat"]))
    encoded = transformer.transform(as_sklearn_frame(test, ["cat"]))

    assert np.isfinite(np.asarray(encoded, dtype=float)).all(), "no NaNs may survive"
    # Unseen levels are encoded as -1 rather than raising.
    assert -1 in np.asarray(encoded, dtype=float)
