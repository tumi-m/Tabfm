"""Scoring for both objectives.

Probability quality is reported alongside accuracy throughout. For the betting
tasks that is not a stylistic preference: a model that is 2% more accurate but
badly calibrated loses money against a bookmaker, while a well-calibrated model
can be profitable without ever being the most accurate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from ..data.base import TabularTask


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, n_bins: int = 10
) -> float:
    """Bin predictions by confidence and average |accuracy - confidence|.

    Answers "when this model says 70%, does it happen 70% of the time?" — the
    property that decides whether a probability can be staked on.
    """
    confidence = probabilities.max(axis=1)
    predicted = classes[probabilities.argmax(axis=1)]
    correct = (predicted == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    """Mean squared error between the probability vector and the one-hot truth."""
    onehot = np.zeros_like(probabilities, dtype=float)
    index = {label: position for position, label in enumerate(classes)}
    for row, label in enumerate(y_true):
        onehot[row, index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1)))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    classes: np.ndarray | None,
) -> dict[str, float]:
    scores: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if probabilities is None or classes is None:
        return scores

    # scikit-learn's log_loss assumes the probability columns are in
    # lexicographic label order regardless of the `labels` argument, so a caller
    # passing H/D/A columns would silently be scored against A/D/H. Sort the
    # classes and permute the columns to match, which makes every metric below
    # independent of the order the caller happened to use.
    classes = np.asarray(classes)
    order = np.argsort(classes)
    classes = classes[order]
    probabilities = np.asarray(probabilities)[:, order]

    binary = len(classes) == 2
    try:
        scores["log_loss"] = float(log_loss(y_true, probabilities, labels=list(classes)))
    except ValueError:
        pass

    try:
        if binary:
            positive = probabilities[:, 1]
            scores["roc_auc"] = float(roc_auc_score(y_true, positive))
            scores["pr_auc"] = float(
                average_precision_score((y_true == classes[1]).astype(int), positive)
            )
        else:
            scores["roc_auc_ovr"] = float(
                roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
            )
    except ValueError:
        pass

    scores["brier"] = multiclass_brier(np.asarray(y_true), probabilities, classes)
    scores["calibration_error"] = expected_calibration_error(
        np.asarray(y_true), probabilities, classes
    )
    return scores


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, task: TabularTask | None = None
) -> dict[str, float]:
    """Errors in model space, plus original units when the target was transformed."""
    scores = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    if task is not None and task.target_transform is not None:
        original_true = task.inverse_transform_target(np.asarray(y_true, dtype=float))
        original_pred = task.inverse_transform_target(np.asarray(y_pred, dtype=float))
        scores["rmse_original_units"] = float(
            np.sqrt(mean_squared_error(original_true, original_pred))
        )
        scores["mae_original_units"] = float(
            mean_absolute_error(original_true, original_pred)
        )
    return scores


def evaluate(
    task: TabularTask,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(task.y_test)
    if task.task_type == "classification":
        return classification_metrics(y_true, np.asarray(y_pred), probabilities, classes)
    return regression_metrics(y_true, np.asarray(y_pred, dtype=float), task)


#: Metric to rank models by, and whether larger is better.
PRIMARY_METRIC = {
    "classification": ("log_loss", False),
    "regression": ("rmse", False),
}
