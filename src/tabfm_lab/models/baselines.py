"""Reference models TabFM has to beat.

The comparison TabFM's zero-shot claim invites is against tuned gradient
boosting, so the ensemble baseline is the one that matters. The linear and
constant models are kept because they set the floor: on imbalanced or
low-signal tasks a strong-looking score sometimes turns out to be barely above
predicting the prior, and that is worth seeing explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from ..data.base import TabularTask
from .preprocess import as_sklearn_frame, linear_transformer, ordinal_transformer


class SklearnModel:
    """Wrap an sklearn estimator in the same surface as :class:`TabFMModel`."""

    def __init__(self, name: str, estimator, encoding: str = "ordinal") -> None:
        self.name = name
        self._estimator = estimator
        self._encoding = encoding
        self._pipeline: Pipeline | None = None
        self._categorical: list[str] = []
        self.classes_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, categorical: list[str] | None = None):
        self._categorical = list(categorical or [])
        frame = as_sklearn_frame(X, self._categorical)

        make = linear_transformer if self._encoding == "linear" else ordinal_transformer
        self._pipeline = Pipeline(
            [("prep", make(frame, self._categorical)), ("model", self._estimator)]
        )
        self._pipeline.fit(frame, np.asarray(y))
        model = self._pipeline.named_steps["model"]
        self.classes_ = getattr(model, "classes_", None)
        return self

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._pipeline is None:
            raise RuntimeError("fit() must be called before predicting")
        return as_sklearn_frame(X, self._categorical)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._pipeline.predict(self._frame(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        model = self._pipeline.named_steps["model"]
        if not hasattr(model, "predict_proba"):
            raise AttributeError(f"{self.name} does not expose predict_proba")
        return np.asarray(self._pipeline.predict_proba(self._frame(X)))


def build_baselines(task: TabularTask, *, seed: int = 42) -> list[SklearnModel]:
    """Baseline suite appropriate to ``task``'s objective."""
    if task.task_type == "classification":
        return [
            SklearnModel(
                "Prior (majority class)",
                DummyClassifier(strategy="prior"),
            ),
            SklearnModel(
                "Logistic Regression",
                LogisticRegression(max_iter=2000, random_state=seed),
                encoding="linear",
            ),
            SklearnModel(
                "HistGradientBoosting",
                HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.06,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=seed,
                ),
            ),
        ]

    return [
        SklearnModel("Mean predictor", DummyRegressor(strategy="mean")),
        SklearnModel("Ridge Regression", Ridge(alpha=1.0, random_state=seed), encoding="linear"),
        SklearnModel(
            "HistGradientBoosting",
            HistGradientBoostingRegressor(
                max_iter=400,
                learning_rate=0.06,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=seed,
            ),
        ),
    ]
