"""Shared containers describing a prepared supervised task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

TaskType = Literal["classification", "regression"]


@dataclass
class TabularTask:
    """A train/test split ready to hand to any estimator in this project.

    ``extras`` carries task-specific side data that must stay row-aligned with
    the test split — most importantly the bookmaker odds used by the betting
    evaluation, which are deliberately kept out of ``X`` so they can never leak
    into a model as a feature.
    """

    name: str
    industry: str
    task_type: TaskType
    target: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    categorical_features: list[str] = field(default_factory=list)
    description: str = ""
    # Name of a monotonic transform applied to the target, inverted at
    # reporting time so metrics are quoted in the original units.
    target_transform: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X_train and y_train have different lengths")
        if len(self.X_test) != len(self.y_test):
            raise ValueError("X_test and y_test have different lengths")
        if list(self.X_train.columns) != list(self.X_test.columns):
            raise ValueError("Train and test feature columns differ")

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int | None:
        if self.task_type != "classification":
            return None
        return int(pd.concat([self.y_train, self.y_test]).nunique())

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """Map predictions back to the target's original units."""
        if self.target_transform is None:
            return values
        if self.target_transform == "log1p":
            # expm1 is the exact inverse of log1p; clip first so a wildly
            # negative prediction cannot overflow the exponential.
            return np.expm1(np.clip(values, -30, 30))
        raise ValueError(f"Unknown target transform: {self.target_transform}")

    def summary(self) -> str:
        bits = [
            f"{self.name} [{self.industry} / {self.task_type}]",
            f"  target        : {self.target}"
            + (f" (modelled as {self.target_transform})" if self.target_transform else ""),
            f"  train / test  : {len(self.X_train):,} / {len(self.X_test):,} rows",
            f"  features      : {self.n_features} "
            f"({len(self.categorical_features)} categorical)",
        ]
        if self.task_type == "classification":
            counts = self.y_train.value_counts(normalize=True).sort_index()
            dist = ", ".join(f"{k}={v:.1%}" for k, v in counts.items())
            bits.append(f"  train balance : {dist}")
        return "\n".join(bits)


def stratified_split(
    frame: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Class-balanced split, used where rows are exchangeable."""
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        frame.index,
        test_size=test_size,
        random_state=seed,
        stratify=frame[target],
    )
    return frame.loc[train_idx], frame.loc[test_idx]
