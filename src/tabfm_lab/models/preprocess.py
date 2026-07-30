"""Feature preparation shared by TabFM and the baseline estimators.

Every transformer here is fitted on the training split only and then applied to
the test split, so no test-set statistic ever informs the encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def as_sklearn_frame(frame: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Cast pandas ``category`` columns to plain strings.

    The categorical dtype carries a fixed set of levels; a level present only in
    the test split would otherwise surface as a NaN that is hard to trace.
    Strings keep the values legible and let the encoder's explicit
    unknown-handling deal with unseen levels.
    """
    out = frame.copy()
    for column in categorical:
        if column in out.columns:
            out[column] = out[column].astype(str)
    return out


def numeric_columns(frame: pd.DataFrame, categorical: list[str]) -> list[str]:
    return [c for c in frame.columns if c not in categorical]


def ordinal_transformer(frame: pd.DataFrame, categorical: list[str]) -> ColumnTransformer:
    """Dense numeric matrix: median-imputed numerics, ordinal-coded categories.

    Suits tree ensembles and TabFM, neither of which needs one-hot expansion.
    """
    numeric = numeric_columns(frame, categorical)
    categorical = [c for c in categorical if c in frame.columns]

    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def linear_transformer(frame: pd.DataFrame, categorical: list[str]) -> ColumnTransformer:
    """Standardised numerics plus one-hot categories, for linear models."""
    numeric = numeric_columns(frame, categorical)
    categorical = [c for c in categorical if c in frame.columns]

    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=10),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def to_dense_float(matrix) -> np.ndarray:
    """Materialise a transformer's output as a float array."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)
