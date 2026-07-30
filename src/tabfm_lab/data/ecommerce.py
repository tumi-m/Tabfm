"""E-commerce tasks built on the UCI Online Shoppers Purchasing Intention data.

The dataset holds 12,330 browsing sessions from an online retailer, each from a
distinct user across one year, with behavioural counters, Google Analytics page
metrics and traffic attributes.

Two tasks are derived from it:

``ecommerce-conversion`` (classification)
    Did the session end in a purchase? This is the headline funnel question and
    is heavily imbalanced (~15.5% positive), which is what makes it a fair test
    of calibrated probabilities rather than raw accuracy.

``ecommerce-page-value`` (regression)
    How much revenue-weighted page value did the session accumulate? Page Value
    is the Analytics revenue attribution for the pages in a session, so it acts
    as a per-session revenue proxy. It is strongly zero-inflated, so it is
    modelled on the ``log1p`` scale and reported back in the original units.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .base import TabularTask, stratified_split
from .sources import ONLINE_SHOPPERS, fetch

#: Stored as integers but semantically unordered — declared categorical so
#: models never read a distance between, say, Browser 1 and Browser 8.
_CODED_CATEGORICALS = ["OperatingSystems", "Browser", "Region", "TrafficType"]
_TEXT_CATEGORICALS = ["Month", "VisitorType"]
_BOOLEANS = ["Weekend", "Revenue"]


def _read_any(path: Path) -> pd.DataFrame:
    """Read the dataset whether it arrived as a bare CSV or a UCI zip."""
    payload = path.read_bytes()
    if payload[:2] == b"PK":  # zip magic number
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV inside archive {path}")
            with archive.open(names[0]) as handle:
                return pd.read_csv(handle)
    return pd.read_csv(path)


def load_online_shoppers(cache_dir: Path | None = None) -> pd.DataFrame:
    """Download (once) and normalise the raw session table."""
    frame = _read_any(fetch(ONLINE_SHOPPERS, cache_dir))

    missing = set(ONLINE_SHOPPERS.required_columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Online Shoppers data is missing expected columns: {sorted(missing)}"
        )

    for column in _BOOLEANS:
        if frame[column].dtype != bool:
            frame[column] = frame[column].astype(str).str.upper().eq("TRUE")

    for column in _CODED_CATEGORICALS + _TEXT_CATEGORICALS:
        frame[column] = frame[column].astype("category")

    return frame


def _feature_columns(frame: pd.DataFrame, drop: list[str]) -> list[str]:
    return [c for c in frame.columns if c not in drop]


def conversion_task(
    cache_dir: Path | None = None,
    *,
    test_size: float = 0.25,
    seed: int = 42,
) -> TabularTask:
    """Predict whether a browsing session converts into a purchase."""
    frame = load_online_shoppers(cache_dir)
    target = "Revenue"

    features = _feature_columns(frame, drop=[target])
    train, test = stratified_split(frame, target, test_size=test_size, seed=seed)

    categoricals = [c for c in _CODED_CATEGORICALS + _TEXT_CATEGORICALS if c in features]

    return TabularTask(
        name="ecommerce-conversion",
        industry="ecommerce",
        task_type="classification",
        target=target,
        X_train=train[features].reset_index(drop=True),
        y_train=train[target].astype(int).reset_index(drop=True),
        X_test=test[features].reset_index(drop=True),
        y_test=test[target].astype(int).reset_index(drop=True),
        categorical_features=categoricals,
        description=(
            "Session-level purchase conversion for an online retailer "
            "(UCI Online Shoppers Purchasing Intention). PageValues is retained "
            "as a feature: it is an Analytics metric available for the session "
            "being scored, and it is the dominant signal in this dataset."
        ),
    )


def page_value_task(
    cache_dir: Path | None = None,
    *,
    test_size: float = 0.25,
    seed: int = 42,
) -> TabularTask:
    """Predict the revenue-weighted page value accumulated by a session."""
    frame = load_online_shoppers(cache_dir)
    target = "PageValues"

    # Revenue is the downstream outcome of the same session. Keeping it as a
    # feature would leak the answer into a revenue-proxy prediction, so it goes.
    features = _feature_columns(frame, drop=[target, "Revenue"])

    # Stratify on the conversion flag so the rare high-value sessions are
    # represented in both splits at the same rate.
    train, test = stratified_split(frame, "Revenue", test_size=test_size, seed=seed)

    categoricals = [c for c in _CODED_CATEGORICALS + _TEXT_CATEGORICALS if c in features]

    return TabularTask(
        name="ecommerce-page-value",
        industry="ecommerce",
        task_type="regression",
        target=target,
        X_train=train[features].reset_index(drop=True),
        y_train=pd.Series(
            np.log1p(train[target].to_numpy(dtype=float)), name=target
        ).reset_index(drop=True),
        X_test=test[features].reset_index(drop=True),
        y_test=pd.Series(
            np.log1p(test[target].to_numpy(dtype=float)), name=target
        ).reset_index(drop=True),
        categorical_features=categoricals,
        target_transform="log1p",
        description=(
            "Per-session Google Analytics Page Value as a revenue proxy "
            "(UCI Online Shoppers Purchasing Intention). Zero-inflated, so it is "
            "modelled on the log1p scale; metrics are reported in both spaces."
        ),
    )


def build_tasks(cache_dir: Path | None = None, **kwargs) -> list[TabularTask]:
    return [conversion_task(cache_dir, **kwargs), page_value_task(cache_dir, **kwargs)]
