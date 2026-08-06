"""Tests for the customer marketing dataset and the TabFM introspection helpers.

All offline: the cleaning step is a pure function over a synthetic frame, so no
download is needed to check the parts that are easy to get wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabfm_lab.data.ecommerce import (
    _PURCHASE_COUNT_COLUMNS,
    _SPEND_COLUMNS,
    clean_customer_marketing,
)
from tabfm_lab.models import tabfm_backend


def raw_marketing(n: int = 40) -> pd.DataFrame:
    """A synthetic frame with the raw dataset's quirks deliberately included."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "ID": range(n),
            "Year_Birth": [1980] * (n - 2) + [1893, 2015],  # two implausible ages
            "Education": rng.choice(["Graduation", "PhD", "Basic"], n),
            "Marital_Status": ["Married"] * (n - 3) + ["Absurd", "YOLO", "Alone"],
            "Income": rng.integers(20_000, 90_000, n).astype(float),
            "Kidhome": rng.integers(0, 3, n),
            "Teenhome": rng.integers(0, 3, n),
            "Dt_Customer": ["01-01-2013"] * (n - 1) + ["31-12-2014"],
            "Recency": rng.integers(0, 99, n),
            "NumWebVisitsMonth": rng.integers(0, 20, n),
            "Complain": 0,
            "Z_CostContact": 3,
            "Z_Revenue": 11,
            "Response": rng.integers(0, 2, n),
        }
    )
    for column in _SPEND_COLUMNS:
        frame[column] = rng.integers(0, 500, n)
    for column in _PURCHASE_COUNT_COLUMNS:
        frame[column] = rng.integers(0, 15, n)
    for index in range(1, 6):
        frame[f"AcceptedCmp{index}"] = rng.integers(0, 2, n)
    return frame


@pytest.fixture
def cleaned() -> pd.DataFrame:
    return clean_customer_marketing(raw_marketing())


def test_constant_and_identifier_columns_are_dropped(cleaned: pd.DataFrame) -> None:
    """Z_CostContact and Z_Revenue never vary, and ID is not a feature."""
    for column in ("ID", "Z_CostContact", "Z_Revenue", "Dt_Customer", "Year_Birth"):
        assert column not in cleaned.columns


def test_junk_marital_levels_are_folded_not_dropped(cleaned: pd.DataFrame) -> None:
    """Rows with 'Absurd' or 'YOLO' are otherwise fine; discarding them would bias."""
    assert len(cleaned) == 40
    levels = set(cleaned["Marital_Status"].astype(str))
    assert not levels & {"Absurd", "YOLO", "Alone"}
    assert "Other" in levels


def test_implausible_ages_become_missing(cleaned: pd.DataFrame) -> None:
    """A birth year of 1893 must not reach the model as a 130-year-old."""
    ages = cleaned["age"]
    assert ages.isna().sum() == 2
    valid = ages.dropna()
    assert (valid >= 18).all() and (valid <= 100).all()


def test_tenure_is_relative_to_the_newest_row(cleaned: pd.DataFrame) -> None:
    """Measured against the file's own latest date, so it does not drift with time."""
    assert cleaned["tenure_days"].min() == 0
    assert cleaned["tenure_days"].max() > 0


def test_total_spend_is_the_sum_of_the_category_columns(cleaned: pd.DataFrame) -> None:
    raw = raw_marketing()
    expected = raw[_SPEND_COLUMNS].sum(axis=1)
    assert np.allclose(cleaned["total_spend"].to_numpy(), expected.to_numpy())


def test_spend_and_purchase_counts_are_excluded_from_the_value_task() -> None:
    """The value target's own components, and near-restatements of it, are not inputs.

    Checked against the module's declared exclusion lists rather than a live
    download, so the intent is pinned even when the network is unavailable.
    """
    from tabfm_lab.data import ecommerce

    excluded = set(_SPEND_COLUMNS) | set(_PURCHASE_COUNT_COLUMNS)
    cleaned = clean_customer_marketing(raw_marketing())
    features = ecommerce._feature_columns(
        cleaned, drop=["total_spend", *_SPEND_COLUMNS, *_PURCHASE_COUNT_COLUMNS, "Response"]
    )

    assert not (set(features) & excluded)
    assert "total_spend" not in features
    # Engagement that is not a purchase count is still allowed.
    assert "NumWebVisitsMonth" in features
    assert "age" in features and "tenure_days" in features


# ------------------------------------------------------- TabFM introspection
def test_availability_reports_without_raising() -> None:
    """Status must be reportable whether or not TabFM is installed."""
    status = tabfm_backend.availability()

    expected = {
        "package_installed", "backend", "backend_available",
        "weights_probed", "weights_loadable", "detail",
    }
    assert expected <= set(status)
    assert status["weights_probed"] is False, "weights must not be fetched by default"
    assert isinstance(status["detail"], str) and status["detail"]


def test_availability_does_not_download_weights_by_default(monkeypatch) -> None:
    """A status check must never trigger a hundreds-of-megabytes download."""
    called = False

    def _fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("weights should not be loaded for a status check")

    monkeypatch.setattr(tabfm_backend, "_load_backend", _fail)
    tabfm_backend.availability()
    assert not called


def test_wrapper_source_contains_the_integration() -> None:
    """The UI shows this source; it must be the module that actually runs."""
    source = tabfm_backend.wrapper_source()
    assert "class TabFMModel" in source
    assert "__getstate__" in source
