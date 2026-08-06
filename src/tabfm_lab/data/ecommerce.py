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
from .sources import CUSTOMER_MARKETING, ONLINE_SHOPPERS, fetch

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


# --------------------------------------------------------------------------
# Customer marketing: campaign response and customer value
# --------------------------------------------------------------------------
#: Per-category spend over two years. These sum to the customer-value target.
_SPEND_COLUMNS = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]

#: Counts of purchases by channel. Legitimate for predicting campaign response,
#: but not for predicting spend — more purchases means more spend almost by
#: definition, so a model using them would score well while learning nothing.
_PURCHASE_COUNT_COLUMNS = [
    "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases",
]

#: Constant in every row; they carry no signal and confuse feature counts.
_CONSTANT_COLUMNS = ["Z_CostContact", "Z_Revenue"]

#: Marital_Status contains a few junk levels ("Absurd", "YOLO"). Anything
#: outside this set is folded into "Other" rather than dropped, because the rows
#: are otherwise fine and discarding them would bias the sample.
_KNOWN_MARITAL = {"Single", "Together", "Married", "Divorced", "Widow"}

_MARKETING_CATEGORICALS = ["Education", "Marital_Status"]


def clean_customer_marketing(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw marketing table. Pure, so it can be tested without a download."""
    missing = set(CUSTOMER_MARKETING.required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Customer marketing data is missing columns: {sorted(missing)}")

    frame = frame.drop(columns=[c for c in ["ID", *_CONSTANT_COLUMNS] if c in frame.columns])

    # Enrolment date -> tenure, measured against the newest date in the file
    # rather than today, so the feature does not drift as the data ages.
    enrolled = pd.to_datetime(frame["Dt_Customer"], format="mixed", dayfirst=True, errors="coerce")
    frame["tenure_days"] = (enrolled.max() - enrolled).dt.days
    frame = frame.drop(columns=["Dt_Customer"])

    # A handful of birth years are implausible (1893 and similar). Blank them so
    # the imputer handles them, rather than feeding a 130-year-old customer in.
    age = enrolled.max().year - frame["Year_Birth"]
    frame["age"] = age.where((age >= 18) & (age <= 100))
    frame = frame.drop(columns=["Year_Birth"])

    frame["Marital_Status"] = frame["Marital_Status"].where(
        frame["Marital_Status"].isin(_KNOWN_MARITAL), "Other"
    )
    for column in _MARKETING_CATEGORICALS:
        frame[column] = frame[column].astype("category")

    frame["total_spend"] = frame[_SPEND_COLUMNS].sum(axis=1)
    return frame


def load_customer_marketing(cache_dir: Path | None = None) -> pd.DataFrame:
    """Download (once) and clean the customer marketing table."""
    return clean_customer_marketing(pd.read_csv(fetch(CUSTOMER_MARKETING, cache_dir)))


def campaign_response_task(
    cache_dir: Path | None = None,
    *,
    test_size: float = 0.25,
    seed: int = 42,
) -> TabularTask:
    """Predict whether a customer accepts the latest marketing campaign."""
    frame = load_customer_marketing(cache_dir)
    target = "Response"

    # total_spend is a derived convenience column, not an input.
    features = _feature_columns(frame, drop=[target, "total_spend"])
    train, test = stratified_split(frame, target, test_size=test_size, seed=seed)
    categoricals = [c for c in _MARKETING_CATEGORICALS if c in features]

    return TabularTask(
        name="ecommerce-campaign-response",
        industry="ecommerce",
        task_type="classification",
        target=target,
        X_train=train[features].reset_index(drop=True),
        y_train=train[target].astype(int).reset_index(drop=True),
        X_test=test[features].reset_index(drop=True),
        y_test=test[target].astype(int).reset_index(drop=True),
        categorical_features=categoricals,
        description=(
            "Direct-marketing response for a retailer's customer base. Prior "
            "campaign outcomes and two years of category spend are legitimate "
            "inputs — they precede the campaign being predicted. Imbalanced at "
            "about 15% acceptance."
        ),
    )


def customer_value_task(
    cache_dir: Path | None = None,
    *,
    test_size: float = 0.25,
    seed: int = 42,
) -> TabularTask:
    """Predict two-year spend from profile, engagement and campaign history."""
    frame = load_customer_marketing(cache_dir)
    target = "total_spend"

    # The per-category amounts sum to the target, and the per-channel purchase
    # counts are close to a restatement of it. Both go, leaving demographics,
    # engagement and campaign history — a question worth asking.
    features = _feature_columns(
        frame, drop=[target, *_SPEND_COLUMNS, *_PURCHASE_COUNT_COLUMNS, "Response"]
    )
    train, test = stratified_split(frame, "Response", test_size=test_size, seed=seed)
    categoricals = [c for c in _MARKETING_CATEGORICALS if c in features]

    return TabularTask(
        name="ecommerce-customer-value",
        industry="ecommerce",
        task_type="regression",
        target=target,
        X_train=train[features].reset_index(drop=True),
        y_train=train[target].astype(float).reset_index(drop=True),
        X_test=test[features].reset_index(drop=True),
        y_test=test[target].astype(float).reset_index(drop=True),
        categorical_features=categoricals,
        description=(
            "Two-year customer spend from demographics, web engagement and prior "
            "campaign responses. Per-category amounts and per-channel purchase "
            "counts are excluded: the first sum to the target and the second "
            "restate it, so keeping either would score well while learning "
            "nothing."
        ),
    )


def build_tasks(cache_dir: Path | None = None, **kwargs) -> list[TabularTask]:
    return [
        conversion_task(cache_dir, **kwargs),
        page_value_task(cache_dir, **kwargs),
        campaign_response_task(cache_dir, **kwargs),
        customer_value_task(cache_dir, **kwargs),
    ]
