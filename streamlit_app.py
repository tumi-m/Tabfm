"""TabFM Lab as a single Streamlit app.

    streamlit run streamlit_app.py

Everything runs in one process — no separate API, no build step, no train-then-
serve dance. Models are loaded from `artifacts/` if they are there and trained
on first run if they are not.

This is the same code path as the FastAPI service: the same pickled artifacts,
the same `MatchFeaturizer`, the same betting maths. Only the presentation
differs, so the two front ends cannot disagree about a prediction.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Import from src/ without requiring an install, so `streamlit run` works on a
# bare clone and on Streamlit Community Cloud.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tabfm_lab.evaluation.betting import (  # noqa: E402
    implied_probabilities,
    poisson_over_probability,
)
from tabfm_lab.serving.artifacts import ArtifactRegistry  # noqa: E402

CONVERSION = "ecommerce-conversion"
PAGE_VALUE = "ecommerce-page-value"
MATCH_RESULT = "sports-match-result"
TOTAL_GOALS = "sports-total-goals"

OVER_UNDER_LINE = 2.5
EDGE_THRESHOLD = 0.05
TRAIN_CONVERSION_RATE = 0.155

# Categorical slots 1-3 from the project's validated palette. They clear the
# colour-vision and normal-vision separation gates against both surfaces; every
# bar is still labelled, so identity never rests on colour alone.
OUTCOMES = [
    ("H", "Home win", "#2a78d6"),
    ("D", "Draw", "#eb6834"),
    ("A", "Away win", "#1baf7a"),
]

st.set_page_config(
    page_title="TabFM Lab",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def artifact_dir() -> Path:
    return Path(os.environ.get("TABFM_ARTIFACT_DIR", REPO_ROOT / "artifacts"))


@st.cache_resource(show_spinner=False)
def load_models() -> ArtifactRegistry:
    """Load the pickled models, training them once if they are missing.

    Cached as a resource, so the cost is paid once per process rather than on
    every widget interaction.
    """
    directory = artifact_dir()
    registry = ArtifactRegistry(directory).load()
    if len(registry):
        return registry

    from tabfm_lab.data import TASK_BUILDERS
    from tabfm_lab.serving.train import train_task

    with st.status("First run — training the models", expanded=True) as status:
        st.write(
            "Downloading the public datasets and fitting four models. "
            "This happens once; later runs load the saved files in a second."
        )
        for name, build in sorted(TASK_BUILDERS.items()):
            st.write(f"Building **{name}** …")
            try:
                task = build(None)
                train_task(task, directory, model_choice="auto")
            except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
                st.warning(f"Could not build {name}: {exc}")
        status.update(label="Models ready", state="complete", expanded=False)

    return ArtifactRegistry(directory).load()


def probability_bars(probabilities: dict[str, float]) -> None:
    """Labelled horizontal bars, one per outcome."""
    rows = []
    for key, label, colour in OUTCOMES:
        value = probabilities.get(key)
        if value is None:
            continue
        width = max(0.0, min(1.0, float(value))) * 100
        rows.append(
            f"""
            <div style="display:grid;grid-template-columns:6rem 1fr 3.5rem;
                        align-items:center;gap:.6rem;margin-bottom:.5rem;">
              <span style="font-size:.9rem;opacity:.8;">{label}</span>
              <span style="display:block;height:12px;border-radius:4px;
                           background:rgba(128,128,128,.22);overflow:hidden;">
                <span style="display:block;height:100%;width:{width:.1f}%;
                             border-radius:4px;background:{colour};"></span>
              </span>
              <span style="font-size:.9rem;font-weight:600;text-align:right;
                           font-variant-numeric:tabular-nums;">{value:.1%}</span>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# E-commerce
# --------------------------------------------------------------------------
def ecommerce_tab(registry: ArtifactRegistry) -> None:
    if CONVERSION not in registry:
        st.info("The e-commerce conversion model is not loaded.")
        return

    artifact = registry.get(CONVERSION)
    specs = {f.name: f for f in artifact.metadata.fields}

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Session")
        st.caption(
            "Describe a browsing session. Anything you leave alone uses the "
            "median or most common value from training."
        )

        record: dict[str, object] = {}
        a, b = st.columns(2)
        with a:
            record["PageValues"] = st.number_input(
                "Page value", min_value=0.0, value=0.0, step=1.0,
                help="Analytics revenue attributed to the pages in this session.",
            )
            record["ProductRelated"] = st.number_input(
                "Product pages viewed", min_value=0, value=18, step=1
            )
            record["ProductRelated_Duration"] = st.number_input(
                "Time on product pages (s)", min_value=0.0, value=600.0, step=30.0
            )
            record["Administrative"] = st.number_input(
                "Account pages viewed", min_value=0, value=1, step=1
            )
        with b:
            record["ExitRates"] = st.slider("Exit rate", 0.0, 0.2, 0.03, 0.005)
            record["BounceRates"] = st.slider("Bounce rate", 0.0, 0.2, 0.02, 0.005)
            record["SpecialDay"] = st.slider("Special day proximity", 0.0, 1.0, 0.0, 0.2)
            if "Month" in specs and specs["Month"].choices:
                record["Month"] = st.selectbox("Month", specs["Month"].choices)
            if "VisitorType" in specs and specs["VisitorType"].choices:
                record["VisitorType"] = st.selectbox(
                    "Visitor type", specs["VisitorType"].choices
                )
        record["Weekend"] = 1 if st.checkbox("Weekend visit") else 0

    with right:
        st.subheader("Prediction")

        # Each model has its own column set — the page-value model has no
        # PageValues column — so narrow the record per model.
        def restricted(task: str) -> list[dict[str, object]]:
            allowed = set(registry.get(task).metadata.feature_columns)
            return [{k: v for k, v in record.items() if k in allowed}]

        frame = artifact.frame_from_records(restricted(CONVERSION))
        probabilities = artifact.predict_proba(frame)
        classes = [str(c) for c in (artifact.metadata.classes or [])]
        probability = None
        if probabilities is not None and "1" in classes:
            probability = float(probabilities[0][classes.index("1")])

        if probability is None:
            st.warning("This model does not expose probabilities.")
            return

        # Percentage-point difference rather than a ratio: it carries a sign, so
        # Streamlit's arrow points the right way, and it stays readable for the
        # many sessions that sit far below the base rate (where a ratio rounds
        # to an unhelpful "0.0x").
        difference = (probability - TRAIN_CONVERSION_RATE) * 100
        st.metric(
            "Probability this session ends in a purchase",
            f"{probability:.1%}",
            delta=f"{difference:+.1f} pts vs the {TRAIN_CONVERSION_RATE:.1%} base rate",
        )
        st.progress(max(0.0, min(1.0, probability)))

        if PAGE_VALUE in registry:
            page_artifact = registry.get(PAGE_VALUE)
            page_frame = page_artifact.frame_from_records(restricted(PAGE_VALUE))
            # The target is modelled on the log1p scale; report original units.
            predicted = float(page_artifact.predict(page_frame)[0])
            st.metric("Predicted page value", f"{np.expm1(predicted):.2f}")
            st.caption("Revenue proxy for the session, in Analytics page-value units.")


# --------------------------------------------------------------------------
# Sports betting
# --------------------------------------------------------------------------
def sports_tab(registry: ArtifactRegistry) -> None:
    if MATCH_RESULT not in registry:
        st.info("The football models are not loaded.")
        return

    artifact = registry.get(MATCH_RESULT)
    featurizer = artifact.featurizer
    teams = artifact.metadata.teams or []
    if featurizer is None or not teams:
        st.warning("This artifact has no featuriser. Retrain to enable fixtures.")
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Fixture")
        st.caption(
            "Pick two teams. Pre-match features — Elo, rolling form, rest days — "
            "are rebuilt from match history by the same code that built the "
            "training set."
        )
        a, b = st.columns(2)
        home = a.selectbox("Home team", teams, index=0)
        away = b.selectbox("Away team", teams, index=min(1, len(teams) - 1))

        st.markdown("**Bookmaker odds** — optional, enables the value analysis")
        c1, c2, c3 = st.columns(3)
        odds_home = c1.number_input("Home", min_value=0.0, value=0.0, step=0.05)
        odds_draw = c2.number_input("Draw", min_value=0.0, value=0.0, step=0.05)
        odds_away = c3.number_input("Away", min_value=0.0, value=0.0, step=0.05)

    with right:
        st.subheader("Prediction")
        if home == away:
            st.warning("Pick two different teams.")
            return

        features = featurizer.features_for(home, away, featurizer.last_date)
        reliable = bool(features.pop("is_valid", True))
        if not reliable:
            st.warning("One or both teams lack enough history for reliable features.")

        def row_for(task: str) -> pd.DataFrame:
            allowed = set(registry.get(task).metadata.feature_columns)
            return registry.get(task).frame_from_records(
                [{k: v for k, v in features.items() if k in allowed}]
            )

        classes = [str(c) for c in (artifact.metadata.classes or [])]
        raw = artifact.predict_proba(row_for(MATCH_RESULT))
        probabilities = {label: float(raw[0][i]) for i, label in enumerate(classes)}

        st.markdown(f"**{home} vs {away}**")
        probability_bars(probabilities)

        expected_goals = None
        if TOTAL_GOALS in registry:
            expected_goals = float(registry.get(TOTAL_GOALS).predict(row_for(TOTAL_GOALS))[0])
            over = float(poisson_over_probability(np.array([expected_goals]), OVER_UNDER_LINE)[0])
            m1, m2 = st.columns(2)
            m1.metric("Expected total goals", f"{expected_goals:.2f}")
            m2.metric("Over 2.5 goals", f"{over:.1%}")

        prices = [odds_home, odds_draw, odds_away]
        if all(price > 1.0 for price in prices):
            value_analysis(probabilities, prices)


def value_analysis(probabilities: dict[str, float], prices: list[float]) -> None:
    """Model probabilities against the quoted market."""
    st.markdown("#### Value analysis")

    matrix = np.array([prices], dtype=float)
    market = implied_probabilities(matrix)[0]
    overround = float(np.sum(1.0 / matrix[0]) - 1.0)

    rows = []
    for index, (key, label, _colour) in enumerate(OUTCOMES):
        model_probability = probabilities.get(key, 0.0)
        price = prices[index]
        edge = model_probability * price - 1.0
        rows.append(
            {
                "Outcome": label,
                "Model": f"{model_probability:.1%}",
                "Market": f"{market[index]:.1%}",
                "Odds": f"{price:.2f}",
                "Edge": f"{edge:+.1%}",
                "Verdict": "✔ Value" if edge >= EDGE_THRESHOLD else "— No bet",
            }
        )

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Bookmaker overround on these prices: {overround:.1%}.")
    st.caption(
        "Edge is expected profit per unit staked: model probability × decimal "
        "odds − 1, flagged above +5%. This measures a model against a market; "
        "it is not betting advice. Closing prices are close to efficient and "
        "carry a margin, so a negative edge is the normal result."
    )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def models_tab(registry: ArtifactRegistry) -> None:
    st.subheader("Loaded models")
    st.caption(
        "One row per pickled artifact. Metrics come from the held-out test "
        "split at training time."
    )

    rows = []
    for artifact in registry.all():
        meta = artifact.metadata
        headline = ", ".join(
            f"{k} {v:.3f}" for k, v in list(meta.metrics.items())[:3] if v == v
        )
        rows.append(
            {
                "Task": meta.task,
                "Industry": meta.industry,
                "Type": meta.task_type,
                "Model": meta.model_name,
                "Train": f"{meta.n_train:,}",
                "Test": f"{meta.n_test:,}",
                "Metrics": headline,
            }
        )

    if not rows:
        st.info("No models loaded.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if registry.errors:
        st.error("Some artifacts failed to load:")
        st.json(registry.errors)


# --------------------------------------------------------------------------
def main() -> None:
    st.title("TabFM Lab")
    st.caption(
        "Zero-shot tabular models for e-commerce and sports betting, "
        "served straight from the pickled artifacts."
    )

    registry = load_models()
    if not len(registry):
        st.error(
            "No models could be loaded or trained. If this machine has no "
            "internet access, run `tabfm-lab-train` somewhere that does and "
            "copy the resulting `artifacts/` directory here."
        )
        return

    ecommerce, sports, models = st.tabs(["E-commerce", "Sports betting", "Models"])
    with ecommerce:
        ecommerce_tab(registry)
    with sports:
        sports_tab(registry)
    with models:
        models_tab(registry)


main()
