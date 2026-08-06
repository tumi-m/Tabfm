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

import inspect
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
from tabfm_lab.models import tabfm_backend  # noqa: E402
from tabfm_lab.serving.artifacts import ArtifactRegistry  # noqa: E402

CONVERSION = "ecommerce-conversion"
PAGE_VALUE = "ecommerce-page-value"
CAMPAIGN = "ecommerce-campaign-response"
CUSTOMER_VALUE = "ecommerce-customer-value"
MATCH_RESULT = "sports-match-result"
TOTAL_GOALS = "sports-total-goals"
MULTI_LEAGUE = "sports-multi-league"

OVER_UNDER_LINE = 2.5
EDGE_THRESHOLD = 0.05
SESSION_BASE_RATE = 0.155
CAMPAIGN_BASE_RATE = 0.149

# Categorical slots 1-3 from the project's validated palette. They clear the
# colour-vision and normal-vision separation gates against both surfaces; every
# bar is still labelled, so identity never rests on colour alone.
OUTCOMES = [
    ("H", "Home win", "#2a78d6"),
    ("D", "Draw", "#eb6834"),
    ("A", "Away win", "#1baf7a"),
]
ACCENT = "#2a78d6"

st.set_page_config(page_title="TabFM Lab", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def artifact_dir() -> Path:
    return Path(os.environ.get("TABFM_ARTIFACT_DIR", REPO_ROOT / "artifacts"))


@st.cache_resource(show_spinner=False)
def load_models() -> ArtifactRegistry:
    """Load the pickled models, training them once if they are missing."""
    directory = artifact_dir()
    registry = ArtifactRegistry(directory).load()
    if len(registry):
        return registry

    from tabfm_lab.data import TASK_BUILDERS
    from tabfm_lab.serving.train import train_task

    with st.status("First run — training the models", expanded=True) as status:
        st.write(
            "Downloading the public datasets and fitting the models. This "
            "happens once; later runs load the saved files in about a second."
        )
        for name, build in sorted(TASK_BUILDERS.items()):
            st.write(f"Building **{name}** …")
            try:
                train_task(build(None), directory, model_choice="auto")
            except Exception as exc:  # noqa: BLE001 - shown, not swallowed
                st.warning(f"Could not build {name}: {exc}")
        status.update(label="Models ready", state="complete", expanded=False)

    return ArtifactRegistry(directory).load()


# --------------------------------------------------------------------------
# Shared rendering helpers
# --------------------------------------------------------------------------
def bar(label: str, value: float, colour: str = ACCENT) -> str:
    width = max(0.0, min(1.0, float(value))) * 100
    return f"""
    <div style="display:grid;grid-template-columns:7rem 1fr 3.5rem;
                align-items:center;gap:.6rem;margin-bottom:.45rem;">
      <span style="font-size:.88rem;opacity:.8;">{label}</span>
      <span style="display:block;height:12px;border-radius:4px;
                   background:rgba(128,128,128,.22);overflow:hidden;">
        <span style="display:block;height:100%;width:{width:.1f}%;
                     border-radius:4px;background:{colour};"></span>
      </span>
      <span style="font-size:.88rem;font-weight:600;text-align:right;
                   font-variant-numeric:tabular-nums;">{value:.1%}</span>
    </div>
    """


def probability_bars(probabilities: dict[str, float]) -> None:
    rows = [
        bar(label, probabilities[key], colour)
        for key, label, colour in OUTCOMES
        if key in probabilities
    ]
    st.markdown("".join(rows), unsafe_allow_html=True)


def restrict(record: dict, registry: ArtifactRegistry, task: str) -> list[dict]:
    """Narrow a record to the columns a given model was trained on."""
    allowed = set(registry.get(task).metadata.feature_columns)
    return [{k: v for k, v in record.items() if k in allowed}]


def positive_probability(registry: ArtifactRegistry, task: str, record: dict) -> float | None:
    """P(class 1) for a binary classifier artifact."""
    artifact = registry.get(task)
    frame = artifact.frame_from_records(restrict(record, registry, task))
    probabilities = artifact.predict_proba(frame)
    classes = [str(c) for c in (artifact.metadata.classes or [])]
    if probabilities is None or "1" not in classes:
        return None
    return float(probabilities[0][classes.index("1")])


def missing_notice(registry: ArtifactRegistry, needed: list[str]) -> bool:
    absent = [t for t in needed if t not in registry]
    if absent:
        st.info(f"Not loaded: {', '.join(absent)}. Run `make train` to build them.")
    return not absent


# --------------------------------------------------------------------------
# Page: web sessions
# --------------------------------------------------------------------------
def sessions_page(registry: ArtifactRegistry) -> None:
    st.header("Web sessions")
    st.caption(
        "UCI Online Shoppers Purchasing Intention — 12,330 browsing sessions "
        "from an online retailer, each from a distinct user over one year."
    )
    if not missing_notice(registry, [CONVERSION]):
        return

    specs = {f.name: f for f in registry.get(CONVERSION).metadata.fields}
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Describe a session")
        st.caption("Anything you leave alone uses the median or modal training value.")
        record: dict = {}
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
            if specs.get("Month") and specs["Month"].choices:
                record["Month"] = st.selectbox("Month", specs["Month"].choices)
            if specs.get("VisitorType") and specs["VisitorType"].choices:
                record["VisitorType"] = st.selectbox("Visitor type", specs["VisitorType"].choices)
        record["Weekend"] = 1 if st.checkbox("Weekend visit") else 0

    with right:
        st.subheader("Prediction")
        probability = positive_probability(registry, CONVERSION, record)
        if probability is None:
            st.warning("The conversion model does not expose probabilities.")
            return

        st.metric(
            "Probability this session ends in a purchase",
            f"{probability:.1%}",
            delta=f"{(probability - SESSION_BASE_RATE) * 100:+.1f} pts vs the "
                  f"{SESSION_BASE_RATE:.1%} base rate",
        )
        st.progress(max(0.0, min(1.0, probability)))

        if PAGE_VALUE in registry:
            artifact = registry.get(PAGE_VALUE)
            frame = artifact.frame_from_records(restrict(record, registry, PAGE_VALUE))
            # Modelled on the log1p scale; report the original units.
            st.metric(
                "Predicted page value",
                f"{np.expm1(float(artifact.predict(frame)[0])):.2f}",
            )
            st.caption("Revenue proxy for the session, in Analytics page-value units.")


# --------------------------------------------------------------------------
# Page: customers
# --------------------------------------------------------------------------
def customers_page(registry: ArtifactRegistry) -> None:
    st.header("Customers")
    st.caption(
        "Customer Personality / marketing dataset — 2,240 retail customers with "
        "two years of category spend, purchase channels and five prior campaigns."
    )
    if not missing_notice(registry, [CAMPAIGN]):
        return

    specs = {f.name: f for f in registry.get(CAMPAIGN).metadata.fields}
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Describe a customer")
        record: dict = {}
        a, b = st.columns(2)
        with a:
            record["age"] = st.number_input("Age", min_value=18, max_value=100, value=45)
            record["Income"] = st.number_input(
                "Household income", min_value=0, value=52000, step=1000
            )
            record["Recency"] = st.slider("Days since last purchase", 0, 99, 30)
            record["tenure_days"] = st.slider("Customer for (days)", 0, 900, 400)
            record["NumWebVisitsMonth"] = st.slider("Web visits per month", 0, 20, 5)
        with b:
            if specs.get("Education") and specs["Education"].choices:
                record["Education"] = st.selectbox("Education", specs["Education"].choices)
            if specs.get("Marital_Status") and specs["Marital_Status"].choices:
                record["Marital_Status"] = st.selectbox(
                    "Marital status", specs["Marital_Status"].choices
                )
            record["Kidhome"] = st.slider("Children at home", 0, 3, 0)
            record["Teenhome"] = st.slider("Teenagers at home", 0, 3, 0)
            record["MntWines"] = st.number_input("Spend on wine", min_value=0, value=300, step=25)
            record["MntMeatProducts"] = st.number_input(
                "Spend on meat", min_value=0, value=150, step=25
            )
        prior = st.slider("Prior campaigns accepted", 0, 5, 0)
        for index in range(1, 6):
            record[f"AcceptedCmp{index}"] = 1 if index <= prior else 0

    with right:
        st.subheader("Prediction")
        probability = positive_probability(registry, CAMPAIGN, record)
        if probability is not None:
            st.metric(
                "Probability of accepting the next campaign",
                f"{probability:.1%}",
                delta=f"{(probability - CAMPAIGN_BASE_RATE) * 100:+.1f} pts vs the "
                      f"{CAMPAIGN_BASE_RATE:.1%} base rate",
            )
            st.progress(max(0.0, min(1.0, probability)))

        if CUSTOMER_VALUE in registry:
            artifact = registry.get(CUSTOMER_VALUE)
            frame = artifact.frame_from_records(restrict(record, registry, CUSTOMER_VALUE))
            st.metric("Predicted two-year spend", f"{float(artifact.predict(frame)[0]):,.0f}")
            st.caption(
                "Per-category amounts and purchase counts are deliberately not "
                "inputs here — the first sum to this target and the second "
                "restate it."
            )


# --------------------------------------------------------------------------
# Page: fixtures
# --------------------------------------------------------------------------
def fixtures_page(registry: ArtifactRegistry) -> None:
    st.header("Football fixtures")
    if not missing_notice(registry, [MATCH_RESULT]):
        return

    available = [t for t in (MATCH_RESULT, MULTI_LEAGUE) if t in registry]
    labels = {
        MATCH_RESULT: "Premier League only",
        MULTI_LEAGUE: "Five leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1)",
    }
    task = st.radio(
        "Model", available, format_func=lambda t: labels.get(t, t), horizontal=True
    )

    artifact = registry.get(task)
    st.caption(artifact.metadata.description)

    featurizer = artifact.featurizer
    teams = artifact.metadata.teams or []
    if featurizer is None or not teams:
        st.warning("This artifact has no featuriser. Retrain to enable fixtures.")
        return

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Pick a fixture")
        st.caption(
            "Pre-match features — Elo, rolling form, venue form, rest days, "
            "head-to-head — are rebuilt from match history by the same code that "
            "built the training set."
        )
        a, b = st.columns(2)
        home = a.selectbox("Home team", teams, index=0)
        away = b.selectbox("Away team", teams, index=min(1, len(teams) - 1))

        st.markdown("**Bookmaker odds** — optional, enables the value analysis")
        c1, c2, c3 = st.columns(3)
        prices = [
            c1.number_input("Home", min_value=0.0, value=0.0, step=0.05),
            c2.number_input("Draw", min_value=0.0, value=0.0, step=0.05),
            c3.number_input("Away", min_value=0.0, value=0.0, step=0.05),
        ]

    with right:
        st.subheader("Prediction")
        if home == away:
            st.warning("Pick two different teams.")
            return

        features = featurizer.features_for(home, away, featurizer.last_date)
        if not bool(features.pop("is_valid", True)):
            st.warning("One or both teams lack enough history for reliable features.")

        def row_for(name: str) -> pd.DataFrame:
            allowed = set(registry.get(name).metadata.feature_columns)
            return registry.get(name).frame_from_records(
                [{k: v for k, v in features.items() if k in allowed}]
            )

        classes = [str(c) for c in (artifact.metadata.classes or [])]
        raw = artifact.predict_proba(row_for(task))
        probabilities = {label: float(raw[0][i]) for i, label in enumerate(classes)}

        st.markdown(f"**{home} vs {away}**")
        probability_bars(probabilities)

        if TOTAL_GOALS in registry:
            expected = float(registry.get(TOTAL_GOALS).predict(row_for(TOTAL_GOALS))[0])
            over = float(poisson_over_probability(np.array([expected]), OVER_UNDER_LINE)[0])
            m1, m2 = st.columns(2)
            m1.metric("Expected total goals", f"{expected:.2f}")
            m2.metric("Over 2.5 goals", f"{over:.1%}")

        if all(price > 1.0 for price in prices):
            value_analysis(probabilities, prices)


def value_analysis(probabilities: dict[str, float], prices: list[float]) -> None:
    st.markdown("#### Value analysis")
    matrix = np.array([prices], dtype=float)
    market = implied_probabilities(matrix)[0]
    overround = float(np.sum(1.0 / matrix[0]) - 1.0)

    rows = []
    for index, (key, label, _colour) in enumerate(OUTCOMES):
        model_probability = probabilities.get(key, 0.0)
        edge = model_probability * prices[index] - 1.0
        rows.append(
            {
                "Outcome": label,
                "Model": f"{model_probability:.1%}",
                "Market": f"{market[index]:.1%}",
                "Odds": f"{prices[index]:.2f}",
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
# Page: how TabFM is used
# --------------------------------------------------------------------------
def tabfm_page(registry: ArtifactRegistry) -> None:
    st.header("How TabFM is used")
    st.caption(
        "TabFM is Google Research's tabular foundation model. This page shows "
        "the real integration — status read from the installed package, "
        "parameters read from its live signature, and code read from disk — so "
        "nothing here can drift from what actually runs."
    )

    status = tabfm_backend.availability()

    st.subheader("Status on this machine")
    # delta_color="off": these deltas are labels, not movements, and Streamlit
    # would otherwise render a green up-arrow beside a version string.
    c1, c2, c3 = st.columns(3)
    c1.metric("Package", "installed" if status["package_installed"] else "missing",
              status["package_version"] or "", delta_color="off")
    c2.metric("Backend", status["backend"],
              "available" if status["backend_available"] else "unavailable",
              delta_color="off")
    c3.metric("Weights", "not checked" if not status["weights_probed"]
              else ("loadable" if status["weights_loadable"] else "unavailable"))

    if status["detail"]:
        st.info(status["detail"])

    if status["backend_available"]:
        st.caption(
            f"Checking the weights downloads them from `{status['weights_repo']}` "
            "on first use — hundreds of megabytes — so it is not done automatically."
        )
        if st.button("Check the pretrained weights"):
            with st.spinner("Loading TabFM weights…"):
                probed = tabfm_backend.availability(probe_weights=True)
            (st.success if probed["weights_loadable"] else st.error)(probed["detail"])

    st.divider()

    st.subheader("What makes it different")
    st.markdown(
        """
Every baseline in this project is **fitted**: gradient boosting builds hundreds
of trees from the training rows, and the fitted trees are the model.

TabFM is **not fitted on your data at all**. The weights are frozen. Calling
`fit` only prepares encoders and stores the training rows; at `predict` time
those rows are fed through the network *as context*, alongside the rows being
predicted, and the answer comes out of a single forward pass. It is the same
trick as showing a language model a few examples in a prompt.

Two consequences shape the code:

- **The context window is bounded.** On a large training split TabFM reads a
  sample, not the whole table. `n_estimators` re-samples the context several
  times and averages, trading latency for variance.
- **The weights are large and shared.** They are identical for every task, so a
  fitted estimator must not carry them into its pickle — see
  *About the pickles* in the README.
        """
    )

    st.divider()
    st.subheader("The integration, read from source")
    st.caption(
        "These are pulled with `inspect.getsource` from the module that runs, "
        "not pasted, so they cannot go stale."
    )

    excerpts = {
        "Loading the frozen weights": tabfm_backend._load_backend,
        "Fitting = preparing encoders and storing the context": tabfm_backend.TabFMModel.fit,
        "Keeping the weights out of the pickle": tabfm_backend.TabFMModel.__getstate__,
    }
    for title, function in excerpts.items():
        with st.expander(title):
            st.code(inspect.getsource(function), language="python")

    st.subheader("Using it")
    st.code(
        """# Install the model (CPU build of torch keeps this ~10x smaller)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "tabfm[pytorch]"

# Train with it. "auto" prefers TabFM and falls back to gradient boosting
# if the weights cannot be fetched, so a run always produces artifacts.
tabfm-lab-train --model tabfm
tabfm-lab-train --model tabfm --context-rows 512 --n-estimators 8

# No Hugging Face access? Download the checkpoint elsewhere and point at it.
tabfm-lab-train --model tabfm --checkpoint-path /path/to/checkpoint""",
        language="bash",
    )

    if status["package_installed"] and status["backend_available"]:
        st.subheader("Estimator parameters, from the installed signature")
        try:
            parameters = tabfm_backend.constructor_parameters("classification")
            passed = {"max_num_rows", "n_estimators", "batch_size", "random_state"}
            frame = pd.DataFrame(
                [
                    {
                        "Parameter": name,
                        "Default": repr(value),
                        "Set by this project": "yes" if name in passed else "",
                    }
                    for name, value in parameters.items()
                ]
            )
            st.dataframe(frame, hide_index=True, width="stretch", height=320)
            st.caption(
                "The wrapper filters what it passes against this signature, so a "
                "release that renames a knob falls back to its default instead of "
                "raising part-way through a run."
            )
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read the estimator signature: {exc}")

    st.divider()
    st.subheader("Which model is actually serving")
    used = sorted({a.metadata.model_name for a in registry.all()})
    if any("TabFM" in name for name in used):
        st.success(f"Artifacts were built with: {', '.join(used)}")
    else:
        st.warning(
            f"Artifacts were built with: {', '.join(used) or 'nothing loaded'}. "
            "TabFM was unavailable when these were trained, so `--model auto` "
            "fell back to gradient boosting. Retrain on a machine that can reach "
            "Hugging Face to see TabFM here."
        )
    st.caption(
        "Licensing: TabFM's source is Apache-2.0, but its pretrained weights are "
        "released under `tabfm-non-commercial-v1.0` and are restricted to "
        "non-commercial, non-production use. The baselines carry no such limit."
    )


# --------------------------------------------------------------------------
# Page: models
# --------------------------------------------------------------------------
def models_page(registry: ArtifactRegistry) -> None:
    st.header("Models")
    st.caption(
        "One row per pickled artifact. Metrics come from the held-out test "
        "split at training time — for the football tasks that is a temporal "
        "split, so the test seasons come strictly after the training ones."
    )

    rows = []
    for artifact in registry.all():
        meta = artifact.metadata
        rows.append(
            {
                "Task": meta.task,
                "Industry": meta.industry,
                "Type": meta.task_type,
                "Model": meta.model_name,
                "Train": meta.n_train,
                "Test": meta.n_test,
                "Features": len(meta.feature_columns),
                **{
                    k: round(v, 4)
                    for k, v in list(meta.metrics.items())[:4]
                    if v == v
                },
            }
        )

    if not rows:
        st.info("No models loaded.")
        return

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Task descriptions"):
        for artifact in registry.all():
            st.markdown(f"**`{artifact.metadata.task}`** — {artifact.metadata.description}")

    if registry.errors:
        st.error("Some artifacts failed to load:")
        st.json(registry.errors)


# --------------------------------------------------------------------------
PAGES = {
    "Web sessions": sessions_page,
    "Customers": customers_page,
    "Football fixtures": fixtures_page,
    "How TabFM is used": tabfm_page,
    "Models": models_page,
}


def main() -> None:
    st.sidebar.title("TabFM Lab")
    st.sidebar.caption(
        "Zero-shot tabular models for e-commerce and sports betting, served "
        "from pickled artifacts."
    )

    registry = load_models()
    if not len(registry):
        st.error(
            "No models could be loaded or trained. If this machine has no "
            "internet access, run `tabfm-lab-train` somewhere that does and "
            "copy the resulting `artifacts/` directory here."
        )
        return

    choice = st.sidebar.radio("View", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.metric("Models loaded", len(registry))
    industries = sorted({a.metadata.industry for a in registry.all()})
    st.sidebar.caption("Industries: " + ", ".join(industries))
    st.sidebar.caption(
        "Predictions are a modelling exercise on public data, not advice. "
        "The betting view measures a model against a market; it is not a "
        "staking system."
    )

    PAGES[choice](registry)


main()
