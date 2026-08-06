# TabFM Lab — zero-shot tabular modelling for e-commerce and sports betting

Benchmarks [Google Research's **TabFM**](https://github.com/google-research/tabfm), a
pretrained tabular foundation model, against tuned conventional baselines on four
supervised tasks drawn from two industries.

TabFM does not train on your data. It reads the training rows as *context* and answers
query rows in a single forward pass through frozen weights, the way a language model
answers from a prompt. The interesting question is not whether that works at all, but
whether it beats a gradient-boosted tree that has actually been fitted to the problem —
and whether its probabilities are calibrated well enough to act on. This repository is
built to answer both, honestly, on public data.

The models are then packaged for use: fitted models are persisted as pickled artifacts,
served over a FastAPI service with a static web UI, containerised, and deployed with
Kubernetes manifests. Jump to [Serving](#serving-the-models), [Docker](#docker) or
[Kubernetes](#kubernetes) for that half.

## Run it

### The short version — Streamlit

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

That is the whole thing. It opens a browser at `http://localhost:8501`, trains
the models on first run if `artifacts/` is empty, and gives you both industries
in one page. No API to start, no build step, no separate training command.

`streamlit_app.py` is a single file that imports the same package the rest of
the project uses — the same pickled artifacts, the same `MatchFeaturizer`, the
same betting maths — so it and the FastAPI service cannot disagree about a
prediction.

### For a public link — Streamlit Community Cloud

Free, and it deploys straight from this repository.

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `tumi-m/Tabfm`
   - **Branch:** the branch you want to serve
   - **Main file path:** `streamlit_app.py`
4. Open **Advanced settings** and set **Python version** to **3.11** or newer —
   the package requires it, and the default may be older. This is the one field
   that is easy to miss and cannot be changed later without deleting and
   redeploying the app.
5. **Deploy**.

The four trained models are committed under `artifacts/`, so the app loads them
and serves immediately rather than training on boot. That matters more than it
sounds: Community Cloud puts an app to sleep after 12 hours without traffic, and
without the artifacts every wake-up would retrain from scratch.

Regenerating them is `make train`, then commit the changed `.pkl` files. Read
the security note under [About the pickles](#about-the-pickles) first — a pickle
arriving in a pull request is a code change in disguise, so review those files
as you would review code.

Community Cloud gives roughly 1 GB of memory, which is ample here: the four
models total 1.2 MB and load in under a second.

### The other ways

**Nothing installed at all — GitHub Codespaces.** *Code* → *Codespaces* →
*Create codespace*. The devcontainer installs, trains and starts the API; port
8000 forwards automatically.

**Docker.** The image CI publishes has the models baked in, so it serves
immediately:

```bash
docker run -p 8000:8000 ghcr.io/tumi-m/tabfm-lab-api:latest
```

From a clone, one command builds and runs the same thing, training on first
start into a named volume:

```bash
docker compose -f docker/docker-compose.yml up
```

**Locally with the API** rather than Streamlit — see [Install](#install) and
[Serving](#serving-the-models). That route gives you the REST endpoints, the
static web UI and OpenAPI docs at `/docs`, which the Streamlit app does not.

### Which one?

| You want | Use |
|---|---|
| To click around and see predictions | Streamlit (`streamlit run streamlit_app.py`) |
| A shareable public link, free | Streamlit Community Cloud |
| To call it from other software | The FastAPI service (`make serve`) |
| To deploy it properly | Docker image → Kubernetes (`k8s/`) |

## The seven tasks

| Task | Industry | Type | Target | Why it matters |
|---|---|---|---|---|
| `ecommerce-conversion` | E-commerce | Classification | Session ends in a purchase | The funnel question. ~15% positive, so calibration matters more than accuracy. |
| `ecommerce-page-value` | E-commerce | Regression | Google Analytics Page Value | Per-session revenue proxy. Heavily zero-inflated. |
| `ecommerce-campaign-response` | E-commerce | Classification | Accepts the next campaign | Direct-marketing response. ~15% positive. |
| `ecommerce-customer-value` | E-commerce | Regression | Two-year customer spend | Customer value from profile and engagement alone. |
| `sports-match-result` | Sports betting | Classification | Home / draw / away | The 1X2 market, scored against real bookmaker prices. |
| `sports-total-goals` | Sports betting | Regression | Total goals in the match | What the over/under 2.5 market prices. |
| `sports-multi-league` | Sports betting | Classification | Home / draw / away | The same 1X2 question across five leagues — does the representation transfer? |

## Data

All data is public and downloaded on first run, then cached under `data/raw/`.

**E-commerce — [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)**
12,330 browsing sessions from an online retailer over one year, each from a distinct
user, with behavioural counters, Analytics page metrics and traffic attributes
(Sakar et al., 2018). Licensed CC BY 4.0.

**E-commerce — [Customer Personality / marketing](https://github.com/nailson/ifood-data-business-analyst-test)**
2,240 retail customers with two years of category spend, purchase channels by
channel, five prior campaign outcomes and demographics. Two quirks are handled
rather than ignored: `Marital_Status` contains junk levels (`Absurd`, `YOLO`)
that are folded into `Other` rather than dropped, since the rows are otherwise
fine; and a handful of birth years are implausible (1893), so ages outside
18–100 become missing and are imputed.

**Sports betting — [football-data.co.uk](https://www.football-data.co.uk/)**
Match results for five leagues — Premier League, La Liga, Serie A, Bundesliga,
Ligue 1 — from 2015/16, free for personal use. The single-league tasks use the
Premier League; `sports-multi-league` uses all five, about 17,500 matches.
Team names do not collide across these five competitions, and no club moves
between them, so one Elo table keyed by team name stays unambiguous.
The loader tries football-data.co.uk first because it ships **Bet365 closing odds**,
and falls back to the daily
[datahub mirror](https://github.com/datasets/football-datasets) when upstream is
unreachable. The mirror carries match statistics only, so on that path the betting
simulation is skipped rather than run against invented prices — see
[Odds availability](#odds-availability).

Adding leagues or seasons is a one-liner:

```python
from tabfm_lab.data.sports import match_result_task
task = match_result_task(leagues=("premier-league", "la-liga"), seasons=("2021", "2122", "2223"))
```

## Methodology

The parts worth scrutinising in a project like this are the ones that quietly inflate
scores. Three are handled explicitly.

### No post-match information in sports features

A football result row is mostly things you only know *after* the whistle: shots,
corners, cards, half-time score. Using any of them to predict the same match is
leakage, and it produces a backtest that looks profitable and a strategy that loses
money.

Every sports feature is therefore built in one chronological pass in which a match is
featurised from state accumulated by *earlier* matches only, and the state is updated
afterwards. That yields Elo ratings (with home advantage and a margin-of-victory
correction), rolling five-match form, venue-specific form, rest days and head-to-head
history. Matches where either team has fewer than five prior matches are **dropped
rather than imputed**, so no model trains on fabricated form.

This is pinned by a property test: perturbing the last match's result must not change
any earlier row's features (`tests/test_sports_features.py`). The feature list is also
derived from the engineered columns rather than by subtracting a denylist from the raw
frame, so a new post-match column appearing upstream cannot silently become a feature.

### Splits that respect time

The sports tasks train on earlier seasons and test on the most recent two. A random
split would let a model learn from matches played after the ones it is scored on. The
e-commerce tasks use a stratified random split, which is appropriate there because
sessions are independent and unordered.

For `ecommerce-page-value`, `Revenue` is dropped from the features: it is the
downstream outcome of the same session, and keeping it would leak the answer into a
revenue-proxy prediction.

### Probabilities, not just accuracy

Every classification task reports log loss, Brier score and expected calibration error
alongside accuracy. A model that is two points more accurate but overconfident loses
money against a bookmaker; a well-calibrated one can be profitable without ever being
the most accurate.

## Betting evaluation

When bookmaker odds are present the sports tasks go further than metrics.

1. **Strip the overround.** Raw reciprocals of decimal odds sum to more than one — the
   bookmaker's margin. Normalising recovers the market's true implied probabilities.
2. **Find value.** For each outcome, expected value per unit staked is
   `p_model x odds - 1`. A bet is placed only where that clears a threshold (default 5%),
   because a model fractionally more confident than the market is usually just wrong.
3. **Settle honestly.** Flat staking (1 unit) and fractional Kelly (¼ Kelly, capped at
   5% of bankroll) are both simulated, reporting ROI, hit rate, average odds and maximum
   drawdown.
4. **Show the benchmark.** The bookmaker's own de-margined prices are scored with the
   same metrics, in the same table.

For `sports-total-goals` the point forecast is converted into the probability the market
actually prices, `P(total > 2.5)`, under a Poisson assumption.

**A negative ROI against closing prices is the expected result.** Closing odds are close
to efficient and carry a margin. The quantity of interest is the *gap* to the benchmark,
not the sign of the profit. This repository is a modelling study, not a betting system.

## Install

Requires Python ≥ 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

That gets you the pipeline and baselines. TabFM itself is optional, because its weights
are a large download with a separate licence:

```bash
pip install "tabfm[pytorch]"
```

On a machine without a GPU, install the CPU build of torch first to avoid pulling ~3 GB
of unused CUDA libraries:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "tabfm[pytorch]"
```

A JAX backend also exists (`pip install "tabfm[jax]"`, then `--backend jax`).

## Usage

```bash
# Everything: four tasks, TabFM plus baselines
tabfm-lab

# Baselines only — no TabFM weights needed
tabfm-lab --no-tabfm

# One task, with a larger TabFM context and context-ensembling
tabfm-lab --task sports-match-result --context-rows 512 --n-estimators 8
```

Results are written to `reports/results.json` and `reports/results.md`, and echoed to
stdout. `tabfm-lab --help` lists every option; `--list-tasks` names the tasks.

```bash
pytest          # unit tests, no network required
```

## Serving the models

Training and serving are separate. `tabfm-lab-train` fits the models and writes
one pickle per task; the API only ever unpickles, so the request path does no
downloading and no fitting, and a readiness probe that passes means the process
can actually answer.

```bash
tabfm-lab-train                      # writes artifacts/*.pkl
make serve                           # http://localhost:8000
```

The UI is at `/`, the OpenAPI docs at `/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness. Never touches the models. |
| `GET /api/ready` | Readiness. 503 until at least one artifact loads. |
| `GET /api/models` | Loaded models with their test-split metrics. |
| `GET /api/models/{task}` | Input schema: fields, types, choices, defaults. |
| `POST /api/models/{task}/predict` | Batch prediction, up to 1,000 records. |
| `POST /api/fixtures/predict` | Score a football fixture from two team names. |

Records may be partial — omitted fields fall back to the training median or
modal value, so a caller only has to send what it actually knows:

```bash
curl -s localhost:8000/api/models/ecommerce-conversion/predict \
  -H 'Content-Type: application/json' \
  -d '{"records":[{"PageValues":42,"ProductRelated":48,"ExitRates":0.01}]}'
```

The fixture endpoint is the interesting one. Rather than making a caller supply
35 engineered features — which they could not compute, and could trivially get
wrong — the server rebuilds them from the `MatchFeaturizer` stored inside the
artifact. That is the same object, running the same code, that produced the
training set, so serving features cannot drift from training features.

```bash
curl -s localhost:8000/api/fixtures/predict \
  -H 'Content-Type: application/json' \
  -d '{"home_team":"Arsenal","away_team":"Chelsea",
       "odds":{"home":2.10,"draw":3.40,"away":3.60}}'
```

Supplying odds adds the value analysis: model probability against the
de-margined market price, the edge on each outcome, and a flag above +5%.

### About the pickles

A fitted TabFM estimator holds two very different things — the in-context
training rows, which are small and genuinely part of the fitted state, and a
reference to the frozen foundation model, which is hundreds of megabytes and
identical for every task. Pickling it naively copies the weights into every
artifact.

So `TabFMModel.__getstate__` drops the weights and `_ensure_estimator` reloads
them on first use, which is nearly free because TabFM's own `load()` keeps a
process-wide cache keyed by `(model_type, checkpoint_path, device, dtype)`. Four
artifacts in one server share one copy of the model. The baseline artifacts in
this repo come out at 180–420 KB each.

> **Unpickling executes code in the file.** The registry loads only from the
> directory an operator configures, never from request data. Treat artifacts as
> deployable code — build them in your own pipeline, don't accept them from
> users. Artifacts are gitignored for the same reason.

## Docker

```bash
docker compose -f docker/docker-compose.yml up      # build, train, serve
```

The entrypoint checks the artifact directory on boot. If it is empty it trains
first; if models are already there — because the volume was populated by an
earlier run, or because they were baked into the image — it serves straight
away. So a bare `docker run` is always enough, and never trains twice.

Explicit commands bypass that entirely, which is how the Kubernetes training Job
and the `retrain` service work:

```bash
docker compose -f docker/docker-compose.yml run --rm retrain
```

Two build arguments:

| Argument | Default | Effect |
|---|---|---|
| `BAKE_ARTIFACTS` | `false` | Train during the build so the image starts instantly. CI sets this. |
| `INSTALL_TABFM` | `false` | Add TabFM and torch. Off because serving needs neither unless an artifact contains a TabFM model, and torch multiplies image size by roughly ten. |

```bash
docker build -f docker/Dockerfile \
  --build-arg INSTALL_TABFM=true --build-arg BAKE_ARTIFACTS=true \
  -t tabfm-lab-api .
```

The image runs as a non-root user, carries a `HEALTHCHECK`, and can take
artifacts at runtime from a volume — so shipping a retrained model does not
require rebuilding.

## CI

`.github/workflows/ci.yml` lints and runs the test suite on every push, then
builds the image with the models baked in, pushes it to GHCR, and smoke-tests
the published image by starting it and hitting `/api/ready`, the UI and a real
prediction endpoint.

That is also the answer to "does the container actually work?" — the image build
is verified there, on a runner with a Docker daemon and unrestricted network,
rather than being asserted here.

## Kubernetes

```bash
kubectl apply -k k8s/
```

That creates a namespace, config, two PVCs, a training Job, a 2-replica
Deployment with a PodDisruptionBudget, a Service, an Ingress and an HPA.

A few choices worth knowing about:

- **Liveness and readiness point at different endpoints on purpose.** Liveness
  hits `/api/health`, which never touches the models, so a bad artifact drains
  traffic instead of triggering a restart loop. Readiness hits `/api/ready`,
  which returns 503 until an artifact loads, so a pod with an empty volume stays
  out of the Service rather than serving 404s.
- **A startup probe covers the unpickling window**, giving up to two minutes of
  cold start before liveness begins.
- **Scale with replicas, not workers.** Every worker unpickles its own copy of
  every model, so `WEB_CONCURRENCY` multiplies memory. The HPA scales up quickly
  and down slowly, since a cold start pays for the unpickle again.
- **The artifact PVC is ReadWriteMany**, because several replicas mount it at
  once. If your cluster has no RWX StorageClass, `k8s/pvc.yaml` documents the two
  alternatives.

On a first install the Deployment stays unready until the training Job finishes
writing artifacts. That is the readiness probe working, not a failure.

## Baseline results

From `tabfm-lab --no-tabfm` on Premier League 2015/16–2023/24 (train 2015/16–2021/22,
test 2022/23–2023/24) and the full Online Shoppers dataset. Lower is better except for
AUC and R².

| Task | Best model | Headline metrics |
|---|---|---|
| `ecommerce-conversion` | HistGradientBoosting | log loss **0.235**, ROC AUC **0.927**, PR AUC 0.730, accuracy 0.898 |
| `ecommerce-page-value` | HistGradientBoosting | RMSE **1.137** (log1p space), R² 0.168 |
| `ecommerce-campaign-response` | Logistic Regression | log loss **0.263**, ROC AUC **0.895**, PR AUC 0.656, accuracy 0.886 |
| `ecommerce-customer-value` | HistGradientBoosting | RMSE **272.8**, MAE 177.5, R² **0.803** |
| `sports-match-result` | Logistic Regression | log loss **0.976**, accuracy 0.549 |
| `sports-multi-league` | Logistic Regression | log loss **0.980**, accuracy 0.532, calibration error **0.021** |
| `sports-total-goals` | HistGradientBoosting | RMSE **1.744**, R² −0.006 |

Five things in that table are worth reading carefully, because they are the results a
leakage-free setup produces and an unsound one hides.

- **Football accuracy of 55% is the honest ceiling**, not a weak model. Bookmakers with
  vastly more information sit around 0.97–1.00 log loss on 1X2 markets. A model
  reporting 80% accuracy on match results has almost certainly seen post-match data.
- **Total goals is close to unpredictable**: R² hovers around zero, and the boosted tree
  barely beats predicting the mean. That is the correct finding. Match totals are
  dominated by variance that pre-match features cannot reach.
- **Linear beats boosting on both small, noisy classification tasks** — the 1X2 markets
  and campaign response. With one to three thousand training rows and low
  signal-to-noise, the flexible model overfits. This is exactly the regime TabFM's
  zero-shot claim targets, which makes these the most interesting tasks to re-run once
  weights are available.
- **More data buys calibration, not accuracy.** Going from one league to five barely
  moves accuracy (0.549 → 0.532) or log loss (0.976 → 0.980), but calibration error
  halves for the linear model (0.044 → 0.021) and drops sevenfold for boosting
  (0.044 → 0.006). For a betting model that is the trade that matters: a well-calibrated
  55% is worth more than an overconfident 57%.
- **Customer value is where boosting earns its keep**: R² 0.803 against ridge regression's
  0.043. Almost all of that gap is nonlinearity and interaction between income, children
  at home and web-visit frequency — and it holds up only because the per-category
  amounts and per-channel purchase counts are excluded from the features. Leave them in
  and R² approaches 1.0 while the model learns nothing.

The e-commerce conversion numbers line up with the published literature on this dataset
(~0.89–0.90 accuracy), which is a useful sanity check that the pipeline is sound.

**TabFM is not in this table.** These runs were produced in an environment whose egress
policy blocks `huggingface.co`, so the pretrained weights could not be downloaded and
the TabFM rows were recorded as skipped rather than silently omitted. The integration is
written against the released API — verified against the installed package's real
signatures for `TabFMClassifier`, `TabFMRegressor` and `tabfm_v1_0_0_pytorch.load()` —
but it has not been executed here. On a machine that can reach Hugging Face,
`tabfm-lab` fills in the missing rows with no code changes; if yours cannot, download the
checkpoint elsewhere and pass `--checkpoint-path`.

## Odds availability

The betting simulation needs the `B365H/D/A` (and `B365>2.5`/`B365<2.5`) columns that
football-data.co.uk publishes. If your network can only reach the GitHub mirror, those
columns are absent and the pipeline logs that it is skipping the ROI analysis; the
classification and regression metrics are unaffected. To restore it, fetch the season
CSVs from football-data.co.uk on a machine that can reach it and drop them into
`data/raw/` as `football_premier-league_<season>.csv`.

## Project layout

```
src/tabfm_lab/
├── data/
│   ├── sources.py     # URL chains, caching, graceful fallback
│   ├── base.py        # TabularTask container, target transforms
│   ├── ecommerce.py   # Online Shoppers -> conversion + page-value tasks
│   └── sports.py      # football-data -> Elo/form features, 1X2 + goals tasks
├── models/
│   ├── preprocess.py  # imputation + encoding, fitted on train only
│   ├── tabfm_backend.py # TabFM adapter, version-tolerant kwargs
│   └── baselines.py   # prior/linear/HistGradientBoosting
├── evaluation/
│   ├── metrics.py     # accuracy, log loss, Brier, calibration, RMSE/MAE/R²
│   └── betting.py     # overround removal, value bets, Kelly, Poisson lines
├── serving/
│   ├── artifacts.py   # pickle format, field specs, registry
│   ├── train.py       # tabfm-lab-train entry point
│   ├── schemas.py     # request/response models
│   └── app.py         # FastAPI service
├── pipeline.py        # orchestration
├── reporting.py       # Markdown rendering
└── cli.py             # tabfm-lab entry point

streamlit_app.py       # single-file Streamlit UI — the simplest way to run it
frontend/              # static UI (no build step) served by the API
docker/                # Dockerfile + compose stack
k8s/                   # namespace, PVCs, training Job, Deployment, HPA, Ingress
```

There are two front ends because they answer different needs: Streamlit for
looking at predictions with nothing installed, the API plus static UI for being
called by other software and deployed behind Kubernetes. Both read the same
artifacts through the same code.

## Licensing

- **This code**: Apache-2.0 (`LICENSE`).
- **TabFM source**: Apache-2.0.
- **TabFM pretrained weights**: distributed under `tabfm-non-commercial-v1.0`, which
  restricts them to **non-commercial, non-production use**. Check the terms before
  using any TabFM result commercially — the baselines in this repo carry no such
  restriction.
- **Data**: UCI Online Shoppers is CC BY 4.0; football-data.co.uk is free for personal
  use. Neither is redistributed here — both are downloaded at runtime.

## Limitations

- TabFM's context window is bounded (`--context-rows`), so on the larger splits it reads
  a sample rather than the full training table. `--n-estimators` averages over several
  sampled contexts to cut the resulting variance.
- Football results are close to Poisson but mildly overdispersed, so the over/under
  probabilities are a first approximation rather than a fitted scoreline model.
- Elo is a single rating per team. It carries across promotion and relegation but does
  not model squad changes, injuries or fixture congestion beyond rest days.
- Metrics come from one train/test split per task. Season-to-season variance in football
  is large; treat small differences between models as noise.

## Responsible use

The betting code exists to measure how far a model sits from an efficient market. It is
not advice, and it is not a system to stake money on. Gambling carries real financial
risk, and a backtest — however carefully built — is not evidence of future profit.
