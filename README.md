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

## The four tasks

| Task | Industry | Type | Target | Why it matters |
|---|---|---|---|---|
| `ecommerce-conversion` | E-commerce | Classification | Session ends in a purchase | The funnel question. ~15% positive, so calibration matters more than accuracy. |
| `ecommerce-page-value` | E-commerce | Regression | Google Analytics Page Value | Per-session revenue proxy. Heavily zero-inflated. |
| `sports-match-result` | Sports betting | Classification | Home / draw / away | The 1X2 market, scored against real bookmaker prices. |
| `sports-total-goals` | Sports betting | Regression | Total goals in the match | What the over/under 2.5 market prices. |

## Data

All data is public and downloaded on first run, then cached under `data/raw/`.

**E-commerce — [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)**
12,330 browsing sessions from an online retailer over one year, each from a distinct
user, with behavioural counters, Analytics page metrics and traffic attributes
(Sakar et al., 2018). Licensed CC BY 4.0.

**Sports betting — [football-data.co.uk](https://www.football-data.co.uk/)**
Match results for the English Premier League, 2015/16 to 2023/24, free for personal use.
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

## Baseline results

From `tabfm-lab --no-tabfm` on Premier League 2015/16–2023/24 (train 2015/16–2021/22,
test 2022/23–2023/24) and the full Online Shoppers dataset. Lower is better except for
AUC and R².

| Task | Best model | Headline metrics |
|---|---|---|
| `ecommerce-conversion` | HistGradientBoosting | log loss **0.235**, ROC AUC **0.927**, PR AUC 0.730, accuracy 0.898 |
| `ecommerce-page-value` | HistGradientBoosting | RMSE **1.137** (log1p space), R² 0.168 |
| `sports-match-result` | Logistic Regression | log loss **0.976**, accuracy 0.549, ROC AUC (ovr) 0.657 |
| `sports-total-goals` | HistGradientBoosting | RMSE **1.744**, R² −0.006 |

Three things in that table are worth reading carefully, because they are the results a
leakage-free setup produces and an unsound one hides.

- **Football accuracy of 55% is the honest ceiling**, not a weak model. Bookmakers with
  vastly more information sit around 0.97–1.00 log loss on 1X2 markets. A model
  reporting 80% accuracy on match results has almost certainly seen post-match data.
- **Total goals is close to unpredictable**: R² hovers around zero, and the boosted tree
  barely beats predicting the mean. That is the correct finding. Match totals are
  dominated by variance that pre-match features cannot reach.
- **Linear beats boosting on the 1X2 task.** With ~2,500 training rows and a
  low signal-to-noise ratio, the flexible model overfits. This is exactly the regime
  TabFM's zero-shot claim targets, which makes it the most interesting of the four
  tasks to re-run once weights are available.

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
├── pipeline.py        # orchestration
├── reporting.py       # Markdown rendering
└── cli.py             # tabfm-lab entry point
```

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
