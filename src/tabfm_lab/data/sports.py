"""Sports-betting tasks built on football-data.co.uk match data.

The modelling contract here is stricter than in the e-commerce tasks, because a
football result row is mostly made of things that are only knowable *after* the
whistle. Shots, corners, cards and half-time scores all describe the match we
are trying to predict, so none of them may be used as features for that match.

Everything in :func:`build_match_features` is therefore derived from a single
chronological pass in which each match is featurised from the state accumulated
by *earlier* matches only, and the state is updated afterwards. Two tasks come
out of it:

``sports-match-result`` (classification)
    Home win / draw / away win — the 1X2 market.

``sports-total-goals`` (regression)
    Total goals scored, which is what the over/under 2.5 market prices.

Where the canonical football-data.co.uk source is reachable the rows also carry
Bet365 closing odds. Those are never features; they are passed through to the
betting evaluation as the benchmark to beat.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .base import TabularTask
from .sources import (
    ODDS_COLUMNS,
    OVER_UNDER_COLUMNS,
    DatasetUnavailableError,
    fetch,
    football_season,
)

LOGGER = logging.getLogger(__name__)

#: Seasons in football-data.co.uk's four-digit form, oldest first.
DEFAULT_SEASONS = ("1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324")
DEFAULT_LEAGUES = ("premier-league",)

#: Per-match statistics we roll forward as form. Kept to columns that both the
#: canonical source and the mirror provide.
_FORM_STATS = {
    "shots": ("HS", "AS"),
    "shots_on_target": ("HST", "AST"),
    "corners": ("HC", "AC"),
}

_ELO_START = 1500.0
_ELO_K = 20.0
_ELO_HOME_ADVANTAGE = 60.0
_FORM_WINDOW = 5
_MIN_HISTORY = 5


def _parse_dates(raw: pd.Series) -> pd.Series:
    """Parse both date conventions in use across the sources.

    football-data.co.uk writes ``11/08/2023`` (day first); the datahub mirror
    writes ``2023-08-11``. Trying ISO first and filling the gaps with day-first
    parsing handles a file of either convention, and a mixture of both.
    """
    parsed = pd.to_datetime(raw, format="ISO8601", errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(raw, dayfirst=True, errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


def load_football(
    leagues: tuple[str, ...] = DEFAULT_LEAGUES,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Fetch and concatenate the requested league-seasons, oldest match first."""
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for league in leagues:
        for season in seasons:
            remote = football_season(league, season)
            try:
                path = fetch(remote, cache_dir)
            except DatasetUnavailableError as exc:
                failures.append(f"{league} {season}: {exc.args[0].splitlines()[0]}")
                continue

            frame = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            frame = frame.dropna(subset=["HomeTeam", "AwayTeam"])
            if frame.empty:
                continue
            frame["league"] = league
            frame["season"] = season
            frames.append(frame)

    if not frames:
        raise DatasetUnavailableError(
            "No football seasons could be loaded.\n" + "\n".join(failures)
        )
    if failures:
        LOGGER.warning("Skipped %d unavailable league-seasons", len(failures))

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["Date"] = _parse_dates(combined["Date"])
    combined = combined.dropna(subset=["Date", "FTHG", "FTAG", "FTR"])

    # A stable chronological order is what makes the single-pass featuriser
    # correct, so sort by date and keep the original order within a date.
    combined = combined.sort_values("Date", kind="mergesort").reset_index(drop=True)
    return combined


@dataclass
class _TeamState:
    """Rolling record for one team, holding only completed matches."""

    elo: float = _ELO_START
    last_played: pd.Timestamp | None = None
    played: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=_FORM_WINDOW))
    recent_home: deque = field(default_factory=lambda: deque(maxlen=_FORM_WINDOW))
    recent_away: deque = field(default_factory=lambda: deque(maxlen=_FORM_WINDOW))

    def form(self, window: deque, prefix: str) -> dict[str, float]:
        """Mean of the buffered past matches, NaN when there is no history."""
        if not window:
            keys = ["points", "goals_for", "goals_against", *_FORM_STATS]
            return {f"{prefix}_{k}": np.nan for k in keys}
        stacked: dict[str, float] = {}
        for key in ["points", "goals_for", "goals_against", *_FORM_STATS]:
            values = [entry[key] for entry in window if not np.isnan(entry[key])]
            stacked[f"{prefix}_{key}"] = float(np.mean(values)) if values else np.nan
        return stacked


def _elo_update(
    home_elo: float, away_elo: float, home_goals: int, away_goals: int
) -> tuple[float, float, float]:
    """Return updated ratings plus the pre-match home win expectancy."""
    expected_home = 1.0 / (1.0 + 10 ** ((away_elo - home_elo - _ELO_HOME_ADVANTAGE) / 400.0))
    if home_goals > away_goals:
        actual = 1.0
    elif home_goals == away_goals:
        actual = 0.5
    else:
        actual = 0.0

    # Widen the update for emphatic wins, the standard margin-of-victory
    # correction used by football Elo implementations.
    margin = abs(home_goals - away_goals)
    if margin <= 1:
        multiplier = 1.0
    elif margin == 2:
        multiplier = 1.5
    else:
        multiplier = (11.0 + margin) / 8.0

    delta = _ELO_K * multiplier * (actual - expected_home)
    return home_elo + delta, away_elo - delta, expected_home


def _side_stats(match, side: str) -> dict[str, float]:
    """Per-match statistics for one side, missing values preserved as NaN.

    Only ever called on a match that has already finished, to record it as
    history for future fixtures.
    """
    out: dict[str, float] = {}
    for name, (home_col, away_col) in _FORM_STATS.items():
        column = home_col if side == "home" else away_col
        value = getattr(match, column, np.nan)
        out[name] = float(value) if pd.notna(value) else np.nan
    return out


def build_match_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-match features to every row of a chronologically sorted frame.

    For each match the features are computed from state that only contains
    earlier matches; the state is updated with this match afterwards. Rows whose
    teams do not yet have ``_MIN_HISTORY`` completed matches are dropped rather
    than imputed, so the model is never trained on fabricated form.
    """
    states: dict[str, _TeamState] = defaultdict(_TeamState)
    head_to_head: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=_FORM_WINDOW))

    rows: list[dict[str, float]] = []

    for match in frame.itertuples(index=False):
        home, away = match.HomeTeam, match.AwayTeam
        home_state, away_state = states[home], states[away]

        # ---- features: strictly pre-match -------------------------------
        home_elo, away_elo = home_state.elo, away_state.elo
        expected_home = 1.0 / (
            1.0 + 10 ** ((away_elo - home_elo - _ELO_HOME_ADVANTAGE) / 400.0)
        )

        record: dict[str, float] = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "elo_expectancy_home": expected_home,
            "home_matches_played": home_state.played,
            "away_matches_played": away_state.played,
        }
        record.update(home_state.form(home_state.recent, "home_form"))
        record.update(away_state.form(away_state.recent, "away_form"))
        # Venue-specific form: home teams at home, away teams away.
        record.update(home_state.form(home_state.recent_home, "home_venue"))
        record.update(away_state.form(away_state.recent_away, "away_venue"))

        for side, state in (("home", home_state), ("away", away_state)):
            if state.last_played is None:
                record[f"{side}_rest_days"] = np.nan
            else:
                record[f"{side}_rest_days"] = float((match.Date - state.last_played).days)

        h2h = head_to_head[(home, away)]
        record["h2h_home_points"] = float(np.mean(h2h)) if h2h else np.nan
        record["h2h_matches"] = float(len(h2h))

        record["is_valid"] = (
            home_state.played >= _MIN_HISTORY and away_state.played >= _MIN_HISTORY
        )
        rows.append(record)

        # ---- state update: this match is now history --------------------
        home_goals, away_goals = int(match.FTHG), int(match.FTAG)
        new_home_elo, new_away_elo, _ = _elo_update(
            home_elo, away_elo, home_goals, away_goals
        )
        home_state.elo, away_state.elo = new_home_elo, new_away_elo

        if home_goals > away_goals:
            home_points, away_points = 3.0, 0.0
        elif home_goals == away_goals:
            home_points, away_points = 1.0, 1.0
        else:
            home_points, away_points = 0.0, 3.0

        home_state.recent.append(
            {"points": home_points, "goals_for": float(home_goals),
             "goals_against": float(away_goals), **_side_stats(match, "home")}
        )
        away_state.recent.append(
            {"points": away_points, "goals_for": float(away_goals),
             "goals_against": float(home_goals), **_side_stats(match, "away")}
        )
        home_state.recent_home.append(home_state.recent[-1])
        away_state.recent_away.append(away_state.recent[-1])

        home_state.played += 1
        away_state.played += 1
        home_state.last_played = match.Date
        away_state.last_played = match.Date
        head_to_head[(home, away)].append(home_points)

    features = pd.DataFrame(rows, index=frame.index)
    return pd.concat([frame, features], axis=1)


def _prepare(
    leagues: tuple[str, ...],
    seasons: tuple[str, ...],
    cache_dir: Path | None,
) -> pd.DataFrame:
    enriched = build_match_features(load_football(leagues, seasons, cache_dir))
    usable = enriched[enriched["is_valid"]].drop(columns=["is_valid"]).copy()
    if usable.empty:
        raise ValueError("No matches left after requiring team history; add more seasons.")
    usable["total_goals"] = usable["FTHG"].astype(int) + usable["FTAG"].astype(int)
    return usable.reset_index(drop=True)


def _feature_names(frame: pd.DataFrame) -> list[str]:
    """Everything the single-pass featuriser produced, and nothing else.

    Deriving the list from the engineered columns rather than by subtracting a
    denylist from the raw frame means a new post-match column appearing upstream
    can never silently become a feature.
    """
    engineered = [
        c
        for c in frame.columns
        if c.startswith(
            ("home_elo", "away_elo", "elo_", "home_form", "away_form",
             "home_venue", "away_venue", "h2h_")
        )
        or c in {"home_rest_days", "away_rest_days",
                 "home_matches_played", "away_matches_played"}
    ]
    return engineered + ["league"]


def _temporal_split(frame: pd.DataFrame, holdout_seasons: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier seasons, test on the most recent ones.

    A random split would let a model learn from matches played after the ones it
    is scored on, which is exactly the mistake that makes backtests look
    profitable and live betting lose money.
    """
    ordered = sorted(frame["season"].unique())
    if len(ordered) <= holdout_seasons:
        raise ValueError(
            f"Need more than {holdout_seasons} seasons to hold out {holdout_seasons}."
        )
    test_seasons = set(ordered[-holdout_seasons:])
    is_test = frame["season"].isin(test_seasons)
    return frame[~is_test], frame[is_test]


def _odds_frame(rows: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame | None:
    """Bookmaker odds aligned to ``rows``, when the source provided them.

    The datahub mirror ships match statistics only, so this returns ``None``
    there and the betting simulation is skipped rather than run on invented
    prices.
    """
    if not all(column in rows.columns for column in columns):
        return None
    odds = rows[list(columns)].apply(pd.to_numeric, errors="coerce")
    if odds.isna().all().all():
        return None
    return odds.reset_index(drop=True)


def _build(
    *,
    name: str,
    task_type: str,
    target: str,
    description: str,
    leagues: tuple[str, ...],
    seasons: tuple[str, ...],
    cache_dir: Path | None,
    holdout_seasons: int,
) -> TabularTask:
    frame = _prepare(leagues, seasons, cache_dir)
    features = _feature_names(frame)
    train, test = _temporal_split(frame, holdout_seasons)

    extras: dict[str, object] = {
        "test_dates": test["Date"].reset_index(drop=True),
        "test_fixtures": test[["HomeTeam", "AwayTeam"]].reset_index(drop=True),
        "seasons_train": sorted(train["season"].unique()),
        "seasons_test": sorted(test["season"].unique()),
    }
    result_odds = _odds_frame(test, ODDS_COLUMNS)
    if result_odds is not None:
        extras["odds"] = result_odds
    over_under = _odds_frame(test, OVER_UNDER_COLUMNS)
    if over_under is not None:
        extras["over_under_odds"] = over_under
    if result_odds is None and over_under is None:
        LOGGER.info(
            "Bookmaker odds unavailable for %s; betting ROI analysis will be skipped.",
            name,
        )

    y_train = train[target]
    y_test = test[target]
    if task_type == "regression":
        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

    return TabularTask(
        name=name,
        industry="sports-betting",
        task_type=task_type,  # type: ignore[arg-type]
        target=target,
        X_train=train[features].reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_test=test[features].reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        categorical_features=["league"],
        description=description,
        extras=extras,
    )


def match_result_task(
    leagues: tuple[str, ...] = DEFAULT_LEAGUES,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    cache_dir: Path | None = None,
    *,
    holdout_seasons: int = 2,
) -> TabularTask:
    """Predict the 1X2 outcome (home win / draw / away win)."""
    return _build(
        name="sports-match-result",
        task_type="classification",
        target="FTR",
        description=(
            "Football 1X2 result from pre-match features only (Elo, rolling form, "
            "venue form, rest days, head-to-head). Split temporally: earlier "
            "seasons train, most recent seasons test."
        ),
        leagues=leagues,
        seasons=seasons,
        cache_dir=cache_dir,
        holdout_seasons=holdout_seasons,
    )


def total_goals_task(
    leagues: tuple[str, ...] = DEFAULT_LEAGUES,
    seasons: tuple[str, ...] = DEFAULT_SEASONS,
    cache_dir: Path | None = None,
    *,
    holdout_seasons: int = 2,
) -> TabularTask:
    """Predict total goals in the match, the quantity the over/under market prices."""
    return _build(
        name="sports-total-goals",
        task_type="regression",
        target="total_goals",
        description=(
            "Total goals scored in a football match, from pre-match features only. "
            "Drives the over/under 2.5 market. Temporal train/test split."
        ),
        leagues=leagues,
        seasons=seasons,
        cache_dir=cache_dir,
        holdout_seasons=holdout_seasons,
    )


def build_tasks(cache_dir: Path | None = None, **kwargs) -> list[TabularTask]:
    return [match_result_task(cache_dir=cache_dir, **kwargs),
            total_goals_task(cache_dir=cache_dir, **kwargs)]
