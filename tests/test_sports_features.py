"""Guards on the sports featuriser.

The headline test is :func:`test_features_ignore_future_matches`. Leakage in a
betting model is not a subtle accuracy bug — it produces a backtest that looks
profitable and a strategy that loses money — and it is easy to reintroduce by
adding one innocuous-looking rolling feature. Pinning the property directly is
cheaper than re-auditing the featuriser every time it changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabfm_lab.data.sports import (
    _FORM_STATS,
    _MIN_HISTORY,
    _feature_names,
    _temporal_split,
    build_match_features,
)

TEAMS = [f"Team {chr(ord('A') + i)}" for i in range(8)]


def synthetic_matches(n_rounds: int = 12, seed: int = 7) -> pd.DataFrame:
    """A deterministic fake league where every team meets every other."""
    rng = np.random.default_rng(seed)
    rows = []
    date = pd.Timestamp("2020-08-01")
    for round_index in range(n_rounds):
        shuffled = list(TEAMS)
        rng.shuffle(shuffled)
        for home, away in zip(shuffled[::2], shuffled[1::2], strict=True):
            home_goals = int(rng.integers(0, 4))
            away_goals = int(rng.integers(0, 4))
            rows.append(
                {
                    "Date": date + pd.Timedelta(days=7 * round_index),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": home_goals,
                    "FTAG": away_goals,
                    "FTR": "H" if home_goals > away_goals else ("D" if home_goals == away_goals else "A"),
                    "HS": int(rng.integers(5, 20)),
                    "AS": int(rng.integers(5, 20)),
                    "HST": int(rng.integers(1, 9)),
                    "AST": int(rng.integers(1, 9)),
                    "HC": int(rng.integers(0, 12)),
                    "AC": int(rng.integers(0, 12)),
                    "league": "test-league",
                    "season": f"S{round_index // 6}",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def matches() -> pd.DataFrame:
    return synthetic_matches()


def _engineered_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in _feature_names(frame) if c != "league"]


def test_features_ignore_future_matches(matches: pd.DataFrame) -> None:
    """Rewriting the last match must not change any earlier row's features."""
    baseline = build_match_features(matches)

    mutated = matches.copy()
    last = mutated.index[-1]
    mutated.loc[last, ["FTHG", "FTAG", "FTR", "HS", "HST"]] = [9, 0, "H", 40, 25]
    perturbed = build_match_features(mutated)

    columns = _engineered_columns(baseline)
    pd.testing.assert_frame_equal(
        baseline.iloc[:-1][columns],
        perturbed.iloc[:-1][columns],
        check_dtype=False,
    )


def test_first_match_for_a_team_has_no_form(matches: pd.DataFrame) -> None:
    enriched = build_match_features(matches)
    first = enriched.iloc[0]

    assert first["home_matches_played"] == 0
    assert first["away_matches_played"] == 0
    assert np.isnan(first["home_form_points"])
    assert np.isnan(first["away_form_points"])
    assert np.isnan(first["home_rest_days"])
    assert not first["is_valid"]


def test_elo_starts_level_and_moves_after_results(matches: pd.DataFrame) -> None:
    enriched = build_match_features(matches)
    assert enriched.iloc[0]["home_elo"] == pytest.approx(1500.0)
    assert enriched.iloc[0]["elo_diff"] == pytest.approx(0.0)
    # Later matches must show ratings that have actually diverged.
    assert enriched["elo_diff"].abs().max() > 1.0


def test_elo_is_zero_sum(matches: pd.DataFrame) -> None:
    """Rating is transferred between the two teams, never created."""
    enriched = build_match_features(matches)
    total = enriched["home_elo"].iloc[0] + enriched["away_elo"].iloc[0]
    assert total == pytest.approx(3000.0)


def test_validity_flag_requires_history_for_both_teams(matches: pd.DataFrame) -> None:
    enriched = build_match_features(matches)
    valid = enriched[enriched["is_valid"]]
    assert (valid["home_matches_played"] >= _MIN_HISTORY).all()
    assert (valid["away_matches_played"] >= _MIN_HISTORY).all()
    assert len(valid) > 0, "fixture should produce some usable rows"


def test_feature_list_excludes_post_match_columns(matches: pd.DataFrame) -> None:
    """No column describing the match being predicted may become a feature."""
    enriched = build_match_features(matches)
    features = set(_feature_names(enriched))

    post_match = {"FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee", "total_goals"}
    post_match.update(column for pair in _FORM_STATS.values() for column in pair)

    assert not (features & post_match), f"post-match columns leaked: {features & post_match}"


def test_form_uses_only_completed_matches(matches: pd.DataFrame) -> None:
    """Form for a team's Nth match must equal the mean over its previous matches."""
    enriched = build_match_features(matches)
    team = TEAMS[0]

    appearances = enriched[
        (enriched["HomeTeam"] == team) | (enriched["AwayTeam"] == team)
    ]
    target_row = appearances.iloc[3]
    previous = appearances.iloc[:3]

    expected_points = []
    for match in previous.itertuples():
        if match.HomeTeam == team:
            scored, conceded = match.FTHG, match.FTAG
        else:
            scored, conceded = match.FTAG, match.FTHG
        expected_points.append(3.0 if scored > conceded else (1.0 if scored == conceded else 0.0))

    prefix = "home_form" if target_row["HomeTeam"] == team else "away_form"
    assert target_row[f"{prefix}_points"] == pytest.approx(np.mean(expected_points))


def test_temporal_split_keeps_test_seasons_after_train(matches: pd.DataFrame) -> None:
    enriched = build_match_features(matches)
    usable = enriched[enriched["is_valid"]]
    train, test = _temporal_split(usable, holdout_seasons=1)

    assert len(train) > 0 and len(test) > 0
    assert set(train["season"]).isdisjoint(set(test["season"]))
    assert max(train["Date"]) <= min(test["Date"])


def test_temporal_split_rejects_impossible_holdout(matches: pd.DataFrame) -> None:
    enriched = build_match_features(matches)
    with pytest.raises(ValueError, match="Need more than"):
        _temporal_split(enriched, holdout_seasons=99)
