"""Turn predicted probabilities into a betting record.

A model is only useful to a bettor if it disagrees with the bookmaker in the
right direction often enough to cover the margin. These helpers make that
explicit: strip the overround out of the quoted odds to recover the market's
true probabilities, look for outcomes where the model's probability implies
positive expected value, and settle those bets against the real results.

The bookmaker's own implied probabilities are scored with the same metrics as
the model, so every table has the benchmark alongside the candidate. Beating it
is difficult and a negative return is the expected outcome; the point of the
simulation is to measure the gap honestly rather than to advertise a system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: 1X2 result label -> football-data.co.uk Bet365 column.
RESULT_ODDS_COLUMNS = {"H": "B365H", "D": "B365D", "A": "B365A"}

#: Over/under 2.5 goals market.
OVER_UNDER_ODDS_COLUMNS = {"over": "B365>2.5", "under": "B365<2.5"}


def implied_probabilities(odds: np.ndarray, *, remove_margin: bool = True) -> np.ndarray:
    """Convert decimal odds to probabilities.

    Raw reciprocals sum to more than one — the bookmaker's margin, or overround.
    Normalising removes it proportionally, which recovers a genuine probability
    distribution and is the standard way to compare a market against a model.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / np.asarray(odds, dtype=float)
    raw = np.where(np.isfinite(raw), raw, np.nan)
    if not remove_margin:
        return raw
    totals = np.nansum(raw, axis=1, keepdims=True)
    return np.divide(raw, totals, out=np.full_like(raw, np.nan), where=totals > 0)


def overround(odds: np.ndarray) -> np.ndarray:
    """Per-row bookmaker margin, e.g. 1.05 means a 5% overround."""
    return np.nansum(implied_probabilities(odds, remove_margin=False), axis=1)


@dataclass
class BettingResult:
    """Outcome of one staking simulation."""

    strategy: str
    n_opportunities: int
    n_bets: int
    total_staked: float
    profit: float
    hit_rate: float
    average_odds: float
    max_drawdown: float
    bankroll_curve: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def roi(self) -> float:
        """Profit per unit staked. Positive means the strategy made money."""
        return float(self.profit / self.total_staked) if self.total_staked > 0 else 0.0

    @property
    def bet_rate(self) -> float:
        return float(self.n_bets / self.n_opportunities) if self.n_opportunities else 0.0

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "strategy": self.strategy,
            "n_bets": self.n_bets,
            "bet_rate": round(self.bet_rate, 4),
            "total_staked": round(self.total_staked, 2),
            "profit": round(self.profit, 2),
            "roi": round(self.roi, 4),
            "hit_rate": round(self.hit_rate, 4),
            "average_odds": round(self.average_odds, 3),
            "max_drawdown": round(self.max_drawdown, 2),
        }


def _max_drawdown(curve: np.ndarray) -> float:
    """Largest peak-to-trough fall in cumulative profit."""
    if curve.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(curve)
    return float(np.max(running_peak - curve))


def simulate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    odds: np.ndarray,
    classes: np.ndarray,
    *,
    edge_threshold: float = 0.05,
    strategy: str = "flat",
    kelly_fraction: float = 0.25,
    bankroll: float = 1000.0,
    max_stake_fraction: float = 0.05,
) -> BettingResult:
    """Back the single best value bet per event, then settle it.

    ``edge_threshold`` is the minimum expected profit per unit staked required
    before a bet is placed: ``p_model * decimal_odds - 1``. Requiring a margin
    rather than any positive edge guards against acting on noise, since a model
    only fractionally more confident than the market is usually just wrong.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    odds = np.asarray(odds, dtype=float)
    classes = np.asarray(classes)

    if probabilities.shape != odds.shape:
        raise ValueError(
            f"probabilities {probabilities.shape} and odds {odds.shape} must align"
        )

    # Expected value per unit stake for every outcome of every event.
    edges = probabilities * odds - 1.0
    edges = np.where(np.isfinite(edges), edges, -np.inf)

    profits: list[float] = []
    wins = 0
    taken_odds: list[float] = []
    staked_total = 0.0
    current_bankroll = bankroll

    for row in range(len(y_true)):
        best = int(np.argmax(edges[row]))
        edge = edges[row, best]
        if not np.isfinite(edge) or edge < edge_threshold:
            continue

        price = odds[row, best]
        probability = probabilities[row, best]
        if not np.isfinite(price) or price <= 1.0:
            continue

        if strategy == "kelly":
            # f* = (p*b - q) / b for decimal odds b = price - 1.
            b = price - 1.0
            fraction = (probability * b - (1.0 - probability)) / b
            fraction = max(0.0, fraction) * kelly_fraction
            stake = min(fraction, max_stake_fraction) * current_bankroll
        else:
            stake = 1.0

        if stake <= 0:
            continue

        won = classes[best] == y_true[row]
        profit = stake * (price - 1.0) if won else -stake

        staked_total += stake
        current_bankroll += profit
        profits.append(profit)
        taken_odds.append(price)
        wins += int(won)

    curve = np.cumsum(profits) if profits else np.array([])
    return BettingResult(
        strategy=f"{strategy} (edge>{edge_threshold:.0%})",
        n_opportunities=len(y_true),
        n_bets=len(profits),
        total_staked=staked_total,
        profit=float(np.sum(profits)) if profits else 0.0,
        hit_rate=wins / len(profits) if profits else 0.0,
        average_odds=float(np.mean(taken_odds)) if taken_odds else 0.0,
        max_drawdown=_max_drawdown(curve),
        bankroll_curve=curve,
    )


def poisson_over_probability(expected_goals: np.ndarray, line: float = 2.5) -> np.ndarray:
    """P(total goals > ``line``) for a Poisson total with the given mean.

    Converts the regression's point estimate into the probability the over/under
    market actually prices. Football scorelines are close enough to Poisson for
    this to be the standard first approximation; it ignores the mild
    overdispersion of real match totals.
    """
    from scipy.stats import poisson

    expected = np.clip(np.asarray(expected_goals, dtype=float), 1e-6, None)
    # A .5 line means "strictly more than floor(line)" — no push is possible.
    return 1.0 - poisson.cdf(np.floor(line), expected)


def market_reference(
    y_true: np.ndarray, odds: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    """Score the bookmaker's own prices, to sit beside the model's metrics."""
    from .metrics import classification_metrics

    market = implied_probabilities(odds)
    usable = ~np.isnan(market).any(axis=1)
    if not usable.any():
        return {}

    market = market[usable]
    truth = np.asarray(y_true)[usable]
    predicted = np.asarray(classes)[market.argmax(axis=1)]

    scores = classification_metrics(truth, predicted, market, np.asarray(classes))
    scores["mean_overround"] = float(np.nanmean(overround(np.asarray(odds)[usable])))
    scores["n_matches"] = int(usable.sum())
    return scores
