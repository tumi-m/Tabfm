"""Tests for the betting maths.

Settlement arithmetic is checked against hand-computed payouts: an off-by-one in
the profit formula (returning stake x odds instead of stake x (odds - 1)) would
otherwise show up as a plausible-looking positive ROI.
"""

from __future__ import annotations

import numpy as np
import pytest

from tabfm_lab.evaluation.betting import (
    implied_probabilities,
    market_reference,
    overround,
    poisson_over_probability,
    simulate,
)

CLASSES = np.array(["H", "D", "A"])


def test_implied_probabilities_sum_to_one_after_margin_removal() -> None:
    odds = np.array([[2.0, 3.5, 4.0], [1.5, 4.5, 6.0]])
    probabilities = implied_probabilities(odds)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities > 0).all()


def test_overround_detects_the_bookmaker_margin() -> None:
    fair = np.array([[2.0, 4.0, 4.0]])  # reciprocals sum to exactly 1
    assert overround(fair)[0] == pytest.approx(1.0)

    juiced = np.array([[1.9, 3.8, 3.8]])
    assert overround(juiced)[0] > 1.0


def test_no_bets_placed_without_sufficient_edge() -> None:
    """A model that agrees with the market should stake nothing."""
    odds = np.array([[2.0, 4.0, 4.0]] * 10)
    probabilities = np.tile(implied_probabilities(odds)[0], (10, 1))
    truth = np.array(["H"] * 10)

    result = simulate(truth, probabilities, odds, CLASSES, edge_threshold=0.05)
    assert result.n_bets == 0
    assert result.roi == 0.0
    assert result.total_staked == 0.0


def test_flat_staking_settles_a_winner_correctly() -> None:
    """One confident, correct bet at 4.0 returns 3 units of profit on 1 staked."""
    odds = np.array([[2.0, 4.0, 4.0]])
    probabilities = np.array([[0.10, 0.80, 0.10]])  # edge on D: 0.8*4 - 1 = 2.2
    truth = np.array(["D"])

    result = simulate(truth, probabilities, odds, CLASSES, edge_threshold=0.05)
    assert result.n_bets == 1
    assert result.total_staked == pytest.approx(1.0)
    assert result.profit == pytest.approx(3.0)
    assert result.roi == pytest.approx(3.0)
    assert result.hit_rate == pytest.approx(1.0)


def test_flat_staking_settles_a_loser_correctly() -> None:
    odds = np.array([[2.0, 4.0, 4.0]])
    probabilities = np.array([[0.10, 0.80, 0.10]])
    truth = np.array(["H"])

    result = simulate(truth, probabilities, odds, CLASSES, edge_threshold=0.05)
    assert result.n_bets == 1
    assert result.profit == pytest.approx(-1.0)
    assert result.roi == pytest.approx(-1.0)
    assert result.hit_rate == pytest.approx(0.0)


def test_kelly_scales_stake_with_bankroll_and_respects_the_cap() -> None:
    odds = np.array([[2.0, 4.0, 4.0]])
    probabilities = np.array([[0.10, 0.80, 0.10]])
    truth = np.array(["D"])

    result = simulate(
        truth, probabilities, odds, CLASSES,
        strategy="kelly", bankroll=1000.0, max_stake_fraction=0.05,
    )
    # Full Kelly here is large, so the 5% cap binds: 5% of 1,000 = 50.
    assert result.total_staked == pytest.approx(50.0)
    assert result.profit == pytest.approx(150.0)


def test_max_drawdown_tracks_the_worst_peak_to_trough() -> None:
    odds = np.array([[10.0, 10.0, 10.0]] * 4)
    probabilities = np.array([[0.9, 0.05, 0.05]] * 4)
    truth = np.array(["H", "A", "A", "A"])  # win, then three losses

    result = simulate(truth, probabilities, odds, CLASSES)
    assert result.n_bets == 4
    # Cumulative profit peaks at +9 then falls to +6, +5, +4 -> drawdown of 3.
    assert result.max_drawdown == pytest.approx(3.0)


def test_simulate_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="must align"):
        simulate(
            np.array(["H"]),
            np.array([[0.5, 0.5]]),
            np.array([[2.0, 3.0, 4.0]]),
            CLASSES,
        )


def test_poisson_over_probability_is_monotonic_and_bounded() -> None:
    means = np.array([0.5, 1.5, 2.5, 3.5, 5.0])
    probabilities = poisson_over_probability(means, line=2.5)

    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.all(np.diff(probabilities) > 0), "more expected goals must mean more overs"


def test_poisson_over_probability_matches_closed_form() -> None:
    from scipy.stats import poisson

    expected = 1.0 - poisson.cdf(2, 2.7)
    assert poisson_over_probability(np.array([2.7]), 2.5)[0] == pytest.approx(expected)


def test_market_reference_scores_the_bookmaker() -> None:
    odds = np.array([[1.5, 4.0, 7.0], [3.0, 3.4, 2.4], [2.1, 3.3, 3.6]])
    truth = np.array(["H", "A", "H"])

    reference = market_reference(truth, odds, CLASSES)
    assert reference["n_matches"] == 3
    assert reference["mean_overround"] > 1.0
    assert 0.0 <= reference["accuracy"] <= 1.0
    assert reference["log_loss"] > 0.0
