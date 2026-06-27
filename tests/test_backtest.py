"""Tests for the M5 backtester: costs, metrics, and the engine.

The headline test is the deterministic scripted scenario for `simulate`
(test_simulate_scripted_*): hand-built prices and weights with a hand-computed
answer. It pins the no-look-ahead timing — the single most important property of
the whole backtester.
"""

import numpy as np
import pandas as pd
import pytest

from fintech.backtest.costs import apply_costs
from fintech.backtest.engine import (
    buy_and_hold_weights,
    equal_weight_weights,
    generate_mv_weights,
    simulate,
)
from fintech.backtest.metrics import (
    cagr,
    max_drawdown,
    sharpe,
    sortino,
    turnover,
)


# --------------------------------------------------------------------------- #
# costs
# --------------------------------------------------------------------------- #
def test_apply_costs_value():
    # 6 bps total on turnover 2.0 -> 0.0006 * 2 = 0.0012.
    assert apply_costs(2.0, commission_bps=1.0, slippage_bps=5.0) == pytest.approx(0.0012)


def test_apply_costs_zero_turnover():
    assert apply_costs(0.0, commission_bps=1.0, slippage_bps=5.0) == 0.0


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_cagr_constant_one_year():
    r = pd.Series([0.01] * 252)
    # Exactly one year of data -> CAGR is just the total compounded growth.
    assert cagr(r) == pytest.approx(1.01 ** 252 - 1)


def test_max_drawdown_hand_computed():
    # equity: 1 -> 1.5 -> 0.75; peak 1.5; worst dd = 0.75/1.5 - 1 = -0.5.
    r = pd.Series([0.5, -0.5])
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_is_zero():
    assert max_drawdown(pd.Series([0.01, 0.02, 0.03])) == pytest.approx(0.0)


def test_sharpe_zero_mean_is_zero():
    assert sharpe(pd.Series([0.01, -0.01, 0.01, -0.01])) == pytest.approx(0.0)


def test_sortino_hand_computed():
    r = pd.Series([0.02, -0.01, 0.03, -0.02])
    # mean excess = 0.005; downside dev = sqrt((0.01^2 + 0.02^2)/4) = sqrt(0.000125)
    expected = 0.005 / np.sqrt(0.000125) * np.sqrt(252)
    assert sortino(r) == pytest.approx(expected)


def test_turnover_excludes_undefined_first_row():
    # [1,0] -> [0,1] -> [0,1]: real per-period changes are 2 then 0 -> mean 1.0.
    w = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], columns=["A", "B"])
    assert turnover(w) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# engine — simulate (the scripted no-look-ahead scenario)
# --------------------------------------------------------------------------- #
def _scripted():
    dates = pd.date_range("2021-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {"A": [0.10, 0.05, -0.10], "B": [0.50, 0.20, 0.10]}, index=dates
    )
    weights = pd.DataFrame(
        {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=dates
    )
    return returns, weights, dates


def test_simulate_scripted_no_costs():
    returns, weights, dates = _scripted()
    net = simulate(weights, returns, commission_bps=0.0, slippage_bps=0.0)

    # t0: held cash -> 0.
    # t1: held A (set at t0), earns A's t1 return 0.05 — NOT B's 0.20. This is the
    #     no-look-ahead check: rebalancing INTO B at t1 must not earn B's t1 move.
    # t2: held B (set at t1), earns B's t2 return 0.10.
    expected = pd.Series([0.0, 0.05, 0.10], index=dates)
    pd.testing.assert_series_equal(net, expected)


def test_simulate_scripted_with_costs():
    returns, weights, dates = _scripted()
    # 10 bps total = 0.001 per unit turnover.
    net = simulate(weights, returns, commission_bps=10.0, slippage_bps=0.0)

    # turnover: t0 = 1 (buy A from cash), t1 = 2 (flip A->B), t2 = 0.
    expected = pd.Series([0.0 - 0.001, 0.05 - 0.002, 0.10 - 0.0], index=dates)
    pd.testing.assert_series_equal(net, expected)


# --------------------------------------------------------------------------- #
# engine — generate_mv_weights
# --------------------------------------------------------------------------- #
def _panel(n=80, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = ["AAPL", "MSFT", "SPY"]
    returns = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=dates, columns=cols)
    mu = pd.DataFrame(rng.normal(0.001, 0.002, (n, 3)), index=dates, columns=cols)
    return returns, mu


def test_mv_weights_shape_and_constraints():
    returns, mu = _panel()
    w = generate_mv_weights(
        returns, mu, lookback=30, rebalance_every=5,
        risk_aversion=5.0, max_weight=0.6, cash_floor=0.1,
    )
    assert list(w.columns) == list(returns.columns)
    assert w.index.isin(returns.index).all()           # rebalance dates are real
    assert (w.to_numpy() >= -1e-6).all()                # long-only
    assert (w.to_numpy() <= 0.6 + 1e-6).all()           # per-asset cap
    assert (w.sum(axis=1) <= 1 - 0.1 + 1e-6).all()      # cash floor


def test_mv_weights_are_causal():
    returns, mu = _panel()
    kwargs = dict(lookback=30, rebalance_every=5, risk_aversion=5.0)
    base = generate_mv_weights(returns, mu, **kwargs)

    # Shock the LAST return. Any rebalance whose trailing window ends before that
    # date must be byte-for-byte unchanged — covariance can't see the future.
    tampered = returns.copy()
    tampered.iloc[-1] *= 5
    shocked = generate_mv_weights(tampered, mu, **kwargs)

    last = returns.index[-1]
    mask = base.index < last
    pd.testing.assert_frame_equal(base[mask], shocked[mask])


# --------------------------------------------------------------------------- #
# engine — baselines + the "costs bite" success criterion
# --------------------------------------------------------------------------- #
def test_baseline_weight_shapes():
    returns, _ = _panel()
    eq = equal_weight_weights(returns)
    bnh = buy_and_hold_weights(returns, "SPY")

    assert eq.shape == (1, 3) and np.allclose(eq.to_numpy(), 1 / 3)
    assert bnh.shape == (1, 3)
    assert bnh["SPY"].iloc[0] == 1.0 and bnh.drop(columns="SPY").to_numpy().sum() == 0.0


def test_costs_make_zero_skill_signal_lose_to_buy_and_hold():
    # §M5 success criterion: a zero-skill signal that churns daily, net of costs,
    # must underperform buy-and-hold. Random mu = no edge; daily rebalancing on it
    # racks up turnover; with real costs the drag sinks it below holding SPY.
    returns, mu = _panel(n=200, seed=7)

    mv = generate_mv_weights(returns, mu, lookback=30, rebalance_every=1, risk_aversion=1.0)
    start = mv.index[0]

    costs = dict(commission_bps=50.0, slippage_bps=50.0)  # heavy, so costs clearly bite
    strat_net = simulate(mv, returns, **costs).loc[start:]
    bnh = buy_and_hold_weights(returns, "SPY", start=start)
    bnh_net = simulate(bnh, returns, **costs).loc[start:]

    strat_total = (1 + strat_net).prod()
    bnh_total = (1 + bnh_net).prod()
    assert strat_total < bnh_total
