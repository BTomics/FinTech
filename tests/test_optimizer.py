"""Tests for fintech.portfolio.optimizer.

estimate_covariance wraps Ledoit-Wolf shrinkage. The properties that matter for a
mean-variance optimiser downstream: the matrix keeps its asset labels/order, is
symmetric, and is positive-definite (well-conditioned) EVEN when the raw sample
covariance would be singular — which is the whole reason for shrinkage.
"""

import numpy as np
import pandas as pd

from fintech.portfolio.optimizer import estimate_covariance, optimize_weights

TOL = 1e-6  # solvers return ~1e-9 boundary violations, not exact equality


def _returns(rows, cols, seed=0):
    """Synthetic wide returns frame: `rows` dates x named asset columns."""
    rng = np.random.default_rng(seed)
    data = rng.normal(scale=0.01, size=(rows, len(cols)))
    return pd.DataFrame(data, columns=cols)


def test_covariance_shape_and_labels():
    returns = _returns(250, ["AAPL", "MSFT", "SPY"])
    cov = estimate_covariance(returns)

    # Square, and labelled with the SAME tickers in the SAME order as the input —
    # this is what keeps mu / cov / weights aligned downstream.
    assert cov.shape == (3, 3)
    assert list(cov.index) == ["AAPL", "MSFT", "SPY"]
    assert list(cov.columns) == ["AAPL", "MSFT", "SPY"]
    # A covariance matrix must be symmetric.
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)


def test_covariance_is_pd_even_when_sample_is_singular():
    # 4 observations, 6 assets -> the raw sample covariance is rank-deficient
    # (singular): MV optimisation on it would blow up. Shrinkage is exactly what
    # rescues this regime, so the shrunk matrix must come out positive-definite.
    returns = _returns(4, [f"A{i}" for i in range(6)])

    raw = np.cov(returns.to_numpy(), rowvar=False)
    assert np.linalg.matrix_rank(raw) < returns.shape[1]  # sample cov is singular

    cov = estimate_covariance(returns)
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
    assert eigenvalues.min() > 0  # shrinkage made it positive-definite


def test_covariance_deterministic():
    returns = _returns(120, ["AAPL", "MSFT", "SPY", "NVDA"])
    pd.testing.assert_frame_equal(
        estimate_covariance(returns), estimate_covariance(returns)
    )


def _mu_cov(tickers=("AAPL", "MSFT", "SPY")):
    tickers = list(tickers)
    # Descending expected returns so the optimiser has a clear preference order.
    mu = pd.Series(np.linspace(0.02, 0.005, len(tickers)), index=tickers)
    cov = estimate_covariance(_returns(250, tickers))
    return mu, cov


def test_weights_respect_every_constraint():
    # M4 success criterion: a returned weight vector must breach NO constraint.
    # Set every cap tight enough to bind, then check all four hold.
    mu, cov = _mu_cov()
    w_prev = pd.Series([0.3, 0.3, 0.0], index=mu.index)
    max_weight, max_turnover, cash_floor = 0.5, 0.4, 0.1

    w = optimize_weights(
        mu, cov, w_prev=w_prev, risk_aversion=5.0,
        max_weight=max_weight, max_turnover=max_turnover, cash_floor=cash_floor,
    )

    assert list(w.index) == list(mu.index)         # alignment preserved
    assert (w >= -TOL).all()                        # long-only
    assert (w <= max_weight + TOL).all()            # per-asset cap
    assert w.sum() <= 1 - cash_floor + TOL          # cash floor
    assert np.abs(w - w_prev).sum() <= max_turnover + TOL  # turnover


def test_weights_deterministic():
    mu, cov = _mu_cov()
    pd.testing.assert_series_equal(
        optimize_weights(mu, cov), optimize_weights(mu, cov)
    )


def test_cov_order_does_not_change_weights():
    # cov given in a different ticker order than mu must yield the SAME weights —
    # proves the optimiser realigns cov to mu's order internally.
    mu, cov = _mu_cov()
    shuffled = ["SPY", "AAPL", "MSFT"]
    cov_shuffled = cov.loc[shuffled, shuffled]
    pd.testing.assert_series_equal(
        optimize_weights(mu, cov, risk_aversion=5.0),
        optimize_weights(mu, cov_shuffled, risk_aversion=5.0),
    )


def test_higher_expected_return_gets_more_weight():
    # Equal variance, no binding cap -> the only driver is mu, so the higher-mu
    # asset must get more weight (sanity that the objective points the right way).
    tickers = ["LOW", "HIGH"]
    mu = pd.Series([0.001, 0.02], index=tickers)
    cov = pd.DataFrame(np.eye(2) * 1e-4, index=tickers, columns=tickers)
    w = optimize_weights(mu, cov, risk_aversion=1.0)
    assert w["HIGH"] > w["LOW"]
