"""Contract tests for the M9 risk overlay (volatility targeting) + Calmar.

Written RED first (skeleton): fintech/risk/overlay.py and metrics.calmar raise
NotImplementedError. Implement until these pass. See
C:\\Users\\20244357\\.claude\\plans\\risk-overlay-m9.md.

The overlay is a pure exposure control on top of the MV weights: it never
changes relative proportions, only how much total gross to hold (rest in cash).
"""

import numpy as np
import pandas as pd
import pytest

from fintech.backtest.metrics import calmar, cagr, max_drawdown
from fintech.risk.overlay import (
    apply_vol_target,
    cppi_exposure,
    expected_shortfall,
    historical_var,
    realized_portfolio_vol,
    vol_target_scale,
)

TOL = 1e-9


# --- vol_target_scale: the clip(sigma_target / realized, 0, k_max) core -------

def test_scale_is_one_when_vol_below_target():
    # Calmer than target -> would lever, but long-only cap keeps k at k_max=1.0.
    k = vol_target_scale(realized_vol=0.06, sigma_target=0.12, k_max=1.0)
    assert k == pytest.approx(1.0, abs=TOL)


def test_scale_halves_when_vol_is_double_target():
    k = vol_target_scale(realized_vol=0.24, sigma_target=0.12, k_max=1.0)
    assert k == pytest.approx(0.5, abs=TOL)


def test_scale_never_exceeds_k_max_and_never_negative():
    # Near-zero vol -> ratio explodes -> clipped to k_max.
    assert vol_target_scale(0.001, 0.12, k_max=1.0) == pytest.approx(1.0, abs=TOL)
    # Allowing leverage: cap binds at the higher k_max.
    assert vol_target_scale(0.001, 0.12, k_max=1.5) == pytest.approx(1.5, abs=TOL)
    # Huge vol -> tiny but non-negative k.
    k = vol_target_scale(10.0, 0.12, k_max=1.0)
    assert 0.0 <= k <= 1.0


def test_scale_handles_degenerate_zero_vol():
    # No usable risk estimate -> fully invested (nothing to target down to), not a
    # divide-by-zero blowup.
    assert vol_target_scale(0.0, 0.12, k_max=1.0) == pytest.approx(1.0, abs=TOL)


# --- apply_vol_target: scales the book, preserves proportions -----------------

def test_apply_preserves_proportions_and_index():
    w = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
    scaled = apply_vol_target(w, realized_vol=0.24, sigma_target=0.12, k_max=1.0)
    assert list(scaled.index) == list(w.index)
    # k = 0.5 -> every weight halved, ratios unchanged.
    pd.testing.assert_series_equal(scaled, w * 0.5)
    assert scaled.sum() < w.sum()   # gross exposure reduced, rest is cash


def test_apply_is_identity_when_calm():
    w = pd.Series({"AAPL": 0.6, "MSFT": 0.4})
    scaled = apply_vol_target(w, realized_vol=0.05, sigma_target=0.12, k_max=1.0)
    pd.testing.assert_series_equal(scaled, w)   # k == 1.0


# --- realized_portfolio_vol: sqrt(wᵀΣw), annualised ---------------------------

def test_realized_vol_single_asset_annualises():
    daily_var = 0.0004                      # daily std 0.02
    cov = pd.DataFrame([[daily_var]], index=["AAPL"], columns=["AAPL"])
    w = pd.Series({"AAPL": 1.0})
    vol = realized_portfolio_vol(w, cov)
    assert vol == pytest.approx(np.sqrt(daily_var) * np.sqrt(252), rel=1e-6)


def test_realized_vol_aligns_cov_to_weights_order():
    # Shuffled cov columns must not change the answer (no pandas auto-align traps).
    cov = pd.DataFrame(
        [[4e-4, 1e-4], [1e-4, 9e-4]], index=["AAPL", "MSFT"], columns=["AAPL", "MSFT"]
    )
    w = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    base = realized_portfolio_vol(w, cov)
    shuffled = realized_portfolio_vol(w, cov.loc[["MSFT", "AAPL"], ["MSFT", "AAPL"]])
    assert base == pytest.approx(shuffled, rel=1e-9)


def test_realized_vol_ignores_extra_cov_names():
    # cov carries names NOT in the book (the backtest case: cov can span more
    # tickers than the nonzero weights). Those must be ignored, not NaN-poison.
    full_cov = pd.DataFrame(
        [[4e-4, 0.0, 0.0], [0.0, 9e-4, 0.0], [0.0, 0.0, 1e-3]],
        index=["AAPL", "MSFT", "TSLA"], columns=["AAPL", "MSFT", "TSLA"],
    )
    w = pd.Series({"AAPL": 0.5, "MSFT": 0.5})   # no TSLA exposure
    book_only = full_cov.loc[["AAPL", "MSFT"], ["AAPL", "MSFT"]]
    assert realized_portfolio_vol(w, full_cov) == pytest.approx(
        realized_portfolio_vol(w, book_only), rel=1e-12
    )
    assert np.isfinite(realized_portfolio_vol(w, full_cov))


# --- the point of the overlay: less drawdown on a high-vol path ---------------

def test_overlay_reduces_drawdown_on_high_vol_series():
    # A book P&L that de-risks (k<1) during the drawdown loses less than the
    # un-overlaid one. Modelled here as scaling the net return by k=0.5 — what a
    # halved gross exposure does to the day's P&L.
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(-0.001, 0.03, size=500))   # negative-drift, high vol
    overlaid = 0.5 * base
    assert abs(max_drawdown(overlaid)) <= abs(max_drawdown(base))


# --- calmar -------------------------------------------------------------------

def test_calmar_is_cagr_over_abs_maxdd():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, size=500))
    expected = cagr(r) / abs(max_drawdown(r))
    assert calmar(r) == pytest.approx(expected, rel=1e-9)


def test_calmar_no_drawdown_is_not_zero_division():
    r = pd.Series([0.01] * 100)            # monotonic up -> max_drawdown == 0
    val = calmar(r)
    assert np.isinf(val) or np.isnan(val)  # defined behaviour, not an exception


# --- stretch: CPPI floor ------------------------------------------------------

def test_cppi_zero_at_or_below_floor():
    assert cppi_exposure(value=90.0, floor=100.0, multiplier=4.0) == 0.0
    assert cppi_exposure(value=100.0, floor=100.0, multiplier=4.0) == 0.0


def test_cppi_is_multiplier_times_cushion_fraction():
    # value 120, floor 100, m=3 -> cushion 20 -> 3*20/120 = 0.5 exposure.
    assert cppi_exposure(120.0, 100.0, 3.0, k_max=1.0) == pytest.approx(0.5)


def test_cppi_clamped_to_k_max_and_nonnegative():
    # Big cushion * multiplier would exceed full investment -> clipped to k_max.
    assert cppi_exposure(200.0, 100.0, 5.0, k_max=1.0) == pytest.approx(1.0)
    assert 0.0 <= cppi_exposure(101.0, 100.0, 4.0, k_max=1.0) <= 1.0


# --- stretch: VaR / CVaR (positive = loss) ------------------------------------

def test_historical_var_is_negative_quantile():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0, 0.02, size=10_000))
    var = historical_var(r, alpha=0.95)
    assert var == pytest.approx(-np.quantile(r, 0.05), rel=1e-9)
    assert var > 0                                   # a real loss


def test_expected_shortfall_at_least_var():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(-0.0005, 0.02, size=10_000))
    es = expected_shortfall(r, alpha=0.95)
    var = historical_var(r, alpha=0.95)
    assert es >= var                                 # tail mean worse than cutoff
    assert es > 0


def test_var_deeper_at_higher_confidence():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0, 0.02, size=10_000))
    assert historical_var(r, alpha=0.99) >= historical_var(r, alpha=0.95)
