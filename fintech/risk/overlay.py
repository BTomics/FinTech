"""Risk overlay (M9): exposure control on TOP of the MV weights.

The composite+MV book picks good *relative* weights but rides a ~-45% drawdown
(λ calibration proved risk_aversion can't fix that — high λ only de-risks via a
cash-drag confound). This module adds a dedicated exposure overlay: the MV
optimizer still chooses the relative weights; the overlay only chooses *how much*
total to hold (0..k_max gross, the rest in cash).

These are PURE functions (no I/O, no state) so they're testable in isolation and
reused unchanged in both the backtest and live `paper_trade.py` — same code path,
no drift. Start with volatility targeting; CPPI / VaR are later stretch rows.

See C:\\Users\\20244357\\.claude\\plans\\risk-overlay-m9.md.

NOTE (skeleton): bodies are intentionally unimplemented — implement until
tests/test_risk_overlay.py is green.
"""

import numpy as np
import pandas as pd

from fintech.backtest.metrics import PERIODS_PER_YEAR


def realized_portfolio_vol(weights, cov, periods_per_year=PERIODS_PER_YEAR):
    """
    Annualised volatility of the book `weights` under covariance `cov`.

    The vol-targeting input: how risky the *current* book is, given the trailing
    covariance you already estimate with `estimate_covariance`. This is the
    portfolio standard deviation sqrt(wᵀ Σ w), annualised from per-period (daily)
    units by sqrt(periods_per_year).

    CONTRACT (assert in tests/test_risk_overlay.py):
      - `cov` is a DAILY covariance matrix (DataFrame, ticker-labelled like
        estimate_covariance output); `weights` a Series on (a subset of) the same
        tickers. Align on `weights.index` — never trust pandas auto-align after
        .to_numpy() (same discipline as optimize_weights).
      - Returns a non-negative float in annualised return units (e.g. 0.18 = 18%).
      - A single asset with weight 1.0 and daily variance v -> sqrt(v)*sqrt(ppy).

    Args:
        weights (pd.Series): book weights, indexed by ticker.
        cov (pd.DataFrame): daily covariance, ticker x ticker.
        periods_per_year (int): annualisation factor.

    Returns:
        float: annualised portfolio volatility.
    """
    # weights.index is canonical: realign cov to the book (cov may carry extra
    # names; a name in weights missing from cov is a real error -> KeyError).
    idx = weights.index
    sub_cov = cov.loc[idx, idx].to_numpy()
    w = weights.to_numpy()

    daily_variance = w @ sub_cov @ w
    return float(np.sqrt(daily_variance) * np.sqrt(periods_per_year))


def vol_target_scale(realized_vol, sigma_target, k_max=1.0):
    """
    Gross-exposure scale factor for volatility targeting.

    k = clip(sigma_target / realized_vol, 0, k_max). When the book is calmer than
    target we'd want to lever up — but k_max=1.0 (long-only, no leverage) caps
    that at fully invested; when vol spikes (crashes cluster in high-vol regimes)
    k shrinks automatically, de-risking the book.

    CONTRACT (assert in tests/test_risk_overlay.py):
      - realized_vol < sigma_target  -> ratio > 1, clipped to k_max (== 1.0 default).
      - realized_vol == 2*sigma_target -> k == 0.5 (halve exposure).
      - 0 <= k <= k_max always (never negative, never above the cap).
      - realized_vol <= 0 (degenerate / no risk estimate) -> k == k_max (fully
        invested; there's no vol to target down to).

    Args:
        realized_vol (float): annualised realised portfolio vol (>= 0).
        sigma_target (float): target annualised vol (e.g. 0.12).
        k_max (float): max gross exposure (1.0 = no leverage).

    Returns:
        float: scale factor k in [0, k_max].
    """
    # TODO: guard realized_vol <= 0 -> return k_max;
    #       return float(np.clip(sigma_target / realized_vol, 0.0, k_max)).
    if realized_vol <= 0:
        return float(k_max)

    scale = sigma_target / realized_vol
    return float(np.clip(scale, 0.0, k_max))


def apply_vol_target(weights, realized_vol, sigma_target, k_max=1.0):
    """
    Scale a weight vector to the volatility target, leaving the rest in cash.

    Final book = k · weights with k from `vol_target_scale`. RELATIVE proportions
    are unchanged (the MV optimizer's job); only the gross stays/shrinks. The
    implied cash weight is 1 - k·Σw (earns 0 unless a risk-free rate is modelled).

    CONTRACT (assert in tests/test_risk_overlay.py):
      - returns a Series on the same index as `weights`.
      - scaled == k * weights elementwise (proportions preserved).
      - k below 1 reduces gross exposure (sum of scaled < sum of original).

    Args:
        weights (pd.Series): MV target weights, indexed by ticker.
        realized_vol (float): annualised realised portfolio vol.
        sigma_target (float): target annualised vol.
        k_max (float): max gross exposure.

    Returns:
        pd.Series: vol-targeted weights, same index as `weights`.
    """
    # TODO: k = vol_target_scale(realized_vol, sigma_target, k_max); return k * weights.
    
    # 1) Get scale factor
    k = vol_target_scale(realized_vol, sigma_target, k_max)
    
    # 2) Scale the MV weights; return a new Series on the same index.
    return k * weights


# --- stretch mechanisms (see plan) -------------------------------------------

def cppi_exposure(value, floor, multiplier, k_max=1.0):
    """
    Constant-Proportion Portfolio Insurance gross-exposure fraction.

    Enforces a hard portfolio floor `F`: hold a risky fraction proportional to the
    *cushion* (how far above the floor you are). cushion = value - floor; risky
    dollars = multiplier · cushion; fraction = that / value, clamped to [0, k_max].
    As value falls toward the floor the cushion shrinks → exposure auto-cuts to 0
    at the floor, so (ignoring gap risk between rebalances) value can't breach F.

    CONTRACT (assert in tests/test_risk_overlay.py):
      - value <= floor -> 0.0 (fully de-risked, protect the floor).
      - value > floor  -> multiplier * (value - floor) / value, clipped to k_max.
      - 0 <= fraction <= k_max always.

    Args:
        value (float): current portfolio value.
        floor (float): hard floor to protect (same units as value).
        multiplier (float): CPPI multiplier m (cushion leverage, e.g. 3-5).
        k_max (float): max gross exposure (1.0 = no leverage).

    Returns:
        float: risky-asset exposure fraction in [0, k_max].
    """
    if value <= floor:
        return 0.0
    cushion = value - floor
    fraction = multiplier * cushion / value
    return float(np.clip(fraction, 0.0, k_max))


def historical_var(returns, alpha=0.95):
    """
    Historical Value-at-Risk at confidence `alpha`, as a positive loss.

    The loss the book is not expected to exceed with probability `alpha` over one
    period — the empirical (1-alpha) quantile of the return distribution, sign-
    flipped so a loss is a positive number (VaR 0.03 = "a 3% loss is the 95%
    worst case"). Non-parametric: just reads the historical tail, no distribution
    assumption.

    CONTRACT (assert in tests/test_risk_overlay.py):
      - equals -quantile(returns, 1 - alpha).
      - positive for a loss-bearing series; higher alpha -> deeper (>=) VaR.

    Args:
        returns (pd.Series | np.ndarray): per-period returns.
        alpha (float): confidence level (e.g. 0.95).

    Returns:
        float: VaR as a positive loss magnitude.
    """
    return float(-np.quantile(returns, 1.0 - alpha))


def expected_shortfall(returns, alpha=0.95):
    """
    Expected Shortfall (CVaR) at confidence `alpha`, as a positive loss.

    The *average* loss in the worst (1-alpha) tail — i.e. given that you breach
    VaR, how bad is it on average. Coherent risk measure (unlike VaR), so it sees
    fat tails VaR misses. Mean of returns at or below the (1-alpha) quantile,
    sign-flipped to a positive loss.

    CONTRACT (assert in tests/test_risk_overlay.py):
      - >= historical_var at the same alpha (tail mean is worse than the cutoff).
      - positive for a loss-bearing series.

    Args:
        returns (pd.Series | np.ndarray): per-period returns.
        alpha (float): confidence level (e.g. 0.95).

    Returns:
        float: expected shortfall as a positive loss magnitude.
    """
    r = np.asarray(returns, dtype=float)
    threshold = np.quantile(r, 1.0 - alpha)
    tail = r[r <= threshold]
    return float(-tail.mean())
