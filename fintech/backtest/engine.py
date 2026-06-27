"""Event-driven daily backtest loop (M5).

Chains the whole system: realized returns + a return signal -> target weights
(constrained mean-variance) -> costs -> a net daily return stream you can score
with metrics.py and compare against buy-and-hold SPY / equal-weight.

Two responsibilities, deliberately split:
  - generate_mv_weights: decide target weights over time, STRICTLY CAUSAL.
  - simulate: turn a weight path + realized returns into net returns, applying
    costs. This is where look-ahead bugs hide, so it's isolated and gets the
    deterministic scripted-scenario test.

THE GOLDEN RULE (no look-ahead): weights chosen using information available at
the close of day t may only earn the return realised from t to t+1. Never
multiply day-t weights by day-t's (already-known) return — that's the classic
backtest lie that invents free money.
"""

import pandas as pd

from fintech.backtest.costs import apply_costs
from fintech.portfolio.optimizer import estimate_covariance, optimize_weights


def simulate(weights, returns, commission_bps=1.0, slippage_bps=5.0):
    """
    Net per-period portfolio returns from a target-weight path.

    Timing contract: a row of `weights` dated t is the book held going INTO the
    next period, so it earns `returns` at t+1. The gross period return is the
    previously-held weights dotted with this period's realised asset returns;
    cost is charged when the book changes.

    Args:
        weights (pd.DataFrame): target weights, one row per date held, columns =
            assets. Held (carried forward) until the next row.
        returns (pd.DataFrame): realised per-period asset returns, same columns.
        commission_bps, slippage_bps (float): passed to apply_costs.

    Returns:
        pd.Series: net per-period portfolio returns, indexed by date.
    """
    weights = weights.reindex(columns=returns.columns).fillna(0.0)
    w_held = pd.Series(0.0, index=returns.columns)  # book carried into each day
    net_returns = pd.Series(0.0, index=returns.index)
    for date in returns.index:
        # Earn this period's return on the book held coming INTO the day — i.e.
        # weights decided on an earlier date, never this period's new target.
        gross = (w_held * returns.loc[date]).sum()
        cost = 0.0
        if date in weights.index:
            # Rebalance at the close: charge the trade, then the new book takes
            # effect for the NEXT period.
            w_target = weights.loc[date]
            turnover = (w_target - w_held).abs().sum()
            cost = apply_costs(turnover, commission_bps, slippage_bps)
            w_held = w_target
        net_returns.loc[date] = gross - cost
    return net_returns


def generate_mv_weights(returns, mu, lookback=60, rebalance_every=1, **opt_kwargs):
    """
    Walk forward producing constrained mean-variance target weights, causally.

    At each rebalance date t: estimate covariance from the trailing `lookback`
    returns ENDING AT t (past only), read the signal mu for date t, and solve
    optimize_weights against the previous book. Between rebalances the book is
    held.

    Args:
        returns (pd.DataFrame): realised per-period asset returns (date x asset).
        mu (pd.DataFrame): expected-return signal per (date, asset) — the M3
            prediction, one row per date aligned to `returns`' columns.
        lookback (int): trailing window length for estimate_covariance.
        rebalance_every (int): rebalance cadence in periods (1 = daily).
        **opt_kwargs: forwarded to optimize_weights (risk_aversion, max_weight,
            max_turnover, cash_floor).

    Returns:
        pd.DataFrame: target weights, one row per rebalance date, columns = assets.
    """
    dates = returns.index
    weights = {}
    w_prev = None
    # Need a full lookback window before the first rebalance, so start at the
    # date in position lookback-1 (its trailing window ends AT it, past only).
    for i in range(lookback - 1, len(dates)):
        if (i - (lookback - 1)) % rebalance_every != 0:
            continue  # hold between rebalances
        t = dates[i]
        window = returns.iloc[i - lookback + 1:i + 1]  # lookback rows ending AT t
        cov = estimate_covariance(window)
        mu_t = mu.loc[t]  # signal known at t — nothing after t feeds the optimiser
        w_prev = optimize_weights(mu_t, cov, w_prev=w_prev, **opt_kwargs)
        weights[t] = w_prev
    return pd.DataFrame(weights).T


def equal_weight_weights(returns, start=None):
    """
    Equal-weight book (1/N per asset) set once at `start` and held.

    A naive diversification benchmark the strategy must beat (development_plan
    M5). Fed through the same `simulate` so it pays the same costs.

    Args:
        returns (pd.DataFrame): realised per-period asset returns (for the asset
            universe and the default start date).
        start: date to buy in; defaults to the first return date.

    Returns:
        pd.DataFrame: a single-row weights frame (date x asset).
    """
    start = returns.index[0] if start is None else start
    n = returns.shape[1]
    return pd.DataFrame([[1.0 / n] * n], index=[start], columns=returns.columns)


def buy_and_hold_weights(returns, asset, start=None):
    """
    100% in one asset, set once at `start` and held.

    The buy-and-hold SPY benchmark — the bar every active strategy has to clear
    net of costs (development_plan M5).

    Args:
        returns (pd.DataFrame): realised per-period asset returns.
        asset: column to hold fully.
        start: date to buy in; defaults to the first return date.

    Returns:
        pd.DataFrame: a single-row weights frame (date x asset).
    """
    start = returns.index[0] if start is None else start
    w = pd.Series(0.0, index=returns.columns)
    w[asset] = 1.0
    return pd.DataFrame([w.to_numpy()], index=[start], columns=returns.columns)
