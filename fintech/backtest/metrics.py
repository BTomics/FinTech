"""Performance metrics for backtests (M5).

Pure functions over a daily return series (and, for turnover, a weights frame).
These are how the strategy is judged against buy-and-hold SPY and equal-weight —
all on the same dates, net of costs (development_plan M5).

Convention: `returns` is a pd.Series of per-period simple returns (one row per
trading day). `PERIODS_PER_YEAR` annualises daily numbers.
"""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252  # trading days


def cagr(returns, periods_per_year=PERIODS_PER_YEAR):
    """
    Compound annual growth rate implied by a return series.

    The single annual rate that, compounded over the period, reproduces the same
    total growth — so two strategies of different lengths are comparable.

    Args:
        returns (pd.Series): per-period simple returns.
        periods_per_year (int): periods per year for annualisation.

    Returns:
        float: annualised compound growth rate (e.g. 0.12 == 12%/yr).
    """
    n = len(returns)
    total_growth = np.prod(1 + returns) - 1
    annualized = (1 + total_growth) ** (periods_per_year / n) - 1
    return annualized




def sharpe(returns, periods_per_year=PERIODS_PER_YEAR, risk_free=0.0):
    """
    Annualised Sharpe ratio: excess return per unit of total volatility.

    Reward-to-risk. Higher = more return for the wobble taken on. Penalises ALL
    volatility, up and down alike.

    Args:
        returns (pd.Series): per-period simple returns.
        periods_per_year (int): for annualising mean and std.
        risk_free (float): per-period risk-free rate to subtract.

    Returns:
        float: annualised Sharpe ratio.
    """
    excess_returns = returns - risk_free
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    annualized_sharpe = (mean_excess_return / std_excess_return) * np.sqrt(periods_per_year)
    return annualized_sharpe


def sortino(returns, periods_per_year=PERIODS_PER_YEAR, risk_free=0.0):
    """
    Annualised Sortino ratio: like Sharpe, but only DOWNSIDE volatility.

    Upside swings aren't "risk" to an investor — Sortino divides by the deviation
    of losing periods only, so a strategy isn't punished for big gains.

    Args:
        returns (pd.Series): per-period simple returns.
        periods_per_year (int): for annualisation.
        risk_free (float): per-period risk-free rate to subtract.

    Returns:
        float: annualised Sortino ratio.
    """
    excess_returns = returns - risk_free
    # Downside deviation: RMS of only the losing periods, but averaged over ALL
    # periods (positive periods contribute 0, not dropped).
    downside = excess_returns.clip(upper=0.0)
    downside_dev = np.sqrt((downside ** 2).mean())
    # Numerator is the mean of ALL excess returns (same as Sharpe) — only the
    # denominator restricts to the downside.
    return (excess_returns.mean() / downside_dev) * np.sqrt(periods_per_year)


def max_drawdown(returns):
    """
    Largest peak-to-trough decline of the equity curve (a negative number).

    The worst loss an investor riding this curve would have lived through — the
    headline measure of pain / risk of ruin.

    Args:
        returns (pd.Series): per-period simple returns.

    Returns:
        float: most negative (equity / running_peak - 1), e.g. -0.35 == -35%.
    """
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1
    return drawdowns.min()


def turnover(weights):
    """
    Average one-period turnover across a sequence of target weight vectors.

    How much the book churns each rebalance — the driver of trading cost. One
    period's turnover is sum(|w_t - w_{t-1}|); this averages it over the path.

    Args:
        weights (pd.DataFrame): one row per rebalance date, one column per asset.

    Returns:
        float: mean per-period sum of absolute weight changes.
    """
    weight_changes = weights.diff().abs().sum(axis=1)
    # diff()'s first row is undefined (no prior weights) — drop it, don't average
    # a spurious zero in.
    return weight_changes.iloc[1:].mean()
