"""End-to-end M5/M7 backtest on the wide (S&P 500) universe.

Runs the whole vertical slice on the real processed bars, with a per-date
dynamic universe (ragged listing dates handled in generate_mv_weights):
  bars -> wide returns + signal(s) (mu) -> mean-variance weights -> costs
       -> net returns -> metrics, side by side, all on the same dates.

Compares two MV strategies (historical-mean mu vs short-term-reversal mu)
against equal-weight and buy-and-hold SPY. Reporting script, not a test.
    .venv\\Scripts\\python.exe scripts\\backtest_compare.py
"""

import pandas as pd

from fintech.features.build import build_features
from fintech.models.baselines import predict_historical_mean
from fintech.backtest.engine import (
    buy_and_hold_weights,
    generate_mv_weights,
    simulate,
)
from fintech.backtest import metrics

BARS_PATH = "data/processed/bars.parquet"
BENCHMARK = "SPY"

LOOKBACK = 60
REBALANCE_EVERY = 21          # ~monthly
# Realistic-ish costs: commission + a half-spread, plus convex market impact so
# heavy churn pays for itself (S&P 500 names, not microcaps).
COMMISSION_BPS = 1.0
SLIPPAGE_BPS = 9.0
IMPACT_BPS = 10.0
SIGNAL_SCALE = 0.01           # maps a 1-sigma composite signal to ~1% expected return
OPT_KWARGS = dict(risk_aversion=10.0, max_weight=0.10, max_turnover=None, cash_floor=0.0)


def _wide_returns(bars):
    """Realised daily returns, date x ticker (NaN before a name lists)."""
    px = bars.pivot(index="date", columns="ticker", values="adj_close")
    return px.pct_change()


def _hist_mean_mu(bars, columns):
    """Historical-mean signal pivoted to wide date x ticker."""
    features = build_features(bars)
    frame = features[["date", "ticker"]].copy()
    frame["mu"] = predict_historical_mean(features).to_numpy()
    return frame.pivot(index="date", columns="ticker", values="mu").reindex(columns=columns)


def _zscore_by_date(frame, col):
    """Cross-sectional z-score of `col` within each date."""
    g = frame.groupby("date")[col]
    return (frame[col] - g.transform("mean")) / g.transform("std")


def _composite_mu(bars, columns):
    """Composite factor signal: long 12-1 momentum + short-term reversal.

    Each factor is standardised cross-sectionally per day, averaged, then scaled
    to return-like units so the optimiser's risk term stays meaningful.
    """
    f = build_features(bars)
    z_mom = _zscore_by_date(f, "mom_252_21")   # high momentum -> buy
    z_rev = -_zscore_by_date(f, "ret_lag1")    # low recent return -> buy (reversal)
    f = f[["date", "ticker"]].copy()
    f["mu"] = 0.5 * (z_mom + z_rev) * SIGNAL_SCALE
    return f.pivot(index="date", columns="ticker", values="mu").reindex(columns=columns)


def _summary(net, w=None):
    out = {
        "CAGR": metrics.cagr(net),
        "Sharpe": metrics.sharpe(net),
        "Sortino": metrics.sortino(net),
        "MaxDD": metrics.max_drawdown(net),
    }
    out["Turnover"] = metrics.turnover(w) if (w is not None and len(w) > 1) else 0.0
    return out


def main():
    bars = pd.read_parquet(BARS_PATH)
    returns = _wide_returns(bars)
    costs = dict(commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
                 impact_bps=IMPACT_BPS)

    # Two signals: historical mean (M5 baseline) and the momentum+reversal composite.
    mu_hist = _hist_mean_mu(bars, returns.columns)
    mu_comp = _composite_mu(bars, returns.columns)

    mv_hist = generate_mv_weights(returns, mu_hist, lookback=LOOKBACK,
                                  rebalance_every=REBALANCE_EVERY, **OPT_KWARGS)
    mv_comp = generate_mv_weights(returns, mu_comp, lookback=LOOKBACK,
                                  rebalance_every=REBALANCE_EVERY, **OPT_KWARGS)

    start = max(mv_hist.index[0], mv_comp.index[0])

    # Equal-weight over names tradable at the start (non-NaN return that day).
    tradable = returns.loc[start].dropna().index
    eq_w = pd.DataFrame(
        [[1.0 / len(tradable)] * len(tradable)], index=[start], columns=tradable
    ).reindex(columns=returns.columns, fill_value=0.0)

    paths = {
        "MV composite": mv_comp,
        "MV hist-mean": mv_hist,
        "Equal-weight": eq_w,
        f"Buy&Hold {BENCHMARK}": buy_and_hold_weights(returns, BENCHMARK, start=start),
    }

    rows = {}
    for name, w in paths.items():
        net = simulate(w, returns, **costs).loc[start:]
        rows[name] = _summary(net, w)

    table = pd.DataFrame(rows).T
    n_names = returns.columns.size
    print(f"\nUniverse: {n_names} tickers (S&P 500 + {BENCHMARK})")
    print(f"Eval window: {start.date()} -> {returns.index[-1].date()} "
          f"({len(returns.loc[start:])} days), "
          f"costs={COMMISSION_BPS + SLIPPAGE_BPS:.0f}bps + {IMPACT_BPS:.0f}bps*turnover^2, "
          f"lookback={LOOKBACK}, rebal={REBALANCE_EVERY}d, max_weight={OPT_KWARGS['max_weight']}\n")
    with pd.option_context("display.float_format", lambda x: f"{x:0.4f}"):
        print(table)
    print()


if __name__ == "__main__":
    main()
