"""End-to-end M5 backtest: MV strategy vs equal-weight vs buy-and-hold SPY.

Runs the whole vertical slice on the real processed bars:
  bars -> features -> baseline signal (mu) -> mean-variance weights -> costs
       -> net returns -> metrics, side by side, all on the same dates.

This is a reporting script, not a test. Run it from the repo root:
    .venv\\Scripts\\python.exe scripts\\backtest_compare.py
"""

import pandas as pd

from fintech.features.build import build_features
from fintech.models.baselines import predict_historical_mean
from fintech.backtest.engine import (
    buy_and_hold_weights,
    equal_weight_weights,
    generate_mv_weights,
    simulate,
)
from fintech.backtest import metrics

BARS_PATH = "data/processed/bars.parquet"
BENCHMARK = "SPY"

# Backtest config.
LOOKBACK = 60          # trading days of history for the covariance estimate
REBALANCE_EVERY = 21   # ~monthly, to keep turnover (and cost drag) sane
COMMISSION_BPS = 1.0
SLIPPAGE_BPS = 5.0
OPT_KWARGS = dict(risk_aversion=10.0, max_weight=0.6, max_turnover=None, cash_floor=0.0)


def _wide_returns(bars):
    """Realised daily returns, date x ticker, from adjusted close."""
    px = bars.pivot(index="date", columns="ticker", values="adj_close")
    return px.pct_change().dropna()


def _wide_mu(bars):
    """Baseline expected-return signal (historical mean), date x ticker."""
    features = build_features(bars)
    pred = predict_historical_mean(features)
    frame = features[["date", "ticker"]].copy()
    frame["mu"] = pred.to_numpy()
    return frame.pivot(index="date", columns="ticker", values="mu")


def _summary(net):
    return {
        "CAGR": metrics.cagr(net),
        "Sharpe": metrics.sharpe(net),
        "Sortino": metrics.sortino(net),
        "MaxDD": metrics.max_drawdown(net),
    }


def main():
    bars = pd.read_parquet(BARS_PATH)

    returns = _wide_returns(bars)
    mu = _wide_mu(bars)

    # Same universe, same dates for every strategy.
    common = returns.index.intersection(mu.index)
    returns = returns.loc[common]
    mu = mu.loc[common, returns.columns]

    costs = dict(commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS)

    mv_w = generate_mv_weights(returns, mu, lookback=LOOKBACK,
                               rebalance_every=REBALANCE_EVERY, **OPT_KWARGS)
    start = mv_w.index[0]  # common evaluation start (after warmup)

    paths = {
        "MV strategy": mv_w,
        "Equal-weight": equal_weight_weights(returns, start=start),
        f"Buy&Hold {BENCHMARK}": buy_and_hold_weights(returns, BENCHMARK, start=start),
    }

    rows = {}
    for name, w in paths.items():
        net = simulate(w, returns, **costs).loc[start:]
        rows[name] = _summary(net)
        rows[name]["Turnover"] = metrics.turnover(w) if len(w) > 1 else 0.0

    table = pd.DataFrame(rows).T
    print(f"\nUniverse: {list(returns.columns)}")
    print(f"Eval window: {start.date()} -> {returns.index[-1].date()} "
          f"({len(returns.loc[start:])} days), costs={COMMISSION_BPS + SLIPPAGE_BPS:.0f} bps\n")
    with pd.option_context("display.float_format", lambda x: f"{x:0.4f}"):
        print(table)
    print()


if __name__ == "__main__":
    main()
