"""Risk-aversion (lambda) calibration for the composite MV strategy.

Why: with the current lambda=10 the optimiser collapses to a corner solution —
~10 names pinned at the max_weight cap (an implicit "top-10 equal weight"). Raising
lambda de-concentrates the book off the cap into confidence-graded weights. But
diversification cuts variance at the cost of more turnover (more names) and a
diluted exposure to the strongest signals, so the winner is an EMPIRICAL,
net-of-cost question, not a "weights look nicer" one.

This sweeps lambda on the exact composite signal + cost model used in
backtest_compare.py and prints net-of-cost metrics side by side, plus how
concentrated the resulting book is (avg #names, avg max weight, avg invested).

    .venv\\Scripts\\python.exe scripts\\calibrate_lambda.py
"""

import pandas as pd

from fintech.backtest import metrics
from fintech.backtest.engine import generate_mv_weights, simulate
from scripts.backtest_compare import (
    COMMISSION_BPS,
    IMPACT_BPS,
    LOOKBACK,
    REBALANCE_EVERY,
    SLIPPAGE_BPS,
    _composite_mu,
    _wide_returns,
)

BARS_PATH = "data/processed/bars.parquet"
LAMBDAS = [10, 50, 100, 300, 1000]
MAX_WEIGHT = 0.10


def _book_shape(w):
    """Average concentration of a weights path over its rebalance rows."""
    rows = w[(w.abs() > 1e-6).any(axis=1)]
    held = (rows > 1e-6).sum(axis=1)
    return {
        "AvgNames": held.mean(),
        "AvgMaxW": rows.max(axis=1).mean(),
        "AvgInvested": rows.sum(axis=1).mean(),
    }


def main():
    bars = pd.read_parquet(BARS_PATH)
    returns = _wide_returns(bars)
    costs = dict(commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS,
                 impact_bps=IMPACT_BPS)
    mu = _composite_mu(bars, returns.columns)

    paths = {}
    for lam in LAMBDAS:
        paths[lam] = generate_mv_weights(
            returns, mu, lookback=LOOKBACK, rebalance_every=REBALANCE_EVERY,
            risk_aversion=lam, max_weight=MAX_WEIGHT, cash_floor=0.0,
        )
    start = max(w.index[0] for w in paths.values())

    rows = {}
    for lam, w in paths.items():
        net = simulate(w, returns, **costs).loc[start:]
        rows[f"lambda={lam}"] = {
            "CAGR": metrics.cagr(net),
            "Sharpe": metrics.sharpe(net),
            "Sortino": metrics.sortino(net),
            "MaxDD": metrics.max_drawdown(net),
            "Turnover": metrics.turnover(w),
            **_book_shape(w),
        }

    table = pd.DataFrame(rows).T
    print(f"\nComposite MV, lambda sweep — same signal/costs as backtest_compare")
    print(f"Eval window: {start.date()} -> {returns.index[-1].date()} "
          f"({len(returns.loc[start:])} days), "
          f"costs={COMMISSION_BPS + SLIPPAGE_BPS:.0f}bps + {IMPACT_BPS:.0f}bps*turnover^2, "
          f"max_weight={MAX_WEIGHT}, rebal={REBALANCE_EVERY}d\n")
    with pd.option_context("display.float_format", lambda x: f"{x:0.4f}",
                           "display.width", 200, "display.max_columns", None):
        print(table)
    print()


if __name__ == "__main__":
    main()
