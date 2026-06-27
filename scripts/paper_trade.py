"""Daily paper-trading run (M6): composite strategy -> Alpaca paper orders.

Pipeline:
  bars -> today's composite signal (momentum + reversal) -> mean-variance target
       -> compare to live Alpaca positions -> market orders -> reconciliation log.

PAPER ONLY. Keys come from .env (ALPACA_API_KEY/SECRET/PAPER). Run from repo root:
    .venv\\Scripts\\python.exe scripts\\paper_trade.py

Note: build_features drops the most recent date (its next-day target is unknown),
so the *live* signal is computed straight from the price panel here — same factor
definitions as features/build.py, just keeping today's row.
"""

import pandas as pd
from dotenv import load_dotenv

from fintech.portfolio.optimizer import estimate_covariance, optimize_weights
from fintech.execution.broker import get_client, get_equity, get_positions, submit_orders
from fintech.execution.orders import compute_orders

BARS_PATH = "data/processed/bars.parquet"
LOOKBACK = 60
SIGNAL_SCALE = 0.01
OPT_KWARGS = dict(risk_aversion=10.0, max_weight=0.10, max_turnover=None, cash_floor=0.0)


def _zscore(s):
    return (s - s.mean()) / s.std()


def latest_target_weights(bars):
    """Composite-strategy target weights for the most recent bar date."""
    px = bars.pivot(index="date", columns="ticker", values="adj_close")
    # Drop trailing incomplete bars: a partial/failed snapshot can leave a
    # mostly-NaN row that would poison the cross-section. Keep only dates where
    # the broad universe is present.
    coverage = px.notna().mean(axis=1)
    px = px.loc[coverage > 0.9]
    returns = px.pct_change(fill_method=None)

    # Today's factor values (same defs as features/build.py), latest row only.
    mom = (px.shift(21) / px.shift(252) - 1).iloc[-1]   # 12-1 momentum
    rev = -returns.iloc[-1]                              # short-term reversal (-r_t)

    # Eligible = finite on both factors AND a full trailing covariance window.
    window = returns.iloc[-LOOKBACK:]
    eligible = window.columns[window.notna().all()]
    eligible = mom.reindex(eligible).dropna().index.intersection(rev.dropna().index)

    mu = 0.5 * (_zscore(mom[eligible]) + _zscore(rev[eligible])) * SIGNAL_SCALE
    cov = estimate_covariance(window[eligible])
    weights = optimize_weights(mu, cov, **OPT_KWARGS)
    prices = px.iloc[-1]  # latest adj_close ~= tradable price (back-adjusted)
    return weights, prices


def main():
    load_dotenv()
    bars = pd.read_parquet(BARS_PATH)
    target_weights, prices = latest_target_weights(bars)

    client = get_client()
    equity = get_equity(client)
    positions = get_positions(client)

    orders = compute_orders(target_weights, positions, prices, equity)
    print(f"Equity ${equity:,.0f} | {len(positions)} positions | "
          f"{(target_weights > 1e-6).sum()} target names | {len(orders)} orders")

    records = submit_orders(client, orders)
    if records:
        report = pd.DataFrame(records)
        print("\nReconciliation (intended vs submitted):")
        print(report.to_string(index=False))
    else:
        print("No orders to submit (already at target).")


if __name__ == "__main__":
    main()
