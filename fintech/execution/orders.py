"""Translate target weights into concrete orders (M6).

The PURE, testable heart of execution: given target portfolio weights, what you
currently hold, current prices, and account value, compute the trades needed to
get from here to there. No broker, no network — so it can be unit-tested exactly,
the same way `simulate` isolates the look-ahead-sensitive logic in M5.

The broker layer (broker.py) handles the I/O; this just does the arithmetic.
"""

import pandas as pd


def compute_orders(target_weights, positions, prices, equity, min_notional=1.0):
    """
    Orders (signed share deltas) to move the book to `target_weights`.

    For each name in the union of targets and current holdings:
      target value  = target_weight * equity
      current value = shares_held * price
      trade         = (target value - current value) / price   (+ buy / - sell)
    Names dropped from the target get fully sold (target weight 0). Trades whose
    notional is below `min_notional` are skipped to avoid dust orders.

    Args:
        target_weights (pd.Series): desired weight per ticker (from the optimizer).
        positions (pd.Series): current shares held per ticker (missing = 0).
        prices (pd.Series): current price per ticker.
        equity (float): total account value (cash + positions) to size against.
        min_notional (float): skip trades smaller than this dollar amount.

    Returns:
        pd.Series: signed shares to trade per ticker (+ buy, - sell); only names
            with a non-trivial trade.
    """
    # Union of targets and current holdings: a held name dropped from the target
    # has weight 0 here and so gets fully sold.
    tickers = target_weights.index.union(positions.index)
    target_w = target_weights.reindex(tickers).fillna(0.0)
    held = positions.reindex(tickers).fillna(0.0)
    px = prices.reindex(tickers)

    trade_value = target_w * equity - held * px  # + buy, - sell
    shares = trade_value / px

    # Drop names with no price, and dust trades below min_notional.
    keep = px.notna() & (trade_value.abs() >= min_notional)
    orders = shares[keep]
    return orders[orders != 0]
