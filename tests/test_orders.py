"""Tests for fintech.execution.orders.compute_orders (pure — no broker/network)."""

import pandas as pd

from fintech.execution.orders import compute_orders


def test_clean_buy_in_from_cash():
    # $10k, target 50/50, no holdings, prices 100 & 50 -> buy 50 A, 100 B.
    target = pd.Series({"A": 0.5, "B": 0.5})
    positions = pd.Series(dtype=float)
    prices = pd.Series({"A": 100.0, "B": 50.0})
    orders = compute_orders(target, positions, prices, equity=10_000)
    assert orders["A"] == 50.0
    assert orders["B"] == 100.0


def test_rebalance_trades_only_the_delta():
    # Already hold 30 A (target wants 50) -> buy just 20 more, not 50.
    target = pd.Series({"A": 0.5, "B": 0.5})
    positions = pd.Series({"A": 30.0})
    prices = pd.Series({"A": 100.0, "B": 50.0})
    orders = compute_orders(target, positions, prices, equity=10_000)
    assert orders["A"] == 20.0
    assert orders["B"] == 100.0


def test_dropped_name_is_fully_sold():
    # Hold C but it's not in the target -> sell the whole position.
    target = pd.Series({"A": 1.0})
    positions = pd.Series({"A": 100.0, "C": 25.0})
    prices = pd.Series({"A": 100.0, "C": 40.0})
    orders = compute_orders(target, positions, prices, equity=10_000)
    assert orders["C"] == -25.0


def test_dust_trades_are_skipped():
    # Already essentially at target -> a sub-min_notional trade is not emitted.
    target = pd.Series({"A": 1.0})
    positions = pd.Series({"A": 100.0})       # 100 * $100 = $10k = full equity
    prices = pd.Series({"A": 100.0})
    orders = compute_orders(target, positions, prices, equity=10_000, min_notional=1.0)
    assert "A" not in orders.index
    assert orders.empty
