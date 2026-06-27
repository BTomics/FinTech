"""Alpaca paper-trading client (M6).

Thin I/O wrapper around alpaca-py: read the account/positions, submit the orders
that compute_orders produced. PAPER ONLY — keys come from the gitignored .env
(ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_PAPER), never the repo.

Kept deliberately thin: all the decision logic lives in orders.py (pure,
tested); this layer only talks to the broker, so the untestable part stays
minimal.
"""

import os

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def get_client():
    """TradingClient from env keys (paper unless ALPACA_PAPER is false)."""
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    return TradingClient(
        os.environ["ALPACA_API_KEY"],
        os.environ["ALPACA_SECRET_KEY"],
        paper=paper,
    )


def get_equity(client):
    """Total account value (cash + positions)."""
    return float(client.get_account().equity)


def get_positions(client):
    """Current holdings as a {ticker: shares} Series (empty if flat)."""
    positions = client.get_all_positions()
    if not positions:
        return pd.Series(dtype=float)
    return pd.Series({p.symbol: float(p.qty) for p in positions})


def submit_orders(client, orders):
    """
    Submit each non-zero order as a market order; return submission records.

    One failing symbol (e.g. not fractionable, halted) is logged and skipped, not
    allowed to abort the whole batch.

    Args:
        client: alpaca TradingClient.
        orders (pd.Series): signed shares per ticker (compute_orders output).

    Returns:
        list[dict]: one record per ticker — intended qty plus the order id/status
            or the error — for the reconciliation log.
    """
    records = []
    for symbol, qty in orders.items():
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=symbol,
            qty=round(abs(float(qty)), 6),  # Alpaca allows fractional shares
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            order = client.submit_order(request)
            records.append({"symbol": symbol, "qty": float(qty),
                            "order_id": str(order.id), "status": str(order.status)})
        except Exception as exc:  # noqa: BLE001 — never let one symbol abort the run
            records.append({"symbol": symbol, "qty": float(qty),
                            "order_id": None, "status": f"ERROR: {exc}"})
    return records
