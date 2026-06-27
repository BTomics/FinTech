"""Reconciliation report: intended vs filled (M6).

Market orders fill asynchronously (and queue overnight when placed after close),
so fills can't be checked in the same run that submits them. Run this AFTER the
next open to see what actually filled — intended qty vs filled qty/price.

    .venv\\Scripts\\python.exe scripts\\reconcile.py
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from fintech.execution.broker import get_client, get_recent_orders

LOOKBACK_DAYS = 4  # cover a weekend


def main():
    load_dotenv()
    client = get_client()
    after = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    orders = get_recent_orders(client, after=after)

    if orders.empty:
        print(f"No orders in the last {LOOKBACK_DAYS} days.")
        return

    filled = orders["status"].eq("FILLED").sum()
    print(f"{len(orders)} orders since {after.date()}: {filled} filled, "
          f"{len(orders) - filled} not.\n")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(orders[["symbol", "side", "intended_qty", "filled_qty",
                      "filled_avg_price", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
