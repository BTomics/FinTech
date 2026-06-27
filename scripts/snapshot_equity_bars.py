"""Daily equity-bars snapshot: refresh recent bars for the FULL universe.

Re-fetches a short lookback window for every S&P 500 name (+SPY) and upserts it
into the processed store. The lookback overlaps previous runs so the idempotent
upsert self-heals any missed day.

Must cover the WHOLE universe — the strategy trades all of it, so refreshing only
a handful of names would leave the rest stale (the bug that poisoned the first
live run). Run after the US close, on a daily schedule.
"""
from pathlib import Path

import pandas as pd

from fintech.data.bars import refresh_universe_bars
from fintech.data.universe import load_universe

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "bars.parquet"
LOOKBACK_DAYS = 10

end = pd.Timestamp.now().strftime("%Y-%m-%d")
start = (pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

tickers = load_universe()
written, failed = refresh_universe_bars(tickers, start, end, PROCESSED_PATH)
print(f"Refreshed {len(tickers)} tickers, {start} -> {end}. "
      f"Store: {len(written)} rows, {written['ticker'].nunique()} tickers.")
if failed:
    print(f"Failed/empty ({len(failed)}): {failed}")
