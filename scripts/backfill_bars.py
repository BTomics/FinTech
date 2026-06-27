"""One-off backfill of daily bars for the full S&P 500 universe.

Thin wrapper around fintech.data.bars.refresh_universe_bars with a 2010 start.
Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\backfill_bars.py
"""

from pathlib import Path

import pandas as pd

from fintech.data.bars import refresh_universe_bars
from fintech.data.universe import load_universe

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "bars.parquet"
START = "2010-01-01"


def main():
    tickers = load_universe()
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    print(f"Backfilling {len(tickers)} tickers, {START} -> {end} "
          f"(survivorship-biased: today's membership)")
    written, failed = refresh_universe_bars(tickers, START, end, PROCESSED_PATH)
    print(f"\nDone. bars.parquet: {len(written)} rows, "
          f"{written['ticker'].nunique()} tickers.")
    if failed:
        print(f"Failed/empty ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
