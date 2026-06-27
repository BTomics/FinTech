"""One-off backfill of daily bars for the full S&P 500 universe.

Fetches in chunks (gentle on yfinance), drops tickers that came back empty so
one bad symbol can't abort the whole batch (parse_bars is strict by design),
then upserts everything into the processed bars.parquet idempotently.

Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\backfill_bars.py
"""

from pathlib import Path

import pandas as pd

from fintech.data.bars import fetch_bars, parse_bars, upsert_bars
from fintech.data.universe import load_universe

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "bars.parquet"

START = "2010-01-01"
CHUNK_SIZE = 50


def _drop_empty_tickers(raw):
    """Drop tickers whose columns came back all-NaN (failed download)."""
    keep = [t for t in raw.columns.get_level_values("Ticker").unique()
            if not raw[t].isna().all().all()]
    return raw.loc[:, raw.columns.get_level_values("Ticker").isin(keep)], keep


def backfill(tickers, start, end, chunk_size=CHUNK_SIZE, path=PROCESSED_PATH):
    parsed = []
    failed = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        raw = fetch_bars(chunk, start, end)
        raw, kept = _drop_empty_tickers(raw)
        failed += [t for t in chunk if t not in kept]
        if not kept:
            continue
        parsed.append(parse_bars(raw))
        print(f"  chunk {i // chunk_size + 1}: {len(kept)}/{len(chunk)} ok")

    all_bars = pd.concat(parsed, ignore_index=True)
    written = upsert_bars(all_bars, path)
    return written, failed


def main():
    tickers = load_universe()
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    print(f"Backfilling {len(tickers)} tickers, {START} -> {end} "
          f"(survivorship-biased: today's membership)")
    written, failed = backfill(tickers, START, end)
    print(f"\nDone. bars.parquet: {len(written)} rows, "
          f"{written['ticker'].nunique()} tickers.")
    if failed:
        print(f"Failed/empty ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
