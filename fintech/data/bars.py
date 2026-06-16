"""Equity daily bars (yfinance): fetching, parsing, and upserting tidy frames."""

from numpy import concat
from pathlib import Path

import pandas as pd
import yfinance as yf


def parse_bars(raw_bars):
    """
    Turn yfinance's download output into one tidy (long) frame of daily bars.

    yfinance pulled with group_by="ticker" returns a *wide* frame: a
    DatetimeIndex of dates, and MultiIndex columns of (ticker, field) e.g.
    ("AAPL", "Close"). This reshapes that into one row per (date, ticker).

    CONTRACT (asserted by tests/test_bars.py):
      - Long shape: one row per (date, ticker), plain integer index.
      - Columns, in this exact order:
            date, ticker, open, high, low, close, adj_close, volume
      - `date` is a tz-naive, midnight-normalized Timestamp (so it compares
        equal to exchange_calendars sessions).
      - Source-of-truth price is `adj_close` (split + dividend adjusted). Note
        yfinance already split-adjusts `close`; auto_adjust only toggles the
        dividend adjustment, which is why both columns are kept.
      - Rows sorted by (ticker, date); no duplicate (date, ticker) pairs.

    Raises KeyError/ValueError on malformed input (e.g. a ticker missing the
    'Adj Close' field) — never skips silently (mirror parse_markets).

    Args:
        raw_bars (pd.DataFrame): yfinance output, MultiIndex (ticker, field)
            columns indexed by date.

    Returns:
        pd.DataFrame: tidy daily bars per the contract above.
    """
    #TODO: implement until tests/test_bars.py is green
    EXPECTED_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for ticker in raw_bars.columns.get_level_values("Ticker").unique():
        present = raw_bars[ticker].columns          # this ticker's fields
        for field in EXPECTED_FIELDS:
            if field not in present:
                raise KeyError(f"{ticker} missing required field {field!r}")
    raw = raw_bars.copy()
    raw = raw.stack(level = 0).reset_index()
    raw.columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw["date"] = raw["date"].dt.tz_localize(None)
    raw = raw.sort_values(["date", "ticker"])
    raw = raw.reset_index(drop=True)
    
    return raw


def fetch_bars(tickers, start, end, auto_adjust=False):
    """
    Fetch daily OHLCV bars from yfinance for one or more tickers.

    auto_adjust is passed explicitly on purpose: its default has changed
    between yfinance versions, so we never let the library decide it for us.

    Args:
        tickers (str | list[str]): ticker symbol(s).
        start (str): inclusive start date, "YYYY-MM-DD".
        end (str): exclusive end date, "YYYY-MM-DD".
        auto_adjust (bool): False keeps raw Close + a separate Adj Close.

    Returns:
        pd.DataFrame: raw yfinance download output.
    """
    return yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=False,
    )


def upsert_bars(new_bars, path):
    """
    Merge freshly-parsed bars into the processed parquet store at `path`.

    The daily snapshot re-fetches an overlapping window every run, so this must
    be idempotent: running it twice over the same dates must NOT create
    duplicate (date, ticker) rows. (date, ticker) is the primary key; on a
    collision the *new* row wins, so a later pull can correct an earlier one.

    CONTRACT (assert these in tests/test_bars.py):
      - If `path` doesn't exist yet, the store is created from `new_bars` alone.
      - After the call, the store has exactly one row per (date, ticker).
      - On a (date, ticker) collision, the value from `new_bars` is kept.
      - Same column order / dtypes / sort as parse_bars output (it's the same
        data, just accumulated): sorted by (ticker, date), plain int index.

    Args:
        new_bars (pd.DataFrame): tidy bars from parse_bars (the contract there).
        path (pathlib.Path or str): processed parquet file (e.g.
            data/processed/bars.parquet).

    Returns:
        pd.DataFrame: the full, deduplicated store that was written to `path`.
    """
    # TODO: implement until tests/test_bars.py is green
    #   1. path = Path(path); read existing store if it exists, else skip
    #   2. concat(existing, new_bars)  -- new rows last so keep="last" favors them
    #   3. drop_duplicates(["date", "ticker"], keep="last")
    #   4. sort_values(["ticker", "date"]) -> reset_index(drop=True)
    #   5. path.parent.mkdir(parents=True, exist_ok=True); to_parquet(path)
    #   6. return the written frame
    path = Path(path)
    try:
        existing = pd.read_parquet(path)
        new = pd.concat([existing, new_bars], ignore_index=True, sort=False)
    except FileNotFoundError:
        new = new_bars.copy()
    new = new.drop_duplicates(["date", "ticker"], keep="last")
    new = new.sort_values(["ticker", "date"])
    new = new.reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path)
    return new
