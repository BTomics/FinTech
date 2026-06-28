"""Equity daily bars (yfinance): fetching, parsing, and upserting tidy frames."""

from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf

# How long after the official close a daily bar must wait before we trust it.
# yfinance returns a *provisional* bar for the in-progress / just-closed US
# session (non-NaN but not final, and it arrives ticker-by-ticker), so writing
# it lands the latest date in the store ragged and revisable. Two hours clears
# the settlement window and means the daily 22:30-local snapshot (~30 min after
# close) defers today's bar to the next run, which re-fetches it via the
# overlapping lookback once it's final.
SETTLE_BUFFER = pd.Timedelta(hours=2)


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
    # Drop incomplete bars (NaN source-of-truth price) — an unsettled/partial
    # fetch is not a real bar, and must never be upserted over good data.
    raw = raw.dropna(subset=["adj_close"])
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


def _drop_empty_tickers(raw):
    """Drop tickers whose columns came back all-NaN (failed download)."""
    keep = [t for t in raw.columns.get_level_values("Ticker").unique()
            if not raw[t].isna().all().all()]
    return raw.loc[:, raw.columns.get_level_values("Ticker").isin(keep)], keep


def drop_incomplete_sessions(bars, now=None, calendar="XNYS", buffer=SETTLE_BUFFER):
    """
    Drop trailing bars whose trading session hasn't fully closed + settled yet.

    The daily snapshot re-fetches a recent window after the US close, but
    yfinance hands back a *provisional* bar for the in-progress / just-closed
    session: the values are non-NaN (so parse_bars' adj_close dropna keeps them)
    yet not final, and different tickers update at different moments. Writing
    that session leaves the latest date in the store with ragged coverage and
    revisable prices — the bug that poisoned the live cross-section (and the
    reason paper_trade has to filter on coverage>0.9).

    A session is trustworthy only once its official exchange close plus a
    settlement `buffer` is in the past. Everything on or after that cutoff is
    dropped here; the overlapping daily lookback re-fetches and writes it on a
    later run once it has settled, so nothing is permanently lost.

    Args:
        bars (pd.DataFrame): tidy bars (parse_bars output); needs a `date`
            column of tz-naive midnight Timestamps.
        now (pd.Timestamp | None): reference "current time"; defaults to now in
            UTC. A tz-naive value is assumed to be UTC. (Injectable for tests.)
        calendar (str): exchange_calendars code for the session schedule.
        buffer (pd.Timedelta): how long past the close a bar must be to count.

    Returns:
        pd.DataFrame: `bars` with un-settled trailing sessions removed.
    """
    if bars.empty:
        return bars
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    # closes: Series indexed by tz-naive session midnight, values tz-aware UTC.
    closes = xcals.get_calendar(calendar).closes
    settled = closes[closes + buffer <= now]
    if settled.empty:
        return bars.iloc[0:0]
    cutoff = settled.index[-1]            # last fully-settled session (midnight)
    return bars[bars["date"] <= cutoff]


def refresh_universe_bars(tickers, start, end, path, chunk_size=50):
    """
    Fetch -> parse -> upsert daily bars for many tickers, robustly.

    Fetches in chunks (gentle on yfinance) and drops tickers that came back
    empty, so one bad symbol can't abort the batch (parse_bars is strict). Used
    by both the one-off backfill and the daily snapshot — same code path, only
    the date window differs.

    Args:
        tickers (list[str]): symbols to fetch.
        start, end (str): "YYYY-MM-DD"; end is exclusive (yfinance convention).
        path: processed parquet store to upsert into.
        chunk_size (int): tickers per yfinance call.

    Returns:
        tuple[pd.DataFrame, list[str]]: (written store, failed/empty tickers).
    """
    parsed = []
    failed = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        raw = fetch_bars(chunk, start, end)
        raw, kept = _drop_empty_tickers(raw)
        failed += [t for t in chunk if t not in kept]
        if kept:
            parsed.append(parse_bars(raw))
    fresh = pd.concat(parsed, ignore_index=True)
    # Never write a session that hasn't settled — keeps the latest store date
    # complete and stable across tickers (see drop_incomplete_sessions).
    fresh = drop_incomplete_sessions(fresh)
    written = upsert_bars(fresh, path)
    return written, failed
