"""Equity daily bars (yfinance): fetching and parsing into tidy frames."""

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
