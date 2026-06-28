"""Tests for fintech.data.bars.

parse_bars(raw_bars) takes yfinance's download output (a wide frame with a
DatetimeIndex of dates and MultiIndex (ticker, field) columns) and returns a
tidy (long) DataFrame: one row per (date, ticker), columns
    date, ticker, open, high, low, close, adj_close, volume
with adj_close as the source-of-truth price.

Two frozen fixtures (pulled once, run offline):
  - equity_bars_sample.parquet : SPY/AAPL/MSFT, ~6 months, no splits.
        Used for the shape / no-NaN / calendar-gap contract.
  - nvda_split_sample.parquet   : NVDA around its 2024-06-10 10-for-1 split.
        Used only for the "a split must not look like a crash" guard.
"""

from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import pytest

from fintech.data.bars import drop_incomplete_sessions, parse_bars, upsert_bars

FIXTURES = Path(__file__).parent / "fixtures"
BARS_PATH = FIXTURES / "equity_bars_sample.parquet"
SPLIT_PATH = FIXTURES / "nvda_split_sample.parquet"

EXPECTED_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def load_bars():
    return pd.read_parquet(BARS_PATH)


def load_split_bars():
    return pd.read_parquet(SPLIT_PATH)


# --- contract tests (against the no-split 6mo fixture) -----------------------

def test_tidy_columns():
    parsed = parse_bars(load_bars())
    assert list(parsed.columns) == EXPECTED_COLUMNS


def test_one_row_per_date_ticker_unique_and_sorted():
    parsed = parse_bars(load_bars())
    # no duplicate (date, ticker) pairs
    assert not parsed.duplicated(subset=["date", "ticker"]).any()
    # dates run forward within each ticker
    for _, group in parsed.groupby("ticker"):
        assert group["date"].is_monotonic_increasing


def test_no_nan_gaps():
    parsed = parse_bars(load_bars())
    assert parsed.isna().sum().sum() == 0


def test_no_missing_sessions_vs_calendar():
    raw = load_bars()
    parsed = parse_bars(raw)

    # exchange_calendars gives the official NYSE trading sessions in a range,
    # so you can compare "dates I have" against "dates the exchange was open".
    # Sessions come back as a tz-naive DatetimeIndex (midnight per session).
    nyse = xcals.get_calendar("XNYS")
    sessions = nyse.sessions_in_range(raw.index.min(), raw.index.max())

    one_ticker = parsed[parsed["ticker"] == "AAPL"]
    assert set(one_ticker["date"]) == set(sessions)
    assert len(one_ticker) == len(sessions)


# --- split handling (against the NVDA split fixture) -------------------------

def test_split_is_not_a_price_crash():
    parsed = parse_bars(load_split_bars())

    # Source-of-truth adj_close is split-adjusted, so the 10-for-1 split on
    # 2024-06-10 must NOT show up as a ~90% overnight drop. Real daily moves in
    # this window peak near 9%, so a 50% bound cleanly separates "normal
    # volatility" from "an unhandled split looked like a crash".
    nvda = parsed[parsed["ticker"] == "NVDA"].sort_values("date")
    daily_returns = nvda["adj_close"].pct_change()
    assert daily_returns.abs().max() < 0.5


# --- malformed input (mirror test_polymarket.test_missing_price_raises) ------

def test_malformed_input_raises():
    raw = load_bars()
    del raw[("AAPL", "Adj Close")]  # columns are (ticker, field)
    with pytest.raises(KeyError):
        parse_bars(raw)


# --- upsert tests ---------------------------------------------------------
def test_upsert_idempotent(tmp_path):
    path = tmp_path / "bars.parquet"   # auto-created, auto-cleaned, no unlink
    # assert that running upsert twice with the same input does not change the output
    new = parse_bars(load_bars())
    written = upsert_bars(new, path)
    written2 = upsert_bars(new, path)
    pd.testing.assert_frame_equal(written, written2)

def test_upsert_collision(tmp_path):
    path = tmp_path / "bars.parquet"   # auto-created, auto-cleaned, no unlink
    original = parse_bars(load_bars())
    upsert_bars(original, path)

    # re-upsert the same (date, ticker) with a changed price: the new row wins.
    some_date = original[original["ticker"] == "AAPL"]["date"].iloc[0]
    modified = original.copy()
    mask = (modified["ticker"] == "AAPL") & (modified["date"] == some_date)
    modified.loc[mask, "adj_close"] = -1.0
    result = upsert_bars(modified, path)

    # new value survived (catches keep="first"), and it replaced rather than appended
    assert result.loc[
        (result["ticker"] == "AAPL") & (result["date"] == some_date), "adj_close"
    ].iloc[0] == -1.0
    assert len(result) == len(original)


# --- settlement filter (the snapshot "corrupt latest bar" fix) ---------------

def test_drop_incomplete_sessions_keeps_settled_drops_unsettled():
    # Three real consecutive NYSE sessions (Wed/Thu/Fri); each closes 20:00 UTC.
    dates = pd.to_datetime(["2026-06-24", "2026-06-25", "2026-06-26"])
    bars = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAPL"] * 3 + ["MSFT"] * 3,
            "adj_close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    # "now" = 21:00 UTC on the 26th: the 26th's close+2h (22:00) is still in the
    # future (unsettled, provisional), the 24th/25th are well past settlement.
    now = pd.Timestamp("2026-06-26 21:00", tz="UTC")
    kept = drop_incomplete_sessions(bars, now=now)

    assert set(kept["date"]) == {dates[0], dates[1]}   # 24th, 25th survive
    assert dates[2] not in set(kept["date"])           # 26th (unsettled) dropped
    # both tickers' rows for a surviving session are kept (no per-ticker raggedness)
    assert (kept["date"] == dates[1]).sum() == 2


def test_drop_incomplete_sessions_keeps_all_when_fully_settled():
    dates = pd.to_datetime(["2026-06-24", "2026-06-25", "2026-06-26"])
    bars = pd.DataFrame({"date": dates, "ticker": "AAPL", "adj_close": [1.0, 2.0, 3.0]})
    # A week later everything has long settled -> nothing is dropped.
    now = pd.Timestamp("2026-07-03 12:00", tz="UTC")
    kept = drop_incomplete_sessions(bars, now=now)
    assert set(kept["date"]) == set(dates)
