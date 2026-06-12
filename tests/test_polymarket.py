"""Tests for fintech.data.polymarket.

parse_markets(raw_markets, snapshot_time) takes the parsed JSON (the list of
market dicts) plus the timestamp of when it was fetched, and returns a pandas
DataFrame with columns: snapshot_time, market_id, question, probability, volume.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from fintech.data.polymarket import parse_markets

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "polymarket_markets_sample.json"

def load_fixture():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_one_row_per_market():
    raw_markets = load_fixture()
    snapshot_time = pd.Timestamp("2026-06-12T18:00:00Z")

    parsed = parse_markets(raw_markets, snapshot_time)

    assert len(parsed) == len(raw_markets)

def test_five_columns():
    raw_markets = load_fixture()
    snapshot_time = pd.Timestamp("2026-06-12T18:00:00Z")

    parsed = parse_markets(raw_markets, snapshot_time)
    assert parsed.shape[1] == 5
    assert list(parsed.columns) == [
        "snapshot_time",
        "market_id",
        "question",
        "probability",
        "volume",
    ]
def test_probability():
    raw_markets = load_fixture()
    snapshot_time = pd.Timestamp("2026-06-12T18:00:00Z")

    parsed = parse_markets(raw_markets, snapshot_time)
    assert parsed["probability"].between(0, 1).all()
    
def test_missing_price_raises():
    broken = load_fixture()
    del broken[0]["outcomePrices"]
    with pytest.raises(KeyError):
        parse_markets(broken, pd.Timestamp("2026-06-12T18:00:00Z"))

