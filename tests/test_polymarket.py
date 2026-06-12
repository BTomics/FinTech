"""Tests for fintech.data.polymarket.

parse_markets(raw_markets, snapshot_time) takes the parsed JSON (the list of
market dicts) plus the timestamp of when it was fetched, and returns a pandas
DataFrame with columns: snapshot_time, market_id, question, probability, volume.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from fintech.data.polymarket import parse_markets, save_snapshot

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

def test_onefile_outputdir():
    out_dir = Path("tests/fixtures/snapshots")
    out_dir.mkdir(exist_ok=True)
    snapshot_time = pd.Timestamp("2026-06-12T18:00:00Z")
    save_snapshot(load_fixture(), snapshot_time, out_dir)
    assert out_dir.joinpath("20260612T180000Z.parquet").exists()

def test_twofiles_difftimes():
    out_dir = Path("tests/fixtures/snapshots")
    out_dir.mkdir(exist_ok=True)
    save_snapshot(load_fixture(), pd.Timestamp("2026-06-12T18:00:00Z"), out_dir)
    save_snapshot(load_fixture(), pd.Timestamp("2026-06-12T19:00:00Z"), out_dir)
    assert out_dir.joinpath("20260612T180000Z.parquet").exists()
    assert out_dir.joinpath("20260612T190000Z.parquet").exists()

def test_twofiles_loudfail():
    out_dir = Path("tests/fixtures/snapshots")
    out_dir.mkdir(exist_ok=True)
    save_snapshot(load_fixture(), pd.Timestamp("2026-06-12T18:00:00Z"), out_dir)
    with pytest.raises(FileExistsError):
        save_snapshot(load_fixture(), pd.Timestamp("2026-06-12T18:00:00Z"), out_dir)