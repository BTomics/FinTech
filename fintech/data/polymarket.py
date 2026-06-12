"""Polymarket API: parsing json data into tidy frames."""

 
import json
from pathlib import Path
import pandas as pd
import requests
POLY_API_URL = "https://gamma-api.polymarket.com/markets"

def parse_markets(raw_markets, snapshot_time):
    """
    Turn a list of raw market dicts into one row per market.

    Assumes binary Yes/No markets where outcomePrices[0] is the "Yes"
    probability. Multi-outcome markets are not handled yet.

    Raises KeyError/ValueError on malformed input — never skips silently.

    Args:
        raw_markets (list[dict]): the raw list of market dicts from the API.
        snapshot_time (pd.Timestamp): when this snapshot was taken.

    Returns:
        pd.DataFrame with columns: snapshot_time, market_id, question, probability, volume
    """
    rows = []
    for m in raw_markets:
        outcome_prices = json.loads(m["outcomePrices"])
        rows.append({
            "snapshot_time": snapshot_time,
            "market_id": m["id"],
            "question": m["question"],
            "probability": float(outcome_prices[0]),
            "volume": float(m["volumeNum"])
        })
    return pd.DataFrame(rows)

def fetch_markets(api_url):
    """
    Fetch all markets from Polymarket API.
    Args:
        api_url (str): the API URL.
    Returns:
        list[dict]: the raw list of market dicts from the API.
    """
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()
    


def save_snapshot(markets, snapshot_time, output_dir):
    """
    Save markets to a time-stamped json file in the given directory.
    within one output directory, one timestamp = one snapshot, and a collision is an error.
    Args:
        markets (pd.DataFrame): output of parse_markets.
        snapshot_time (pd.Timestamp): snapshot timestamp.
        output_dir (pathlib.Path or str): where to save the file.

    Returns:
        pathlib.Path: absolute path to the created json file.
    Raises:
        FileExistsError: if the same file already exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    filename = snapshot_time.strftime("%Y%m%dT%H%M%SZ.json")
    filepath = output_dir / filename
    if filepath.exists():
        raise FileExistsError(f"File already exists: {filepath}")
   
    with open(filepath, "x", encoding="utf-8") as f:
        json.dump(markets, f, indent=2)
    return filepath

