"""Polymarket API: parsing json data into tidy frames."""

import json

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

