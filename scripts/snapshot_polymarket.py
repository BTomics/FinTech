from pathlib import Path

import pandas as pd

from fintech.data.polymarket import fetch_markets
from fintech.data.storage import save_snapshot

POLY_API_URL = "https://gamma-api.polymarket.com/markets?limit=500&active=true&closed=false"

snapshot_time = pd.Timestamp.now(tz="UTC")
markets = fetch_markets(POLY_API_URL)
filepath = save_snapshot(markets, snapshot_time, Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket")
print(filepath)