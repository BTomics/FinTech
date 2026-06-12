from pathlib import Path

import pandas as pd

from fintech.data.kalshi import fetch_series_markets
from fintech.data.storage import save_snapshot

WATCHLIST = ["KXFED", "KXFEDDECISION", "KXCPI", "KXCPIYOY", "KXGDP", "KXRECSSNBER"]

snapshot_time = pd.Timestamp.now(tz="UTC")
payload = {ticker: fetch_series_markets(ticker) for ticker in WATCHLIST}
filepath = save_snapshot(
    payload, snapshot_time, Path(__file__).resolve().parents[1] / "data" / "raw" / "kalshi"
)
print(filepath)
