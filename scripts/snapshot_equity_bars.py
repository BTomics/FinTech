"""Daily equity-bars snapshot: fetch -> save raw -> upsert into processed bars.parquet."""
from pathlib import Path

import pandas as pd

from fintech.data.bars import fetch_bars, parse_bars, upsert_bars

UNIVERSE = ["SPY", "AAPL", "MSFT"]      # config at top, not hardcoded in the logic
LOOKBACK_DAYS = 10                     # DECISION 2

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "equity_bars"
PROCESSED_PATH = ROOT / "data" / "processed" / "bars.parquet"
snapshot_time = pd.Timestamp.now(tz="UTC")
# 1. window
end = (snapshot_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
start = (snapshot_time - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

# 2. fetch raw
raw = fetch_bars(UNIVERSE, start, end)

# 3. save raw  --> DECISION 1
RAW_DIR.mkdir(parents=True, exist_ok=True)
raw.to_parquet(RAW_DIR / snapshot_time.strftime("%Y%m%dT%H%M%SZ.parquet"))


# 4. parse to tidy
bars = parse_bars(raw)

# 5. upsert into processed: read (if exists) -> concat -> drop_duplicates(["date","ticker"], keep="last") -> sort -> write

written = upsert_bars(bars, PROCESSED_PATH)
print(PROCESSED_PATH)