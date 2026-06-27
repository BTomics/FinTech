from pathlib import Path
import yfinance as yf

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "equity_bars_sample.parquet"

df = yf.download(
    tickers=["SPY", "AAPL", "MSFT"],
    period="3y",         # >252 sessions so the 12-month momentum feature has data
    interval="1d",
    auto_adjust=False,   # <-- YOUR DECISION (see below)
    group_by="ticker",
    progress=False,
)
df = df.dropna(how="any")  # drop any incomplete trailing bar (NaN close)
df_split = yf.download(
    tickers="NVDA",
    start="2024-05-01", end="2024-07-01",
    interval="1d", auto_adjust=False,
    group_by="ticker",   # keep (ticker, field) column order, like the main pull
    progress=False,
)
df_split.to_parquet(OUT.parent / "nvda_split_sample.parquet")
print(df.tail())
print("shape:", df.shape)
df.to_parquet(OUT)
print("saved ->", OUT)