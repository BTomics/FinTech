from pathlib import Path
import yfinance as yf

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "equity_bars_sample.parquet"

df = yf.download(
    tickers=["SPY", "AAPL", "MSFT"],
    period="6mo",
    interval="1d",
    auto_adjust=False,   # <-- YOUR DECISION (see below)
    group_by="ticker",
    progress=False,
)
df_split = yf.download(
    tickers="NVDA",
    start="2024-05-01", end="2024-07-01",
    interval="1d", auto_adjust=False, progress=False,
)
df_split.to_parquet(OUT.parent / "nvda_split_sample.parquet")
print(df.tail())
print("shape:", df.shape)
df.to_parquet(OUT)
print("saved ->", OUT)