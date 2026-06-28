"""Turn the Kalshi macro snapshots into daily signals + a first 'is it useful?' read.

The Kalshi logger captures macro strike-ladders (KXFED rate path, KXCPIYOY
inflation, KXRECSSNBER recession). Each ladder is a survival curve P(value > X)
across strikes X, so its implied MEDIAN (the X where P crosses 0.5) is a clean
point estimate of what the market expects. This script builds those daily series
and checks (a) they move sensibly and (b) whether daily *changes* line up with
SPY returns — purely exploratory: ~17 days is far too short to conclude anything,
it's a sanity/POC pass, not a validated feature.

    .venv\\Scripts\\python.exe -m scripts.kalshi_macro_signal

Research script, not part of the live path. If this proves worthwhile the loader
graduates into fintech/data/kalshi.py with tests.
"""

import glob
import gzip
import json
import os

import numpy as np
import pandas as pd

KALSHI_DIR = "data/raw/kalshi"
BARS_PATH = "data/processed/bars.parquet"


def _load(fp):
    op = gzip.open if fp.endswith(".gz") else open
    with op(fp, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_kalshi_panel():
    """All snapshots -> long frame: snapshot_time, series, event, strike, price."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(KALSHI_DIR, "*"))):
        t = pd.Timestamp(os.path.basename(fp).split(".")[0])
        for series, mks in _load(fp).items():
            for m in mks:
                rows.append({
                    "t": t,
                    "series": series,
                    "event": m.get("event_ticker"),
                    "close": pd.Timestamp(m["close_time"]) if m.get("close_time") else pd.NaT,
                    # "above X%" ladders carry the threshold X in floor_strike
                    # (strike_type='greater'); fall back for other layouts.
                    "strike": m.get("floor_strike", m.get("cap_strike", m.get("custom_strike"))),
                    "price": m.get("last_price_dollars"),
                    "title": m.get("title", ""),
                })
    df = pd.DataFrame(rows)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def _implied_median(strikes, probs):
    """Strike where the survival curve P(value > strike) crosses 0.5 (interp)."""
    d = pd.DataFrame({"k": strikes, "p": probs}).dropna().sort_values("k")
    if len(d) < 2:
        return np.nan
    k, p = d["k"].to_numpy(), d["p"].to_numpy()       # p is decreasing in k
    for i in range(len(k) - 1):
        if (p[i] - 0.5) * (p[i + 1] - 0.5) <= 0 and p[i] != p[i + 1]:
            return k[i] + (k[i + 1] - k[i]) * (p[i] - 0.5) / (p[i] - p[i + 1])
    if p[0] < 0.5:   # median below the lowest strike
        return k[0]
    if p[-1] > 0.5:  # median above the highest strike
        return k[-1]
    return np.nan


def _nearest_event(df, series):
    """Front contract: soonest-closing event that stays quoted all window.

    Restrict to events present in >=95% of snapshots so we don't pick one that
    settles mid-window (e.g. the June FOMC), then take the soonest close.
    """
    g = df[df["series"] == series]
    if g.empty:
        return None
    n = df["t"].nunique()
    cover = g.groupby("event")["t"].nunique()
    full = cover[cover >= 0.95 * n].index
    g = g[g["event"].isin(full)]
    return g.loc[g["close"].idxmin(), "event"] if not g.empty else None


def build_daily_signals(df):
    """Daily (last-snapshot-of-day) macro signals from the ladders."""
    cpi_event = _nearest_event(df, "KXCPIYOY")
    fed_front = _nearest_event(df, "KXFED")

    per_snap = []
    for t, snap in df.groupby("t"):
        row = {"t": t}
        # implied front-month CPI YoY and front-meeting fed-funds upper bound
        cpi = snap[snap["event"] == cpi_event]
        row["cpi_yoy"] = _implied_median(cpi["strike"], cpi["price"])
        fed = snap[snap["event"] == fed_front]
        row["fed_rate"] = _implied_median(fed["strike"], fed["price"])
        # recession is a direct binary per year
        rec = snap[snap["series"] == "KXRECSSNBER"]
        for _, m in rec.iterrows():
            yr = m["event"].split("-")[-1]
            row[f"rec_20{yr}"] = m["price"]
        per_snap.append(row)

    sig = pd.DataFrame(per_snap).set_index("t").sort_index()
    # collapse to one value per calendar day = last snapshot that day
    daily = sig.groupby(sig.index.normalize().tz_localize(None)).last()
    daily.index.name = "date"
    return daily, {"cpi_event": cpi_event, "fed_front": fed_front}


def _spy_returns():
    bars = pd.read_parquet(BARS_PATH)
    spy = bars[bars["ticker"] == "SPY"].set_index("date")["adj_close"].sort_index()
    return spy.pct_change().rename("spy_ret")


def main():
    df = load_kalshi_panel()
    daily, meta = build_daily_signals(df)

    print(f"\nKalshi macro daily signals  (front CPI event={meta['cpi_event']}, "
          f"front FOMC={meta['fed_front']})")
    print(f"{df['t'].nunique()} snapshots over {len(daily)} days "
          f"({daily.index.min().date()} -> {daily.index.max().date()})\n")
    with pd.option_context("display.float_format", lambda x: f"{x:0.3f}", "display.width", 200):
        print(daily.to_string())

    # movement check: how much did each signal travel over the window?
    print("\nMove over window (first -> last, and daily-change std):")
    for c in daily.columns:
        s = daily[c].dropna()
        if len(s) > 1:
            print(f"  {c:10s} {s.iloc[0]:.3f} -> {s.iloc[-1]:.3f}   "
                  f"range[{s.min():.3f},{s.max():.3f}]  dstd={s.diff().std():.4f}")

    # --- 'is it good to have?': do daily CHANGES relate to SPY returns? ---------
    spy = _spy_returns()
    chg = daily.diff()
    aligned = chg.join(spy, how="inner").dropna(how="all")
    n = aligned["spy_ret"].notna().sum()
    print(f"\nAlignment with SPY: {n} overlapping trading days (TOO FEW to conclude — "
          "illustrative only).")
    print("corr(daily signal change, SPY return)  [same-day | next-day SPY]:")
    for c in daily.columns:
        same = aligned[c].corr(aligned["spy_ret"])
        nxt = aligned[c].corr(aligned["spy_ret"].shift(-1))
        print(f"  {c:10s} same={same:+.2f}  next={nxt:+.2f}")
    print()


if __name__ == "__main__":
    main()
