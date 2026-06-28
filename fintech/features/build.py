"""Supervised feature construction from daily bars (M3).

Turns the tidy daily bars (parse_bars output) into a model-ready frame:
strictly causal features paired with a next-day return target.
"""

import numpy as np
import pandas as pd

# Feature config. Each family captures a distinct, well-documented effect:
#   ret_lag*  short-term returns (days)        -> short-horizon reversal
#   mom_*     cumulative return over W days     -> momentum (3-12 month)
#   mom_252_21  12-month return skipping last month -> classic UMD (drops the
#               most recent month, which is reversal, not momentum)
#   vol_*     rolling stdev of daily returns    -> realised volatility / risk
#   rvol_*    today's volume / its W-day mean    -> relative volume (activity)
#   illiq_*   mean |return| / dollar volume      -> Amihud (2002) illiquidity:
#             price impact per dollar traded; higher = more illiquid
RETURN_LAGS = [1, 2, 3, 5, 10]
MOM_WINDOWS = [21, 63, 126, 252]   # ~1, 3, 6, 12 months
VOL_WINDOWS = [10, 20, 63]
VOLUME_WINDOWS = [20, 63]
ILLIQ_WINDOWS = [20, 63]


def build_features(bars):
    """
    Build a supervised learning frame from tidy daily bars (parse_bars output).

    The task is to predict next-day return, so each row pairs strictly-causal
    features (functions of prices on day t and earlier) with target = r_{t+1}.

    CAUSALITY CONTRACT (pinned by tests/test_features.py):
      - Every feature at row t depends only on adj_close / volume up to and
        including t. Built per-ticker (groupby) so windows never bleed across
        tickers.
      - The ONLY forward-looking column is `target` (next-day simple return) —
        the single deliberate .shift(-1). Everything else looks backward.
      - Output is tidy/long: one row per (date, ticker), plain int index,
        columns in order: date, ticker, <feature cols...>, target; sorted by
        (ticker, date).
      - Warmup rows left NaN by the lags/rolling windows AND the final row per
        ticker (its target is unknown) are dropped — the output has no NaNs.

    Args:
        bars (pd.DataFrame): tidy daily bars from parse_bars (columns include
            date, ticker, adj_close; sorted by (ticker, date)).

    Returns:
        pd.DataFrame: supervised frame per the contract above.
    """
    # 1-day return per ticker (grouped so each ticker's first row is NaN, never
    # the previous ticker's last price).
    r = bars.groupby("ticker")["adj_close"].pct_change()
    # Re-group the return series so EVERY shift/rolling below also respects
    # ticker boundaries — otherwise lags/vols bleed across tickers at the seams.
    rg = r.groupby(bars["ticker"])
    # Grouped price and volume for the longer-horizon features; both shift/roll
    # within a ticker only.
    pg = bars.groupby("ticker")["adj_close"]
    vg = bars.groupby("ticker")["volume"]
    price = bars["adj_close"]

    out = pd.DataFrame({"date": bars["date"], "ticker": bars["ticker"]})
    for k in RETURN_LAGS:
        # k-th return feature, shifted so ret_lag1 == r_t (today's return, known
        # at close t), ret_lag2 == r_{t-1}, ... — all strictly backward-looking.
        out[f"ret_lag{k}"] = rg.shift(k - 1)
    for w in MOM_WINDOWS:
        # Momentum = cumulative return over the past w days: price_t / price_{t-w}
        # - 1. Grouped shift keeps it within-ticker; aligns by row index.
        out[f"mom_{w}"] = price / pg.shift(w) - 1
    # Classic 12-1 momentum: the 12-month return EXCLUDING the most recent month
    # (the last month is short-term reversal, which would dilute the signal).
    out["mom_252_21"] = pg.shift(21) / pg.shift(252) - 1
    for w in VOL_WINDOWS:
        # groupby().rolling() returns a (ticker, row) MultiIndex; drop the ticker
        # level so the result realigns onto out's row index.
        out[f"vol_{w}"] = rg.rolling(w).std().reset_index(level=0, drop=True)
    for w in VOLUME_WINDOWS:
        # Relative volume: today's volume vs its own w-day average (>1 = unusually
        # active). Uses volume through t only.
        avg_vol = vg.rolling(w).mean().reset_index(level=0, drop=True)
        out[f"rvol_{w}"] = bars["volume"] / avg_vol
    # Amihud illiquidity: the daily |return| / dollar-volume ratio (price impact
    # per dollar traded), averaged over the past w days. Built from r (today's
    # return) and today's dollar volume — both known at close t, so causal.
    # Zero-volume days make this inf; the replace([inf], nan) below clears them.
    daily_illiq = r.abs() / (bars["adj_close"] * bars["volume"])
    ig = daily_illiq.groupby(bars["ticker"])
    for w in ILLIQ_WINDOWS:
        out[f"illiq_{w}"] = ig.rolling(w).mean().reset_index(level=0, drop=True)
    out["target"] = rg.shift(-1)  # r_{t+1} — the one deliberate look-ahead

    # Guard against div-by-zero (e.g. a zero-volume window) producing inf, which
    # dropna would NOT remove.
    out = out.replace([np.inf, -np.inf], np.nan)
    # Drop warmup rows (NaN lags/vols/momentum) AND the last row per ticker (NaN
    # target) in one pass, THEN sort/reindex — so target stays aligned to its row.
    out = out.dropna().sort_values(["ticker", "date"]).reset_index(drop=True)
    return out
