"""Supervised feature construction from daily bars (M3).

Turns the tidy daily bars (parse_bars output) into a model-ready frame:
strictly causal features paired with a next-day return target.
"""

import pandas as pd

# 1-day return lags used as features (lag 1 == today's return r_t), and the
# rolling-volatility windows. Kept small on purpose — richer technicals only
# land once the loop can measure whether they help (development_plan M3).
RETURN_LAGS = [1, 2, 3, 5, 10]
VOL_WINDOWS = [10, 20]


def build_features(bars):
    """
    Build a supervised learning frame from tidy daily bars (parse_bars output).

    The task is to predict next-day return, so each row pairs strictly-causal
    features (functions of prices on day t and earlier) with target = r_{t+1}.

    CAUSALITY CONTRACT (pinned by tests/test_features.py):
      - Every feature at row t depends only on adj_close up to and including t.
        Built per-ticker (groupby) so windows/lags never bleed across tickers.
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
    # TODO: implement until tests/test_features.py is green.
    # Work PER TICKER (groupby("ticker")) — keep it causal, never mix tickers:
    #   1. r = adj_close.pct_change()                    # 1-day return at t
    #   2. features (all backward-looking):
    #        ret_lag_k = r.shift(k - 1) for k in RETURN_LAGS   # lag 1 == r_t
    #        vol_w     = r.rolling(w).std() for w in VOL_WINDOWS
    #   3. target = r.shift(-1)        # r_{t+1} — the ONLY look-ahead, per group
    #   4. assemble [date, ticker, <features>, target]; dropna();
    #      sort_values(["ticker", "date"]); reset_index(drop=True)
    # 1-day return per ticker (grouped so each ticker's first row is NaN, never
    # the previous ticker's last price).
    r = bars.groupby("ticker")["adj_close"].pct_change()
    # Re-group the return series so EVERY shift/rolling below also respects
    # ticker boundaries — otherwise lags/vols bleed across tickers at the seams.
    rg = r.groupby(bars["ticker"])

    out = pd.DataFrame({"date": bars["date"], "ticker": bars["ticker"]})
    for k in RETURN_LAGS:
        # k-th return feature, shifted so ret_lag1 == r_t (today's return, known
        # at close t), ret_lag2 == r_{t-1}, ... — all strictly backward-looking.
        out[f"ret_lag{k}"] = rg.shift(k - 1)
    for w in VOL_WINDOWS:
        # groupby().rolling() returns a (ticker, row) MultiIndex; drop the ticker
        # level so the result realigns onto out's row index.
        out[f"vol_{w}"] = rg.rolling(w).std().reset_index(level=0, drop=True)
    out["target"] = rg.shift(-1)  # r_{t+1} — the one deliberate look-ahead

    # Drop warmup rows (NaN lags/vols) AND the last row per ticker (NaN target)
    # in one pass, THEN sort/reindex — so target stays aligned to its own row.
    out = out.dropna().sort_values(["ticker", "date"]).reset_index(drop=True)
    return out
