"""Tests for fintech.features.build.

build_features(bars) turns the tidy daily bars (parse_bars output) into a
supervised frame: one row per (date, ticker), a set of strictly-causal feature
columns, and a `target` = next-day return r_{t+1}.

The central guarantee is CAUSALITY: every feature at row t is a function of
prices on day t and earlier only. The single forward-looking column is
`target`, by design. The leakage test below pins exactly that boundary —
perturbing a future price must not move any past feature.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fintech.data.bars import parse_bars
from fintech.features.build import build_features

FIXTURES = Path(__file__).parent / "fixtures"
BARS_PATH = FIXTURES / "equity_bars_sample.parquet"


def load_bars():
    return pd.read_parquet(BARS_PATH)


def test_features_are_causal():
    """Perturbing a FUTURE price must not change any earlier feature.

    Build features from the bars, then rebuild from a copy whose most recent
    session (for one ticker) has had its adj_close shocked. Every *feature*
    value for every earlier row must be byte-for-byte identical — features look
    only backward. `target` is excluded from the comparison on purpose: the
    second-to-last row's target = r_{t+1} legitimately uses the perturbed day,
    and that is the one allowed forward dependency, not a leak.
    """
    bars = parse_bars(load_bars())
    base = build_features(bars)

    # Perturb the FIRST ticker's most recent session. AAPL sorts first, so if any
    # shift/rolling crossed ticker boundaries, its tail would bleed into the next
    # ticker's early rows — and shocking AAPL's future would then move them.
    tkr = "AAPL"
    tampered = bars.copy()
    last_date = tampered.loc[tampered["ticker"] == tkr, "date"].max()
    mask = (tampered["ticker"] == tkr) & (tampered["date"] == last_date)
    tampered.loc[mask, "adj_close"] *= 1.10  # +10% shock to the future
    shocked = build_features(tampered)

    # No FEATURE anywhere may change when only a future price moved — not in
    # AAPL's own earlier rows (within-ticker causality), and not in any later
    # ticker (cross-ticker bleed). `target` is excluded: the penultimate AAPL
    # row's r_{t+1} legitimately uses the perturbed day, and that row is the only
    # place the shock may land.
    feature_cols = [c for c in base.columns if c != "target"]
    pd.testing.assert_frame_equal(base[feature_cols], shocked[feature_cols])


def test_illiquidity_is_inverse_dollar_volume():
    """Amihud illiquidity scales as 1 / dollar volume.

    Two tickers on an identical price path (so identical |returns|) but one trades
    at 10x the volume of the other. Since illiq = mean(|r| / (price*volume)) and
    only volume differs, the thin name's illiq_20 must be ~10x the liquid name's —
    a behavioural check of the factor, not a re-derivation of the formula.
    """
    n = 300  # > 252 warmup so rows survive the dropna
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = 100.0 * (1.01 ** np.arange(n))  # steady uptrend -> nonzero returns
    rows = []
    for ticker, vol in [("THIN", 1e5), ("LIQUID", 1e6)]:
        for d, p in zip(dates, price):
            rows.append({"date": d, "ticker": ticker, "adj_close": p, "volume": vol})
    feats = build_features(pd.DataFrame(rows))

    thin = feats.loc[feats["ticker"] == "THIN", "illiq_20"].to_numpy()
    liquid = feats.loc[feats["ticker"] == "LIQUID", "illiq_20"].to_numpy()
    assert (thin > liquid).all()                       # thinner name is more illiquid
    np.testing.assert_allclose(thin / liquid, 10.0)    # exactly inverse to volume


def test_no_nan_and_tidy_shape():
    """Output is clean and tidy: no NaNs, plain int index, target present."""
    parsed = build_features(parse_bars(load_bars()))
    assert len(parsed) > 0          # guard: a too-short fixture would empty out
    assert parsed.isna().sum().sum() == 0
    assert list(parsed.columns[:2]) == ["date", "ticker"]
    assert parsed.columns[-1] == "target"
    assert not parsed.duplicated(subset=["date", "ticker"]).any()
