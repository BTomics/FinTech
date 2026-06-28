"""Contracts for the factor lab (scripts/factor_lab.py).

These are written to FAIL until daily_rank_ic / ic_table / factor_correlation are
implemented — they pin the behaviour each function must satisfy. Implement until
green. The frames are tiny and hand-checked so the expected IC/correlation values
are obvious by eye.
"""
import numpy as np
import pandas as pd
import pytest

from fintech.models.validation import information_coefficient
from scripts.factor_lab import daily_rank_ic, ic_table, factor_correlation


def _two_day_frame():
    """2 dates x 3 tickers. `sig` ranks WITH the target on day 1 (IC +1) and
    AGAINST it on day 2 (IC -1) -> mean daily IC is exactly 0, daily series [1,-1].
    Non-trivial on purpose, so a mean-vs-information_coefficient check isn't 1==1."""
    d1, d2 = pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")
    rows = [
        {"date": d1, "ticker": "A", "target": 0.01, "sig": 1.0},
        {"date": d1, "ticker": "B", "target": 0.02, "sig": 2.0},
        {"date": d1, "ticker": "C", "target": 0.03, "sig": 3.0},
        {"date": d2, "ticker": "A", "target": 0.01, "sig": 3.0},
        {"date": d2, "ticker": "B", "target": 0.02, "sig": 2.0},
        {"date": d2, "ticker": "C", "target": 0.03, "sig": 1.0},
    ]
    return pd.DataFrame(rows)


def _xs_frame():
    """3 dates x 3 tickers, target strictly ordered A < B < C every day."""
    rows = []
    for d in pd.date_range("2021-01-01", periods=3):
        for ticker, tgt in zip(["A", "B", "C"], [0.01, 0.02, 0.03]):
            rows.append({"date": d, "ticker": ticker, "target": tgt})
    return pd.DataFrame(rows)


# ---- daily_rank_ic -------------------------------------------------------

def test_daily_rank_ic_is_per_date_series():
    f = _two_day_frame()
    ic = daily_rank_ic(f, f["sig"])
    assert len(ic) == 2                                   # one value per date
    assert set(pd.to_datetime(ic.index)) == set(f["date"])
    assert ic.loc[pd.Timestamp("2021-01-01")] == pytest.approx(1.0)
    assert ic.loc[pd.Timestamp("2021-01-02")] == pytest.approx(-1.0)


def test_daily_rank_ic_mean_matches_information_coefficient():
    # The whole point of the separate series: its mean must equal the existing
    # information_coefficient (same definition, just not pre-averaged).
    f = _two_day_frame()
    assert daily_rank_ic(f, f["sig"]).mean() == pytest.approx(
        information_coefficient(f, f["sig"])
    )
    assert daily_rank_ic(f, f["sig"]).mean() == pytest.approx(0.0)


# ---- ic_table ------------------------------------------------------------

def test_ic_table_structure():
    f = _xs_frame()
    cands = {"perfect": lambda x: x["target"], "reversed": lambda x: -x["target"]}
    table = ic_table(f, cands)
    assert set(table.index) == set(cands)
    assert list(table.columns) == ["mean_ic", "ic_t", "n_days"]


def test_ic_table_ranks_and_signs():
    f = _xs_frame()
    cands = {"perfect": lambda x: x["target"], "reversed": lambda x: -x["target"]}
    table = ic_table(f, cands)
    assert table.loc["perfect", "mean_ic"] == pytest.approx(1.0)
    assert table.loc["reversed", "mean_ic"] == pytest.approx(-1.0)
    # sorted by t-stat descending -> the positive signal is on top.
    assert table.index[0] == "perfect"


def test_ic_table_mean_ic_matches_information_coefficient():
    f = _two_day_frame()
    cands = {"sig": lambda x: x["sig"]}
    table = ic_table(f, cands)
    assert table.loc["sig", "mean_ic"] == pytest.approx(
        information_coefficient(f, f["sig"])
    )


# ---- factor_correlation --------------------------------------------------

def test_factor_correlation_structure_and_diagonal():
    f = _xs_frame()
    cands = {"a": lambda x: x["target"], "b": lambda x: -x["target"]}
    corr = factor_correlation(f, cands)
    assert list(corr.index) == list(corr.columns) == ["a", "b"]
    assert corr.loc["a", "a"] == pytest.approx(1.0)
    assert corr.loc["b", "b"] == pytest.approx(1.0)


def test_factor_correlation_identical_and_opposite():
    # Robust to the pooled-vs-mean-of-daily choice: identical factors correlate
    # +1, exactly-opposite factors -1, under either convention.
    f = _xs_frame()
    cands = {
        "x": lambda d: d["target"],
        "same": lambda d: d["target"] * 10 + 3,   # monotonic -> same ranks
        "opp": lambda d: -d["target"],
    }
    corr = factor_correlation(f, cands)
    assert corr.loc["x", "same"] == pytest.approx(1.0)
    assert corr.loc["x", "opp"] == pytest.approx(-1.0)
