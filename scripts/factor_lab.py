"""Factor lab: screen candidate signals by standalone IC + decorrelation.

The lambda calibration showed the composite's edge lives in its top ~10 names —
so confidence-grading beyond that is noise UNLESS the signal has real conviction
deeper in the cross-section. The way to earn that is richer, *decorrelated*
factors. This is the bench for deciding which candidates are worth adding.

Two questions per candidate, both required to keep it:
  1. Does it rank winners above losers?  -> mean daily rank IC (and a t-stat, so
     we don't chase a factor whose IC is positive but indistinguishable from 0).
  2. Is it NEW?  -> low correlation to the factors already in the composite. A
     great-IC factor that's 0.9-correlated with momentum adds nothing.

A candidate earns a slot only if it has a real (t-stat > ~2) IC AND is
meaningfully decorrelated from what we already trade.

    .venv\\Scripts\\python.exe -m scripts.factor_lab

NOTE: run as a module (-m) so the repo root is importable (script-to-script
imports fail when run as a bare file).
"""

import numpy as np
import pandas as pd

from fintech.features.build import build_features
from fintech.models.validation import information_coefficient

BARS_PATH = "data/processed/bars.parquet"

# Candidate factors to screen. Each entry maps a NAME to a callable
# features -> pd.Series (the factor value per row, aligned to features.index).
#
# Encode the SIGN here so "higher = buy": short-term reversal means low recent
# return is bullish, so it's -ret_lag1. (A factor with strong NEGATIVE IC is
# still useful — just negate it.)
#
# The first two are the signals already in the live composite, shown as the
# calling convention / baseline to beat. Add your candidates below them.
CANDIDATES = {
    "mom_252_21": lambda f: f["mom_252_21"],     # 12-1 momentum  (in composite)
    "reversal_1d": lambda f: -f["ret_lag1"],     # short reversal (in composite)
    # TODO: add candidates to screen, e.g.
    #   "low_vol":      lambda f: -f["vol_20"]        (low-volatility anomaly)
    #   "illiquidity":  lambda f: ... (Amihud-style, from rvol / returns)
    #   "reversal_5d":  lambda f: -f["ret_lag5"]      (longer reversal)
    #   "mom_126":      lambda f: f["mom_126"]        (6-month momentum)
}


def daily_rank_ic(features, signal):
    """Per-date cross-sectional Spearman rank IC of `signal` vs the target.

    Like information_coefficient (same definition, same NaN-day handling) but
    returns the WHOLE daily series, not just its mean — we need the dispersion to
    compute a t-stat. Index is date; one IC per day with >= 2 ranked names.

    Args:
        features (pd.DataFrame): build_features output (has `date`, `target`).
        signal (pd.Series): factor values aligned to features.index.

    Returns:
        pd.Series: daily rank IC, indexed by date, NaN days dropped.
    """
    # TODO: per date, spearman corr between `signal` and features["target"].
    #   Mirror information_coefficient's grouping; return the daily series
    #   (drop NaN days) instead of .mean(). The mean of this == information_coefficient.
    raise NotImplementedError


def ic_table(features, candidates):
    """Mean IC, IC t-stat and #days for each candidate factor.

    The t-stat treats the daily IC series as ~iid draws: t = mean / (std / sqrt(n)).
    A |t| above ~2 means the edge is unlikely to be noise. Sort by t-stat so the
    real signals float to the top.

    Args:
        features (pd.DataFrame): build_features output.
        candidates (dict[str, callable]): name -> (features -> factor Series).

    Returns:
        pd.DataFrame: index = factor name; columns = [mean_ic, ic_t, n_days];
            sorted by ic_t descending.
    """
    # TODO: for each candidate, signal = fn(features); ic = daily_rank_ic(...);
    #   row = {mean_ic: ic.mean(), ic_t: ic.mean()/ic.std()*sqrt(len(ic)),
    #          n_days: len(ic)}. Assemble -> DataFrame, sort by ic_t desc.
    raise NotImplementedError


def factor_correlation(features, candidates):
    """Cross-sectional correlation matrix of the candidate factor VALUES.

    Decorrelation check: two factors that rank the universe the same way carry
    the same bet. Compute the correlation of the factor *values* across the panel
    (NOT their ICs). Spearman/rank is the natural choice since the allocator only
    uses ordering. Mean-of-daily or pooled is your call — document which you pick.

    Args:
        features (pd.DataFrame): build_features output.
        candidates (dict[str, callable]): name -> (features -> factor Series).

    Returns:
        pd.DataFrame: square correlation matrix, indexed/columned by factor name.
    """
    # TODO: build a DataFrame of factor columns {name: fn(features)}, then either
    #   .corr(method="spearman") pooled, or average a per-date rank corr. Keep the
    #   diagonal 1.0; off-diagonals near 0 = the factors are independent bets.
    raise NotImplementedError


def main():
    bars = pd.read_parquet(BARS_PATH)
    features = build_features(bars)

    print(f"\nFactor lab — {len(CANDIDATES)} candidates, "
          f"{features['date'].nunique()} dates, {features['ticker'].nunique()} tickers\n")

    with pd.option_context("display.float_format", lambda x: f"{x:0.4f}",
                           "display.width", 200, "display.max_columns", None):
        print("Standalone IC (sorted by t-stat):")
        print(ic_table(features, CANDIDATES))
        print("\nFactor-value correlation (decorrelation check):")
        print(factor_correlation(features, CANDIDATES))
    print()


if __name__ == "__main__":
    main()
