"""M9 A/B: the composite+MV book WITH vs WITHOUT the risk overlay.

Reuses scripts/backtest_compare.py wholesale (same μ, same costs, same OPT_KWARGS,
same dates) so the ONLY difference is the overlay. Compares, side by side:
  - overlay OFF                  (the live composite+MV book)
  - vol-target ON (trailing-cov) (k = clip(σ_target/σ_realized, 0, k_max))
  - vol-target ON (GARCH)        (same, but σ_realized from a GARCH(1,1) forecast)
  - CPPI floor                   (exposure ∝ cushion above a high-water-mark floor)
Metrics: CAGR / Sharpe / Sortino / MaxDD / Calmar / AvgInvested + logged daily
95% VaR / Expected-Shortfall. Reporting script, not a test.

    .venv\\Scripts\\python.exe -m scripts.backtest_risk_overlay

Honest bar (see plan): ship to live ONLY if MaxDD drops materially AND
Sharpe/Sortino/Calmar are at least flat. Cutting return alone is a valid "no".
"""

import numpy as np
import pandas as pd

from fintech.backtest import metrics
from fintech.backtest.costs import apply_costs
from fintech.backtest.engine import generate_mv_weights, simulate
from fintech.portfolio.optimizer import estimate_covariance
from fintech.risk.overlay import (
    apply_vol_target,
    cppi_exposure,
    expected_shortfall,
    historical_var,
    realized_portfolio_vol,
    vol_target_scale,
)
from scripts.backtest_compare import (
    COMMISSION_BPS,
    IMPACT_BPS,
    LOOKBACK,
    OPT_KWARGS,
    REBALANCE_EVERY,
    SLIPPAGE_BPS,
    _composite_mu,
    _wide_returns,
)

SIGMA_TARGET = 0.12     # target annualised portfolio vol
K_MAX = 1.0             # long-only, no leverage
CPPI_FLOOR_FRAC = 0.80  # protect 80% of the running high-water mark
CPPI_MULTIPLIER = 4.0
VAR_ALPHA = 0.95
COSTS = dict(commission_bps=COMMISSION_BPS, slippage_bps=SLIPPAGE_BPS, impact_bps=IMPACT_BPS)


def _garch_annual_vol(book_returns):
    """Next-day GARCH(1,1) conditional vol of a book-return series, annualised.

    arch fits best on percentage-scale returns, so scale up for the fit and back
    down for the forecast. Falls back to the trailing std on a short/failed fit.
    """
    from arch import arch_model

    r = pd.Series(book_returns).dropna() * 100.0
    if len(r) < 30 or r.std() == 0:
        return float(pd.Series(book_returns).std() * np.sqrt(metrics.PERIODS_PER_YEAR))
    try:
        res = arch_model(r, vol="Garch", p=1, q=1, mean="Constant", dist="normal").fit(disp="off")
        daily_var_pct = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
        daily_vol = np.sqrt(daily_var_pct) / 100.0
    except Exception:  # noqa: BLE001 — a non-converging fit shouldn't abort the run
        daily_vol = float(r.std()) / 100.0
    return float(daily_vol * np.sqrt(metrics.PERIODS_PER_YEAR))


def _overlay_weights(mv_weights, returns, vol_method="trailing"):
    """Vol-target overlay as a post-processing pass over an MV weights path.

    For each rebalance date t: estimate the current book's realised vol from the
    trailing window ENDING AT t (causal, same window as generate_mv_weights), get
    k = clip(σ_target/σ_realised, 0, k_max), and store k·w_t (the rest is cash).
    """
    scaled = mv_weights.copy()
    for t in mv_weights.index:
        w = mv_weights.loc[t]
        book = w[w != 0]
        if book.empty:
            continue
        window = returns.loc[:t].iloc[-LOOKBACK:][book.index].dropna(axis=1)
        book = book[window.columns]
        if book.empty:
            continue
        if vol_method == "garch":
            vol = _garch_annual_vol((window * book).sum(axis=1))
        else:  # trailing covariance -> sqrt(wᵀΣw)
            vol = realized_portfolio_vol(book, estimate_covariance(window))
        k = vol_target_scale(vol, SIGMA_TARGET, K_MAX)
        scaled.loc[t] = mv_weights.loc[t] * k
    return scaled


def simulate_cppi(base_net):
    """Path-dependent CPPI overlay on the fully-invested book's net returns.

    Walks the equity curve daily: floor ratchets to CPPI_FLOOR_FRAC of the
    high-water mark, exposure = cppi_exposure(value, floor, m), so the book
    auto-de-risks as it nears the floor. Charges the incremental cost of changing
    exposure |k_t - k_{t-1}| (the book's own rebalancing cost is already inside
    base_net). Returns (net returns series, average exposure).
    """
    value, peak, k_prev = 1.0, 1.0, 0.0
    out = pd.Series(0.0, index=base_net.index)
    k_sum = 0.0
    for t, r in base_net.items():
        floor = CPPI_FLOOR_FRAC * peak
        k = cppi_exposure(value, floor, CPPI_MULTIPLIER, K_MAX)
        cost = apply_costs(abs(k - k_prev), **COSTS)
        day_ret = k * r - cost
        value *= (1.0 + day_ret)
        peak = max(peak, value)
        out[t] = day_ret
        k_sum += k
        k_prev = k
    return out, k_sum / len(base_net)


def _summary(net, avg_invested):
    return {
        "CAGR": metrics.cagr(net),
        "Sharpe": metrics.sharpe(net),
        "Sortino": metrics.sortino(net),
        "MaxDD": metrics.max_drawdown(net),
        "Calmar": metrics.calmar(net),
        "AvgInv": avg_invested,
        "VaR95d": historical_var(net, VAR_ALPHA),
        "ES95d": expected_shortfall(net, VAR_ALPHA),
    }


def main():
    bars = pd.read_parquet("data/processed/bars.parquet")
    returns = _wide_returns(bars)
    mu = _composite_mu(bars, returns.columns)

    mv = generate_mv_weights(returns, mu, lookback=LOOKBACK,
                             rebalance_every=REBALANCE_EVERY, **OPT_KWARGS)
    mv_trailing = _overlay_weights(mv, returns, vol_method="trailing")
    mv_garch = _overlay_weights(mv, returns, vol_method="garch")

    start = mv.index[0]
    base_net = simulate(mv, returns, **COSTS).loc[start:]
    cppi_net, cppi_avg = simulate_cppi(base_net)

    rows = {
        "overlay OFF": _summary(base_net, mv.sum(axis=1).mean()),
        "vol-target (trailing)": _summary(
            simulate(mv_trailing, returns, **COSTS).loc[start:], mv_trailing.sum(axis=1).mean()),
        "vol-target (GARCH)": _summary(
            simulate(mv_garch, returns, **COSTS).loc[start:], mv_garch.sum(axis=1).mean()),
        "CPPI floor": _summary(cppi_net, cppi_avg),
    }

    table = pd.DataFrame(rows).T
    print(f"\nRisk-overlay A/B  (sigma_target={SIGMA_TARGET:.0%}, k_max={K_MAX}, "
          f"CPPI floor={CPPI_FLOOR_FRAC:.0%}*HWM x{CPPI_MULTIPLIER:g})")
    print(f"Eval: {start.date()} -> {returns.index[-1].date()}  "
          f"({len(base_net)} days), costs={COMMISSION_BPS+SLIPPAGE_BPS:.0f}bps"
          f"+{IMPACT_BPS:.0f}bps*turnover^2\n")
    with pd.option_context("display.float_format", lambda x: f"{x:0.4f}",
                           "display.width", 200, "display.max_columns", None):
        print(table)
    print()


if __name__ == "__main__":
    main()
