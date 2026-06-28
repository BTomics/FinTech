# FinTech: Centralized Quantitative Trading System

A personal quant system: data ingestion (market bars, Polymarket/Kalshi prediction markets, news sentiment), a prediction pipeline, a constrained portfolio controller, and hard risk management — built classical-first, deep learning later.

## Architecture

Four phases, built as a single end-to-end vertical slice and then deepened:

1. **Data ingestion** — daily equity bars (yfinance), prediction-market snapshots (Polymarket/Kalshi), with raw/processed separation and idempotent upserts.
2. **Prediction pipeline** — strictly-causal feature engineering and walk-forward validation; signals scored by cross-sectional Information Coefficient, not just MSE.
3. **Portfolio controller** — constrained mean-variance optimisation (cvxpy) over a momentum + reversal + illiquidity factor composite, with Ledoit-Wolf covariance shrinkage.
4. **Risk & execution** — convex cost model (commission/slippage/market-impact), cost-aware backtest vs. equal-weight and buy-and-hold benchmarks, and live paper trading via Alpaca.

The guiding principle throughout: **no look-ahead bias**, and every added complexity must beat the simple baseline net of realistic costs or it's removed.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then fill in keys (only Kalshi needed for M1)
pytest
```

## Layout

```
fintech/
  data/        Phase 1 — feed clients & loggers (Polymarket, Kalshi, market bars)
  features/    Phase 1/2 — alignment & feature engineering
  models/      Phase 2 — prediction pipeline
  portfolio/   Phase 3 — portfolio controller
  risk/        Phase 4 — risk constraints & monitoring
  backtest/    cost-aware evaluation vs. baselines
data/
  raw/         immutable API responses (gitignored, never re-fetchable — don't delete)
  processed/   derived frames (gitignored, always re-derivable)
tests/
```

> **Note:** this repo lives inside OneDrive. `data/raw/` will grow as the feed loggers run; if OneDrive sync becomes a problem, exclude this folder from sync or move the data directory elsewhere via config.
