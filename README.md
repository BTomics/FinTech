# FinTech: Centralized Quantitative Trading System

A personal quant system: data ingestion (market bars, Polymarket/Kalshi prediction markets, news sentiment), a prediction pipeline, a constrained portfolio controller, and hard risk management — built classical-first, deep learning later.

## Documents

- [`research_and_architecture_plan.md`](research_and_architecture_plan.md) — theory and algorithms per phase
- [`development_plan.md`](development_plan.md) — milestones, success criteria, build order
- [`fintech_coding_guidelines.md`](fintech_coding_guidelines.md) — coding rules (lookahead bias, simplicity, secrets, reproducibility)

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
