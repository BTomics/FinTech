"""Constrained mean-variance portfolio controller (M4).

Turns a vector of predicted returns into target portfolio weights via a convex
mean-variance optimisation under HARD constraints (development_plan M4). The
prediction source is decoupled on purpose: this takes expected returns `mu` as
input, so any M3 model (baseline or LightGBM) can feed it unchanged.

Objective (risk-aversion form): maximise  mu @ w - (risk_aversion / 2) * w @ Σ @ w
subject to the constraints below. It's a convex QP — cvxpy + a deterministic
solver gives reproducible weights for fixed inputs.

Constraints (M4 success criterion — a breaching vector must be clamped/rejected):
  - long-only:        w >= 0
  - max weight/asset: w <= max_weight
  - cash floor:       sum(w) <= 1 - cash_floor   (the rest is held as cash)
  - max turnover:     sum(|w - w_prev|) <= max_turnover
"""

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def estimate_covariance(returns):
    """
    Ledoit-Wolf shrunk covariance of asset returns.

    Shrinkage pulls the noisy sample covariance toward a structured target so the
    matrix is well-conditioned (invertible, stable) even when assets are few or
    highly correlated — the regime where raw sample covariance wrecks MV weights.

    Args:
        returns (pd.DataFrame): wide returns, one column per ticker, one row per
            date (no NaN). Column order defines the asset order downstream.

    Returns:
        pd.DataFrame: covariance matrix, indexed and columned by ticker (same
            order as `returns.columns`).
    """
    lw = LedoitWolf().fit(returns.to_numpy())
    return pd.DataFrame(
        lw.covariance_, index=returns.columns, columns=returns.columns
    )


def optimize_weights(mu, cov, w_prev=None, risk_aversion=1.0,
                     max_weight=1.0, max_turnover=None, cash_floor=0.0):
    """
    Solve the constrained mean-variance problem for target weights.

    Args:
        mu (pd.Series): expected next-period return per ticker (the M3 prediction,
            aggregated to one number per asset). Its index is the asset order.
        cov (pd.DataFrame): covariance aligned to `mu`'s index (estimate_covariance
            output).
        w_prev (pd.Series | None): current weights for the turnover term; treat
            None as all-cash (zeros).
        risk_aversion (float): lambda on the variance penalty; higher = more
            risk-averse.
        max_weight (float): per-asset cap.
        max_turnover (float | None): cap on sum(|w - w_prev|); None = no cap.
        cash_floor (float): minimum fraction held as cash (0..1).

    Returns:
        pd.Series: target weights indexed like `mu`, satisfying every constraint.
    """
    # mu defines the canonical asset order; align cov (and w_prev) to it so the
    # numpy arrays handed to cvxpy line up — implicit pandas alignment is gone the
    # moment we drop to .to_numpy(), so do it explicitly here.
    tickers = list(mu.index)
    mu_vec = mu.to_numpy()
    sigma = cov.loc[tickers, tickers].to_numpy()
    sigma = (sigma + sigma.T) / 2  # kill any float asymmetry before cvxpy

    if w_prev is None:
        w_prev_vec = np.zeros(len(tickers))
    else:
        w_prev_vec = w_prev.reindex(tickers).fillna(0.0).to_numpy()

    w = cp.Variable(len(tickers))
    objective = cp.Maximize(
        mu_vec @ w - (risk_aversion / 2) * cp.quad_form(w, cp.psd_wrap(sigma))
    )
    constraints = [
        w >= 0,                       # long-only
        w <= max_weight,              # per-asset cap
        cp.sum(w) <= 1 - cash_floor,  # leave at least cash_floor in cash
    ]
    if max_turnover is not None:
        constraints.append(cp.norm1(w - w_prev_vec) <= max_turnover)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)  # fixed solver -> deterministic output
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"optimizer did not converge: status={problem.status}")

    return pd.Series(w.value, index=tickers)
