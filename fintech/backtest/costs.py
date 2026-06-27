"""Trading cost model for the backtester (M5/M7).

Two parts:
  - LINEAR (commission + slippage, in bps): a flat cost per unit of notional
    traded — fine for small trades.
  - CONVEX market impact (impact_bps, in bps): your own trades push the price
    against you, and the push grows with trade size, so cost per unit RISES with
    turnover. Modelled as a quadratic term — the standard cheap proxy for
    Almgren-style impact. This is what makes a high-turnover strategy pay for its
    churn instead of getting away with a flat fee.

1 bp = 0.01% = 1e-4. "Notional traded" for one rebalance is the turnover
sum(|w_t - w_prev|): every unit of weight that moves is a unit traded.
"""


def apply_costs(turnover, commission_bps=1.0, slippage_bps=5.0, impact_bps=0.0):
    """
    Cost of one rebalance, as a fraction of portfolio value.

    The number to subtract from that period's portfolio return.

    Args:
        turnover (float): sum(|w_t - w_prev|) for this rebalance — the fraction
            of the book traded (0 = no trade, 2 = fully flip a long book).
        commission_bps (float): broker commission per unit traded, in bps.
        slippage_bps (float): assumed slippage per unit traded, in bps.
        impact_bps (float): convex market-impact coefficient, in bps; cost from
            this term scales with turnover SQUARED (big rebalances cost more per
            unit). 0 recovers the pure linear model.

    Returns:
        float: cost as a fraction of portfolio value (e.g. 0.0006 == 6 bps).
    """
    linear = (commission_bps + slippage_bps) / 10_000.0 * turnover  # 1 bp = 1e-4
    impact = impact_bps / 10_000.0 * turnover ** 2
    return linear + impact
