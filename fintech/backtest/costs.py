"""Trading cost model for the backtester (M5).

The simplest defensible model: a linear cost in basis points charged on the
notional traded at each rebalance. Commission + slippage, both in bps. Costs are
mandatory in M5 — a strategy that looks good gross but churns heavily must be
seen to lose once they bite (development_plan M5 success criterion).

1 bp = 0.01% = 1e-4. "Notional traded" for one rebalance is the turnover
sum(|w_t - w_prev|): every unit of weight that moves is a unit traded.
"""


def apply_costs(turnover, commission_bps=1.0, slippage_bps=5.0):
    """
    Cost of one rebalance, as a fraction of portfolio value.

    The number to subtract from that period's portfolio return.

    Args:
        turnover (float): sum(|w_t - w_prev|) for this rebalance — the fraction
            of the book traded (0 = no trade, 2 = fully flip a long book).
        commission_bps (float): broker commission per unit traded, in bps.
        slippage_bps (float): assumed slippage per unit traded, in bps.

    Returns:
        float: cost as a fraction of portfolio value (e.g. 0.0006 == 6 bps).
    """
    total_bps = commission_bps + slippage_bps
    return (total_bps / 10_000.0) * turnover  # 1 bp = 1e-4
