"""The tradable equity universe.

The S&P 500 constituents (a liquid, well-defined, ~500-name large-cap universe)
plus SPY as the benchmark ETF. The list lives in a committed file `sp500.txt`,
generated once from Wikipedia's "List of S&P 500 companies" with symbols
dash-normalised for yfinance (e.g. BRK.B -> BRK-B).

⚠️ SURVIVORSHIP BIAS: this is TODAY's membership. Backtests over it only ever
see names that survived to now — delisted/bankrupt losers are absent, so results
skew optimistic. A point-in-time membership history would be needed to remove
this; treat broad-universe backtests as upper bounds until then.
"""

from pathlib import Path

_SP500_FILE = Path(__file__).with_name("sp500.txt")
BENCHMARK = "SPY"


def load_sp500():
    """Return the S&P 500 constituent tickers (yfinance-formatted)."""
    return _SP500_FILE.read_text().split()


def load_universe():
    """Constituents + the benchmark ETF (deduped, benchmark last)."""
    tickers = [t for t in load_sp500() if t != BENCHMARK]
    return tickers + [BENCHMARK]
