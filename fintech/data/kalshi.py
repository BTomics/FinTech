"""Kalshi API: fetching raw market data for the macro watchlist.

Market-data reads on Kalshi's public API need no authentication, so this
module is a thin unauthenticated client. Parsing into tidy frames is
deferred to M7 — only raw logging happens for now.
"""

import requests

KALSHI_API_URL = "https://api.elections.kalshi.com/trade-api/v2/markets"

# One page is plenty: our watchlist series each have far fewer open markets.
# Hitting this limit means the response was probably truncated.
PAGE_LIMIT = 1000


def fetch_series_markets(series_ticker):
    """
    Fetch all open markets for one Kalshi series (e.g. "KXFED").

    Returns:
        list[dict]: the raw market dicts from the API, untouched.
    Raises:
        requests.HTTPError: on a non-200 response.
        RuntimeError: if the page limit is hit (pagination not implemented,
            and silently logging a truncated snapshot would corrupt history).
    """
    response = requests.get(
        KALSHI_API_URL,
        params={"series_ticker": series_ticker, "status": "open", "limit": PAGE_LIMIT},
    )
    response.raise_for_status()
    markets = response.json()["markets"]
    if len(markets) >= PAGE_LIMIT:
        raise RuntimeError(
            f"{series_ticker}: got {len(markets)} markets, page limit hit — implement pagination"
        )
    return markets
