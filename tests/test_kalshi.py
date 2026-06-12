"""Tests for fintech.data.kalshi.

The network call is faked with monkeypatch so no test ever touches the real
API — fetch_series_markets's logic (unwrapping, truncation guard) is what's
under test, not Kalshi's servers.
"""

import pytest

from fintech.data import kalshi


class FakeResponse:
    def __init__(self, markets):
        self._markets = markets

    def raise_for_status(self):
        pass

    def json(self):
        return {"markets": self._markets, "cursor": ""}


def test_returns_market_list(monkeypatch):
    fake_markets = [{"ticker": "KXFED-A"}, {"ticker": "KXFED-B"}]
    monkeypatch.setattr(kalshi.requests, "get", lambda *a, **k: FakeResponse(fake_markets))
    assert kalshi.fetch_series_markets("KXFED") == fake_markets

def test_page_limit_raises(monkeypatch):
    truncated = [{"ticker": f"M{i}"} for i in range(kalshi.PAGE_LIMIT)]
    monkeypatch.setattr(kalshi.requests, "get", lambda *a, **k: FakeResponse(truncated))
    with pytest.raises(RuntimeError):
        kalshi.fetch_series_markets("KXFED")
