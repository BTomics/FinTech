import json
import requests

response = requests.get(
    "https://gamma-api.polymarket.com/markets",
    params={"limit": 5},
)
response.raise_for_status()  # crash loudly if the API didn't return 200
markets = response.json()

with open("tests/fixtures/polymarket_markets_sample.json", "w", encoding="utf-8") as f:
    json.dump(markets, f, indent=2)

print(f"Saved {len(markets)} markets")