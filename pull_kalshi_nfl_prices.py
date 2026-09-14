"""
Pull Kalshi NFL "yes" contract price history for both teams in every settled
KXNFLGAME event, at 1-minute intervals from kickoff to game end, into a CSV.

No Kalshi account or API key needed - these are public, unauthenticated GET
endpoints on Kalshi's Trade API v2.

Run locally (needs open internet access to external-api.kalshi.com):
    pip install requests
    python pull_kalshi_nfl_prices.py

Output:
    kalshi_nfl_prices.csv  in the same folder, with columns:
        event_ticker, game_date, team_ticker, side, end_period_ts,
        datetime_utc, yes_price_open, yes_price_close, yes_price_mean

Notes / limitations to know before you run this:
- Kalshi's NFL game-winner markets (KXNFLGAME series) only go back a few
  seasons, not the full nfl-data-py history. This will NOT cover "all NFL
  games ever" - only the subset with a Kalshi market.
- A single candlesticks request is capped at 5000 bars. At 1-minute
  resolution that's ~83 hours, way more than one game window, so this is
  not an issue per-game - it's only a concern if you widen the interval
  logic yourself.
- Recently settled markets come from the live endpoint; older ones need
  the /historical/ endpoint. This script tries live first and falls back
  to historical automatically.
- Rate limiting: this script sleeps briefly between calls to stay well
  under Kalshi's public rate limits. Pulling multiple seasons will take a
  while - let it run.
"""

import csv
import time
import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = "KXNFLGAME"
OUT_CSV = "kalshi_nfl_prices.csv"
SLEEP_BETWEEN_CALLS = 0.25  # be polite to the public API


def get_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  [warn] {resp.status_code} on {url} params={params}")
                return None
        except requests.RequestException as e:
            print(f"  [warn] request error: {e}")
            time.sleep(2 * (attempt + 1))
    return None


def get_all_settled_events():
    """Page through every settled KXNFLGAME event."""
    events = []
    cursor = None
    while True:
        params = {
            "series_ticker": SERIES_TICKER,
            "status": "settled",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        data = get_json(f"{BASE}/events", params=params)
        if not data:
            break
        batch = data.get("events", [])
        events.extend(batch)
        cursor = data.get("cursor")
        print(f"  fetched {len(batch)} events (total so far: {len(events)})")
        if not cursor or not batch:
            break
        time.sleep(SLEEP_BETWEEN_CALLS)
    return events


def get_event_markets(event_ticker):
    """Get markets for an event, falling back to the historical endpoint
    for events whose markets settled before Kalshi's live-data cutoff
    (currently ~3 months). Without this fallback, GET /events/{ticker}
    silently returns an empty "markets" list for older events."""
    data = get_json(f"{BASE}/events/{event_ticker}")
    if data and data.get("markets"):
        return data["markets"]

    # live endpoint returned nothing usable - try the historical one
    hist_data = get_json(f"{BASE}/historical/markets", params={
        "event_ticker": event_ticker,
        "limit": 200,
    })
    if hist_data:
        return hist_data.get("markets", [])
    return []


def get_candlesticks(ticker, series_ticker, start_ts, end_ts, period_interval=1):
    """Try the live candlesticks endpoint, then fall back to /historical/."""
    url = f"{BASE}/series/{series_ticker}/markets/{ticker}/candlesticks"
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
    }
    data = get_json(url, params=params)
    if data and data.get("candlesticks"):
        return data["candlesticks"]

    # fall back to historical endpoint for older settled markets
    hist_url = f"{BASE}/historical/markets/{ticker}/candlesticks"
    data = get_json(hist_url, params=params)
    if data:
        return data.get("candlesticks", [])
    return []


def main():
    print("Fetching list of settled NFL events from Kalshi (public, no auth)...")
    events = get_all_settled_events()
    print(f"Found {len(events)} settled NFL events.\n")

    rows = []
    for i, event in enumerate(events, 1):
        event_ticker = event.get("event_ticker")
        print(f"[{i}/{len(events)}] {event_ticker}")

        markets = get_event_markets(event_ticker)
        if len(markets) < 2:
            print("  [skip] fewer than 2 markets found for this event")
            continue

        # open_time / close_time on the market bracket kickoff -> settlement
        for market in markets:
            ticker = market.get("ticker")
            side = market.get("yes_sub_title") or market.get("subtitle") or ticker
            open_ts = market.get("open_time")
            close_ts = market.get("close_time")
            if not (ticker and open_ts and close_ts):
                continue

            # open_time is when the market opened (days before kickoff), so
            # narrow the window to roughly the game itself: use the last ~5
            # hours before close_time as a proxy for kickoff-to-final-whistle.
            import datetime as dt
            close_dt = dt.datetime.fromisoformat(close_ts.replace("Z", "+00:00"))
            start_dt = close_dt - dt.timedelta(hours=5)
            start_ts = int(start_dt.timestamp())
            end_ts = int(close_dt.timestamp())

            candles = get_candlesticks(ticker, SERIES_TICKER, start_ts, end_ts, 1)
            time.sleep(SLEEP_BETWEEN_CALLS)

            for c in candles:
                price = c.get("price", {})
                rows.append({
                    "event_ticker": event_ticker,
                    "game_date": close_dt.date().isoformat(),
                    "team_ticker": ticker,
                    "side": side,
                    "end_period_ts": c.get("end_period_ts"),
                    "datetime_utc": dt.datetime.utcfromtimestamp(
                        c.get("end_period_ts", 0)
                    ).isoformat() if c.get("end_period_ts") else None,
                    "yes_price_open": price.get("open_dollars") or price.get("open"),
                    "yes_price_close": price.get("close_dollars") or price.get("close"),
                    "yes_price_mean": price.get("mean_dollars") or price.get("mean"),
                })

        # write incrementally so partial progress isn't lost on interruption
        if rows:
            with open(OUT_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    print(f"\nDone. Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
