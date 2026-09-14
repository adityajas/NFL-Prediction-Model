"""
Kalshi NFL Game-Contract Trading Bot
=====================================

Pulls live YES/NO prices for both sides of an NFL moneyline-style event
market on Kalshi, applies a simple trading strategy, and places (or
paper-trades) orders based on that signal.

IMPORTANT
---------
- Defaults to PAPER TRADING (no real orders sent). Set PAPER_TRADING=False
  only after you've watched it run and understand the logic.
- This is a template, not a proven strategy. Kalshi event contracts are
  binary (settle at $1 or $0), defined-risk instruments — you can still
  lose your whole stake on a position. Past behavior of a "strategy" here
  is not evidence it will be profitable going forward.
- You need a Kalshi account + API key (Settings > API Keys in the Kalshi
  app/site) which gives you a Key ID and an RSA private key (.pem file).
- Robinhood does NOT currently expose a public trading API for its event
  contracts, so this script targets Kalshi directly. If Robinhood (or any
  other venue) publishes one later, you'd only need to write a new class
  matching the same interface as KalshiClient below.

Install deps:
    pip install requests cryptography python-dotenv --break-system-packages
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"  # prod
KALSHI_DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"        # sandbox

USE_DEMO = True            # start on the sandbox/demo environment
PAPER_TRADING = True       # if True, never sends real orders even on prod

KALSHI_KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")

POLL_INTERVAL_SECONDS = 15
MAX_POSITION_CONTRACTS = 10     # hard cap on contracts held per side
MAX_DOLLARS_AT_RISK = 50.00     # hard cap on total capital committed
ENTRY_EDGE_CENTS = 4             # min mispricing (in cents) required to trade
EXIT_PROFIT_CENTS = 6            # take-profit threshold
STOP_LOSS_CENTS = 10             # stop-loss threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("kalshi_nfl_bot")


# --------------------------------------------------------------------------
# KALSHI CLIENT (REST + RSA-PSS request signing)
# --------------------------------------------------------------------------

class KalshiClient:
    """Minimal Kalshi REST client: signed GET/POST, market data, orders."""

    def __init__(self, key_id: str, private_key_path: str, use_demo: bool = True):
        if not key_id or not private_key_path:
            raise ValueError(
                "Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH env vars "
                "(from Kalshi > Settings > API Keys)."
            )
        self.key_id = key_id
        self.base_url = KALSHI_DEMO_URL if use_demo else KALSHI_BASE_URL
        with open(private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.session = requests.Session()

    def _sign(self, method: str, path: str) -> dict:
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None):
        # path used for signing must be the path only (no query string), matching
        # Kalshi's documented signing scheme.
        headers = self._sign(method, path)
        url = self.base_url + path
        resp = self.session.request(method, url, headers=headers, params=params, json=body, timeout=10)
        if not resp.ok:
            raise RuntimeError(f"Kalshi API error {resp.status_code}: {resp.text}")
        return resp.json()

    # ---- market data ----

    def get_events(self, series_ticker: str = "KXNFLGAME", status: str = "open") -> list[dict]:
        """List NFL game events (each event has YES/NO markets for the matchup)."""
        data = self._request("GET", "/events", params={"series_ticker": series_ticker, "status": status})
        return data.get("events", [])

    def get_markets_for_event(self, event_ticker: str) -> list[dict]:
        data = self._request("GET", "/markets", params={"event_ticker": event_ticker})
        return data.get("markets", [])

    def get_market(self, ticker: str) -> dict:
        data = self._request("GET", f"/markets/{ticker}")
        return data["market"]

    def get_orderbook(self, ticker: str) -> dict:
        data = self._request("GET", f"/markets/{ticker}/orderbook")
        return data.get("orderbook", {})

    # ---- trading ----

    def get_positions(self) -> list[dict]:
        data = self._request("GET", "/portfolio/positions")
        return data.get("market_positions", [])

    def get_balance(self) -> float:
        data = self._request("GET", "/portfolio/balance")
        return data.get("balance", 0) / 100.0  # cents -> dollars

    def place_order(self, ticker: str, side: str, action: str, count: int, price_cents: int) -> dict:
        """
        side: 'yes' or 'no'
        action: 'buy' or 'sell'
        price_cents: limit price for the given side, 1-99
        """
        body = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "side": side,
            "action": action,
            "count": count,
            "type": "limit",
            "yes_price": price_cents if side == "yes" else None,
            "no_price": price_cents if side == "no" else None,
        }
        body = {k: v for k, v in body.items() if v is not None}
        return self._request("POST", "/portfolio/orders", body=body)


# --------------------------------------------------------------------------
# STRATEGY
# --------------------------------------------------------------------------

@dataclass
class Quote:
    ticker: str
    team_name: str
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2


@dataclass
class Position:
    ticker: str
    side: str          # 'yes' or 'no'
    count: int
    avg_price_cents: float


class TwoSidedArbStrategy:
    """
    Simple, explainable strategy for a two-team binary game market:

    1. Cross-market consistency check: on Kalshi each team typically has
       its own YES market for "Team X wins". A fully efficient market has
       yes_ask(TeamA) + yes_ask(TeamB) >= 100 (no free arbitrage) and
       yes_bid(TeamA) + yes_bid(TeamB) <= 100. If the *combined ask* price
       to buy YES on both sides is comfortably under 100c, buying both
       locks in a risk-free profit at settlement (guaranteed $1 payout on
       whichever team wins, cost < $1). This is the only genuinely
       "risk-free" signal here; even it depends on execution risk (prices
       can move/fill partially between your two orders).
    2. Momentum/mean-reversion fallback: if no arbitrage exists, track
       short-term price drift and fade large, fast moves that look more
       like order-flow noise than new information (very naive — treat as
       a placeholder to refine, not an edge).

    This class only decides WHAT to do; TradingBot below decides whether
    to actually send the order (paper vs live, risk limits, etc).
    """

    def __init__(self, entry_edge_cents: int = ENTRY_EDGE_CENTS):
        self.entry_edge_cents = entry_edge_cents
        self._price_history: dict[str, list[tuple[datetime, float]]] = {}

    def _record(self, quote: Quote):
        hist = self._price_history.setdefault(quote.ticker, [])
        hist.append((quote.fetched_at, quote.mid))
        # keep last ~20 observations
        if len(hist) > 20:
            hist.pop(0)

    def check_arbitrage(self, quote_a: Quote, quote_b: Quote) -> Optional[dict]:
        combined_ask = quote_a.yes_ask + quote_b.yes_ask
        if combined_ask <= 100 - self.entry_edge_cents:
            edge = 100 - combined_ask
            return {
                "type": "arbitrage",
                "legs": [
                    {"ticker": quote_a.ticker, "side": "yes", "action": "buy", "price_cents": quote_a.yes_ask},
                    {"ticker": quote_b.ticker, "side": "yes", "action": "buy", "price_cents": quote_b.yes_ask},
                ],
                "edge_cents": edge,
                "reason": (
                    f"Combined YES ask {combined_ask}c < 100c: buying both sides "
                    f"locks in {edge}c/contract regardless of outcome."
                ),
            }
        return None

    def check_momentum_fade(self, quote: Quote) -> Optional[dict]:
        self._record(quote)
        hist = self._price_history[quote.ticker]
        if len(hist) < 5:
            return None
        old_price = hist[0][1]
        move = quote.mid - old_price
        if abs(move) >= 8:  # cents, over the lookback window
            fade_side = "no" if move > 0 else "yes"
            price = quote.no_ask if fade_side == "no" else quote.yes_ask
            return {
                "type": "momentum_fade",
                "legs": [{"ticker": quote.ticker, "side": fade_side, "action": "buy", "price_cents": price}],
                "edge_cents": abs(move),
                "reason": f"{quote.team_name} moved {move:+.0f}c fast; fading the move with a small {fade_side.upper()} position.",
            }
        return None

    def check_exit(self, position: Position, quote: Quote) -> Optional[dict]:
        current_price = quote.yes_bid if position.side == "yes" else quote.no_bid
        pnl_cents = current_price - position.avg_price_cents
        if pnl_cents >= EXIT_PROFIT_CENTS:
            return {"action": "sell", "reason": f"Take profit: +{pnl_cents:.0f}c"}
        if pnl_cents <= -STOP_LOSS_CENTS:
            return {"action": "sell", "reason": f"Stop loss: {pnl_cents:.0f}c"}
        return None


# --------------------------------------------------------------------------
# RISK MANAGEMENT
# --------------------------------------------------------------------------

class RiskManager:
    def __init__(self, max_contracts: int = MAX_POSITION_CONTRACTS, max_dollars: float = MAX_DOLLARS_AT_RISK):
        self.max_contracts = max_contracts
        self.max_dollars = max_dollars
        self.dollars_committed = 0.0

    def size_order(self, price_cents: int, desired_count: int) -> int:
        dollars_per_contract = price_cents / 100.0
        room_left = self.max_dollars - self.dollars_committed
        max_by_dollars = int(room_left // dollars_per_contract) if dollars_per_contract > 0 else 0
        count = max(0, min(desired_count, self.max_contracts, max_by_dollars))
        return count

    def register_fill(self, price_cents: int, count: int):
        self.dollars_committed += (price_cents / 100.0) * count

    def release(self, price_cents: int, count: int):
        self.dollars_committed = max(0.0, self.dollars_committed - (price_cents / 100.0) * count)


# --------------------------------------------------------------------------
# TRADING BOT
# --------------------------------------------------------------------------

class TradingBot:
    def __init__(self, client: KalshiClient, ticker_a: str, ticker_b: str,
                 team_a_name: str, team_b_name: str, paper_trading: bool = True):
        self.client = client
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
        self.strategy = TwoSidedArbStrategy()
        self.risk = RiskManager()
        self.paper_trading = paper_trading
        self.open_positions: dict[str, Position] = {}

    def _fetch_quote(self, ticker: str, team_name: str) -> Quote:
        m = self.client.get_market(ticker)
        return Quote(
            ticker=ticker,
            team_name=team_name,
            yes_bid=m["yes_bid"],
            yes_ask=m["yes_ask"],
            no_bid=100 - m["yes_ask"],
            no_ask=100 - m["yes_bid"],
        )

    def _send_order(self, ticker: str, side: str, action: str, count: int, price_cents: int, reason: str):
        log.info(
            "%s %s %d contract(s) of %s @ %sc [%s] -- %s",
            "PAPER" if self.paper_trading else "LIVE",
            action.upper(), count, ticker, price_cents, side.upper(), reason,
        )
        if self.paper_trading:
            return {"paper": True}
        return self.client.place_order(ticker=ticker, side=side, action=action, count=count, price_cents=price_cents)

    def run_once(self):
        quote_a = self._fetch_quote(self.ticker_a, self.team_a_name)
        quote_b = self._fetch_quote(self.ticker_b, self.team_b_name)
        log.info(
            "%s YES %d/%d | %s YES %d/%d",
            quote_a.team_name, quote_a.yes_bid, quote_a.yes_ask,
            quote_b.team_name, quote_b.yes_bid, quote_b.yes_ask,
        )

        # 1) check for exits on anything already open
        for ticker, quote in ((self.ticker_a, quote_a), (self.ticker_b, quote_b)):
            pos = self.open_positions.get(ticker)
            if pos:
                exit_signal = self.strategy.check_exit(pos, quote)
                if exit_signal:
                    price = quote.yes_bid if pos.side == "yes" else quote.no_bid
                    self._send_order(ticker, pos.side, "sell", pos.count, price, exit_signal["reason"])
                    self.risk.release(pos.avg_price_cents, pos.count)
                    del self.open_positions[ticker]

        # 2) arbitrage check (highest priority signal)
        arb = self.strategy.check_arbitrage(quote_a, quote_b)
        if arb:
            for leg in arb["legs"]:
                count = self.risk.size_order(leg["price_cents"], desired_count=5)
                if count > 0:
                    self._send_order(leg["ticker"], leg["side"], leg["action"], count, leg["price_cents"], arb["reason"])
                    self.risk.register_fill(leg["price_cents"], count)
                    self.open_positions[leg["ticker"]] = Position(leg["ticker"], leg["side"], count, leg["price_cents"])
            return  # don't also fire momentum trades this cycle

        # 3) momentum fade fallback, per side
        for ticker, quote in ((self.ticker_a, quote_a), (self.ticker_b, quote_b)):
            if ticker in self.open_positions:
                continue
            signal = self.strategy.check_momentum_fade(quote)
            if signal:
                for leg in signal["legs"]:
                    count = self.risk.size_order(leg["price_cents"], desired_count=3)
                    if count > 0:
                        self._send_order(leg["ticker"], leg["side"], leg["action"], count, leg["price_cents"], signal["reason"])
                        self.risk.register_fill(leg["price_cents"], count)
                        self.open_positions[leg["ticker"]] = Position(leg["ticker"], leg["side"], count, leg["price_cents"])

    def run_forever(self, poll_interval: int = POLL_INTERVAL_SECONDS):
        log.info(
            "Starting bot | paper_trading=%s | %s vs %s",
            self.paper_trading, self.team_a_name, self.team_b_name,
        )
        while True:
            try:
                self.run_once()
            except Exception:
                log.exception("Error during trading loop iteration")
            time.sleep(poll_interval)


# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------

def find_nfl_game_markets(client: KalshiClient, team_a: str, team_b: str) -> tuple[str, str]:
    """
    Look up the current week's NFL game events and return the two
    per-team YES market tickers for the matchup between team_a and team_b.
    Kalshi's NFL series/ticker naming has changed over time -- if this
    lookup fails, check https://kalshi.com and pull the tickers manually
    from the event page URL, then skip this function.
    """
    events = client.get_events(series_ticker="KXNFLGAME", status="open")
    for event in events:
        title = event.get("title", "").lower()
        if team_a.lower() in title and team_b.lower() in title:
            markets = client.get_markets_for_event(event["event_ticker"])
            tickers = {m["yes_sub_title"].lower(): m["ticker"] for m in markets if "yes_sub_title" in m}
            ticker_a = next((t for name, t in tickers.items() if team_a.lower() in name), None)
            ticker_b = next((t for name, t in tickers.items() if team_b.lower() in name), None)
            if ticker_a and ticker_b:
                return ticker_a, ticker_b
    raise RuntimeError(
        f"Couldn't auto-find markets for {team_a} vs {team_b}. "
        "Pull the two market tickers manually from the Kalshi event page and pass them directly."
    )


if __name__ == "__main__":
    client = KalshiClient(
        key_id=KALSHI_KEY_ID,
        private_key_path=KALSHI_PRIVATE_KEY_PATH,
        use_demo=USE_DEMO,
    )

    # Example: replace with the two teams playing, or hardcode tickers
    # directly (e.g. "KXNFLGAME-26SEP14DALPHI-DAL") if auto-lookup fails.
    TEAM_A, TEAM_B = "Cowboys", "Eagles"
    ticker_a, ticker_b = find_nfl_game_markets(client, TEAM_A, TEAM_B)

    bot = TradingBot(
        client=client,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        team_a_name=TEAM_A,
        team_b_name=TEAM_B,
        paper_trading=PAPER_TRADING,
    )
    bot.run_forever()
