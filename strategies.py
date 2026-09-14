"""
Strategy definitions.

TO ADD A NEW STRATEGY: write a class extending Strategy (see engine.py),
implement on_bar() with your decision logic, then add an instance of it to
the STRATEGIES list at the bottom of this file. That's the only change
needed -- the backtest engine, the API, and (later) the frontend all pick
up new strategies automatically from that list.
"""
from __future__ import annotations

from engine import CONTRACT_SIZE, Strategy, TradeAction


class Strategy1FlipDoubleDown(Strategy):
    id = "strategy_1"
    name = "Strategy 1: Initial Buy + Flip Double-Down"
    description = (
        "Buy 100 contracts of both teams at kickoff. The first time the "
        "initial favorite's price drops below the underdog's, sell the "
        "entire underdog position and buy 100 more of the favorite."
    )

    def on_bar(self, state, favorite, underdog, fav_price, dog_price, timestamp):
        if state.get("triggered"):
            return []
        if fav_price < dog_price:
            state["triggered"] = True
            return [
                TradeAction("SELL", underdog, None,
                            "Favorite's price dropped below underdog's -- exiting underdog position"),
                TradeAction("BUY", favorite, CONTRACT_SIZE,
                            "Doubling down on original favorite at the dip"),
            ]
        return []


class Strategy2FavoriteBreakout(Strategy):
    id = "strategy_2"
    name = "Strategy 2: Sell Underdog on Favorite Breakout"
    description = (
        "Buy 100 contracts of both teams at kickoff. The first time the "
        "initial favorite's price rises above 80c, sell the entire "
        "underdog position."
    )
    BREAKOUT_THRESHOLD = 0.80

    def on_bar(self, state, favorite, underdog, fav_price, dog_price, timestamp):
        if state.get("triggered"):
            return []
        if fav_price > self.BREAKOUT_THRESHOLD:
            state["triggered"] = True
            return [
                TradeAction("SELL", underdog, None,
                            f"Favorite's price exceeded {self.BREAKOUT_THRESHOLD:.0%} -- exiting underdog position"),
            ]
        return []


class Strategy3HalftimeCheck(Strategy):
    id = "strategy_3"
    name = "Strategy 3: Halftime Crossover Check"
    description = (
        "Buy 100 contracts of both teams at kickoff. At halftime "
        "(approximated as the midpoint in time between this market's "
        "first and last price observation -- the data has no actual game "
        "clock), if the underdog's price has never risen above 50c up to "
        "that point, sell the entire underdog position at the current "
        "price and buy 100 more of the favorite."
    )
    CROSS_THRESHOLD = 0.50

    def on_start(self, state, favorite, underdog, series):
        # Halftime isn't recorded in the data -- only price ticks are. The
        # best available proxy is the temporal midpoint between the first
        # and last observation for this market. This will be imprecise if
        # the market opened well before kickoff or kept trading briefly
        # after the final whistle before settling -- the true in-game
        # halftime could sit earlier or later than this midpoint.
        t_start = series["timestamp"].iloc[0]
        t_end = series["timestamp"].iloc[-1]
        state["halftime_ts"] = t_start + (t_end - t_start) / 2
        state["underdog_crossed"] = False
        state["halftime_action_taken"] = False

    def on_bar(self, state, favorite, underdog, fav_price, dog_price, timestamp):
        if state.get("halftime_action_taken"):
            return []

        if dog_price > self.CROSS_THRESHOLD:
            state["underdog_crossed"] = True

        if timestamp >= state["halftime_ts"]:
            state["halftime_action_taken"] = True
            if not state["underdog_crossed"]:
                return [
                    TradeAction("SELL", underdog, None,
                                f"Underdog never crossed {self.CROSS_THRESHOLD:.0%} by (approximate) "
                                f"halftime -- exiting underdog position"),
                    TradeAction("BUY", favorite, CONTRACT_SIZE,
                                "Buying more of the favorite at the halftime mark"),
                ]
        return []


# --- registry: add new strategy instances here ---
STRATEGIES: list[Strategy] = [
    Strategy1FlipDoubleDown(),
    Strategy2FavoriteBreakout(),
    Strategy3HalftimeCheck(),
]

STRATEGIES_BY_ID: dict[str, Strategy] = {s.id: s for s in STRATEGIES}
