"""
Generic backtest engine. Strategies don't implement their own bookkeeping --
they just declare what trades to make, and this engine handles cash
tracking, position tracking, trade logging, and the equity curve the same
way for every strategy. This is what makes adding a new strategy cheap: you
only write the decision logic, never the accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

CONTRACT_SIZE = 100


@dataclass
class TradeAction:
    action: str            # "BUY" or "SELL"
    team: str              # team code this action applies to
    qty: Optional[int]     # contracts. For SELL, None means "sell the entire current position"
    reason: str             # human-readable explanation, shown in the trade log


class Strategy:
    """Base class every strategy extends. Only `on_bar` is required."""
    id: str
    name: str
    description: str

    def on_start(self, state: dict, favorite: str, underdog: str, series: pd.DataFrame):
        """
        Called once before the backtest loop begins, with the FULL price
        series for this game already available. Use this to precompute
        anything that depends on knowing the whole game up front -- e.g. a
        time-based marker like "halftime" (which this data doesn't record
        directly; the best available proxy is the temporal midpoint between
        the first and last price observation for this market). Optional --
        default does nothing. Write whatever you need into `state`.
        """
        pass

    def initial_actions(self, favorite: str, underdog: str, fav_price0: float, dog_price0: float) -> list[TradeAction]:
        """
        Called once, on the first bar of the game. Default (shared by every
        strategy so far) is buying 100 contracts of both teams. Override
        this if a future strategy needs a different opening move.
        """
        return [
            TradeAction("BUY", favorite, CONTRACT_SIZE, "Initial buy (favorite)"),
            TradeAction("BUY", underdog, CONTRACT_SIZE, "Initial buy (underdog)"),
        ]

    def on_bar(self, state: dict, favorite: str, underdog: str, fav_price: float, dog_price: float,
               timestamp: pd.Timestamp) -> list[TradeAction]:
        """
        Called once per bar (after the first). Return a list of TradeActions
        to execute at this bar's price, or an empty list to do nothing.
        `state` is a plain dict that's yours to read/write across calls
        (e.g. to remember "have I already triggered?" or anything set in
        on_start) -- the engine also refreshes state["fav_qty"] /
        state["dog_qty"] with current holdings before each call, in case
        your logic needs them. `timestamp` is this bar's actual time,
        useful for any time-based condition (e.g. comparing against a
        halftime marker computed in on_start).
        """
        raise NotImplementedError


def run_backtest(strategy: Strategy, series: pd.DataFrame, favorite: str, underdog: str) -> dict:
    """
    series: DataFrame with columns timestamp, price_a, price_b, team_a, team_b
    (this is exactly what GameCatalog.get_series() returns).
    """
    first_row = series.iloc[0]
    fav_col = "price_a" if favorite == first_row["team_a"] else "price_b"
    dog_col = "price_b" if fav_col == "price_a" else "price_a"

    cash = 0.0
    qty = {favorite: 0, underdog: 0}
    trades: list[dict] = []
    equity_curve: list[dict] = []

    def apply(trade_action: TradeAction, price: float, timestamp_iso: str):
        nonlocal cash
        team = trade_action.team
        if trade_action.action == "BUY":
            q = trade_action.qty
            if q is None:
                raise ValueError("BUY actions must specify a qty")
            cash -= q * price
            qty[team] += q
        elif trade_action.action == "SELL":
            q = trade_action.qty if trade_action.qty is not None else qty[team]
            cash += q * price
            qty[team] -= q
        else:
            raise ValueError(f"Unknown action type: {trade_action.action}")
        trades.append({
            "timestamp": timestamp_iso,
            "action": trade_action.action,
            "team": team,
            "qty": q,
            "price": round(float(price), 4),
            "reason": trade_action.reason,
        })

    # --- initial bar ---
    ts0 = first_row["timestamp"].isoformat()
    fav_price0, dog_price0 = float(first_row[fav_col]), float(first_row[dog_col])
    price_lookup = {favorite: fav_price0, underdog: dog_price0}
    for action in strategy.initial_actions(favorite, underdog, fav_price0, dog_price0):
        apply(action, price_lookup[action.team], ts0)
    equity_curve.append({
        "timestamp": ts0,
        "equity": round(cash + qty[favorite] * fav_price0 + qty[underdog] * dog_price0, 4),
    })

    # --- remaining bars ---
    state: dict = {}
    strategy.on_start(state, favorite, underdog, series)
    for _, row in series.iloc[1:].iterrows():
        ts = row["timestamp"].isoformat()
        fav_price, dog_price = float(row[fav_col]), float(row[dog_col])
        price_lookup = {favorite: fav_price, underdog: dog_price}

        state["fav_qty"] = qty[favorite]
        state["dog_qty"] = qty[underdog]
        for action in (strategy.on_bar(state, favorite, underdog, fav_price, dog_price, row["timestamp"]) or []):
            apply(action, price_lookup[action.team], ts)

        equity_curve.append({
            "timestamp": ts,
            "equity": round(cash + qty[favorite] * fav_price + qty[underdog] * dog_price, 4),
        })

    last_row = series.iloc[-1]
    final_fav_price, final_dog_price = float(last_row[fav_col]), float(last_row[dog_col])
    final_value = cash + qty[favorite] * final_fav_price + qty[underdog] * final_dog_price

    return {
        "strategy_id": strategy.id,
        "strategy_name": strategy.name,
        "favorite": favorite,
        "underdog": underdog,
        "trades": trades,
        "equity_curve": equity_curve,
        "summary": {
            "final_pnl": round(float(final_value), 2),
            "total_contracts_bought": sum(t["qty"] for t in trades if t["action"] == "BUY"),
            "total_contracts_sold": sum(t["qty"] for t in trades if t["action"] == "SELL"),
            "n_trades": len(trades),
            "final_position": {favorite: qty[favorite], underdog: qty[underdog]},
        },
    }
