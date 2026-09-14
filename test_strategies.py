"""
Quick way to test strategies without spinning up the Flask server.

Usage:
    python3 test_strategies.py                          # all strategies, all games
    python3 test_strategies.py --game KXNFLGAME-26AUG22DALARI
    python3 test_strategies.py --strategy strategy_1

EXCLUDING PRESEASON GAMES (July/August): append (nps) -- "no preseason" --
to the strategy id. NOTE: quote it, since parentheses are special
characters in bash and will cause a syntax error if left unquoted:

    python3 test_strategies.py --strategy "strategy_1(nps)"

Or use the standalone flag, which works with --strategy omitted too (runs
every registered strategy, preseason excluded):

    python3 test_strategies.py --no-preseason
"""
from __future__ import annotations

import argparse
import re

from data import GameCatalog
from engine import run_backtest
from strategies import STRATEGIES, STRATEGIES_BY_ID

NPS_SUFFIX_RE = re.compile(r"^(?P<strategy_id>.+?)\s*\(nps\)$", re.IGNORECASE)


def parse_strategy_arg(raw: str | None) -> tuple[str | None, bool]:
    """Returns (strategy_id_or_None, exclude_preseason). Strips a trailing
    "(nps)" (case-insensitive, optional whitespace before it) if present."""
    if raw is None:
        return None, False
    m = NPS_SUFFIX_RE.match(raw.strip())
    if m:
        return m.group("strategy_id"), True
    return raw, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, help="Run against a single game_id instead of all games")
    parser.add_argument("--strategy", type=str,
                         help='Run a single strategy_id instead of all registered strategies. '
                              'Append (nps) to exclude preseason games, e.g. "strategy_1(nps)" (quote it -- '
                              'parentheses are special in bash).')
    parser.add_argument("--no-preseason", action="store_true",
                         help="Exclude July/August (preseason) games. Works whether or not --strategy is set.")
    args = parser.parse_args()

    strategy_id, nps_from_suffix = parse_strategy_arg(args.strategy)
    exclude_preseason = args.no_preseason or nps_from_suffix

    catalog = GameCatalog()
    catalog.load()

    if args.game:
        meta = catalog.get_game_meta(args.game)
        if meta is None:
            print(f"No such game: {args.game}")
            return
        games = [meta]
        if exclude_preseason and meta["is_preseason"]:
            print(f"Note: {args.game} is a preseason game and --no-preseason/(nps) was set -- "
                  f"nothing to run.")
            return
    else:
        games = catalog.list_games(exclude_preseason=exclude_preseason)

    if strategy_id:
        if strategy_id not in STRATEGIES_BY_ID:
            print(f"No such strategy: {strategy_id}. Available: {list(STRATEGIES_BY_ID)}")
            return
        strategies = [STRATEGIES_BY_ID[strategy_id]]
    else:
        strategies = STRATEGIES

    filter_note = " (preseason excluded)" if exclude_preseason else ""
    print(f"Running {len(strategies)} strategy(ies) across {len(games)} game(s){filter_note}...\n")
    print(f"{'game':<28} {'strategy':<45} {'pnl':>10} {'trades':>7} {'triggered':>10}")
    print("-" * 105)

    totals = {s.id: 0.0 for s in strategies}
    for meta in games:
        series = catalog.get_series(meta["game_id"])
        for strategy in strategies:
            result = run_backtest(strategy, series, meta["favorite"], meta["underdog"])
            pnl = result["summary"]["final_pnl"]
            triggered = result["summary"]["n_trades"] > 2  # more than just the 2 initial buys
            totals[strategy.id] += pnl
            print(f"{meta['game_id']:<28} {strategy.name:<45} {pnl:>10.2f} "
                  f"{result['summary']['n_trades']:>7} {str(triggered):>10}")

    print("-" * 105)
    for strategy in strategies:
        avg = totals[strategy.id] / len(games) if games else 0
        print(f"{strategy.name}: total P&L across {len(games)} games{filter_note} = {totals[strategy.id]:+.2f} "
              f"(avg {avg:+.2f}/game)")


if __name__ == "__main__":
    main()
