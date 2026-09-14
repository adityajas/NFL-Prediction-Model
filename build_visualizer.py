"""
Regenerates nfl_strategy_visualizer.html's embedded ALL_GAMES data from your
actual Python backend (data.py + engine.py + strategies.py), so the visualizer
always reflects your current CSV and your current strategy list -- with zero
changes needed to the HTML/JS itself.

WHY THIS WORKS WITHOUT TOUCHING THE HTML: the visualizer's JS already reads
games and strategies generically --
    const gameIds = Object.keys(ALL_GAMES);              // new games just appear
    Object.keys(game().strategies).forEach(sid => ...)   // new strategies just appear
So the only thing that ever needs to change is the ALL_GAMES data itself.

USAGE
-----
Add a new game -> put its data in nfl_price_history.csv (re-run
import_nfl_data.py, or merge in nfl_price_history_live_polymarket.csv),
then re-run this script.

Add a new strategy -> add it to STRATEGIES in strategies.py (per its own
docstring: subclass Strategy, implement on_bar(), add an instance to the
list), then re-run this script. No other file needs to change.

    python3 build_visualizer.py
    python3 build_visualizer.py --template other.html --output new.html
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data import GameCatalog
from engine import run_backtest
from strategies import STRATEGIES

DATA_LINE_PREFIX = "const ALL_GAMES = "


def build_all_games_dict(round_prices: int = 2) -> dict:
    catalog = GameCatalog()
    catalog.load()

    all_games = {}
    game_metas = catalog.list_games()
    print(f"Building data for {len(game_metas)} games x {len(STRATEGIES)} strategies...")

    for meta in game_metas:
        series = catalog.get_series(meta["game_id"])
        if series is None or series.empty:
            continue

        strategies_out = {}
        for strategy in STRATEGIES:
            result = run_backtest(strategy, series, meta["favorite"], meta["underdog"])
            strategies_out[strategy.id] = {
                "name": strategy.name,
                "description": strategy.description,
                "trades": result["trades"],
                "equity_curve": result["equity_curve"],
                "summary": result["summary"],
            }

        all_games[meta["game_id"]] = {
            "game_id": meta["game_id"],
            "favorite": meta["favorite"],
            "underdog": meta["underdog"],
            "team_a": meta["team_a"],
            "team_b": meta["team_b"],
            "date": meta["date"],
            "timestamps": [t.isoformat() for t in series["timestamp"]],
            "price_a": [round(float(p), round_prices) for p in series["price_a"]],
            "price_b": [round(float(p), round_prices) for p in series["price_b"]],
            "strategies": strategies_out,
        }

    print(f"Built {len(all_games)} games successfully.")
    return all_games


def splice_into_html(template_path: Path, output_path: Path, all_games: dict):
    lines = template_path.read_text().splitlines(keepends=True)

    target_idx = None
    for i, line in enumerate(lines):
        if line.startswith(DATA_LINE_PREFIX):
            target_idx = i
            break

    if target_idx is None:
        print(f"ERROR: couldn't find a line starting with {DATA_LINE_PREFIX!r} in {template_path}. "
              f"Is this the right template file? Aborting without writing anything.")
        sys.exit(1)

    new_line = DATA_LINE_PREFIX + json.dumps(all_games, separators=(",", ":")) + ";\n"
    old_size = len(lines[target_idx])
    lines[target_idx] = new_line

    output_path.write_text("".join(lines))
    print(f"Data line: {old_size:,} chars -> {len(new_line):,} chars")
    print(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes total)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=Path("nfl_strategy_visualizer.html"),
                         help="Existing HTML file to use as the shell (its non-data code is kept as-is)")
    parser.add_argument("--output", type=Path, default=None,
                         help="Where to write the result. Defaults to overwriting --template.")
    args = parser.parse_args()
    output = args.output or args.template

    if not args.template.exists():
        print(f"ERROR: template file not found: {args.template}")
        sys.exit(1)

    all_games = build_all_games_dict()
    splice_into_html(args.template, output, all_games)


if __name__ == "__main__":
    main()
