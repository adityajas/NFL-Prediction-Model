"""
Backend API for backtesting NFL prediction-market trading strategies
against real historical data (nfl_price_history.csv). Frontend comes later
-- this is a pure JSON API for now, easy to hit with curl or a future UI.

Run:
    python3 app.py
Then, e.g.:
    curl http://127.0.0.1:5050/api/games
    curl http://127.0.0.1:5050/api/strategies
    curl -X POST http://127.0.0.1:5050/api/simulate \
        -H "Content-Type: application/json" \
        -d '{"game_id": "KXNFLGAME-26AUG22DALARI", "strategy_ids": ["strategy_1", "strategy_2"]}'
"""
from __future__ import annotations

from flask import Flask, jsonify, request

from data import GameCatalog
from engine import run_backtest
from strategies import STRATEGIES, STRATEGIES_BY_ID

app = Flask(__name__)
catalog = GameCatalog()
catalog.load()


@app.route("/api/games")
def api_games():
    """All games available to backtest against."""
    return jsonify(catalog.list_games())


@app.route("/api/games/<game_id>")
def api_game_detail(game_id):
    """Metadata + full aligned price series for one game (for charting later)."""
    meta = catalog.get_game_meta(game_id)
    series = catalog.get_series(game_id)
    if meta is None or series is None:
        return jsonify({"error": "game not found"}), 404
    return jsonify({
        "meta": meta,
        "series": {
            "timestamps": series["timestamp"].apply(lambda t: t.isoformat()).tolist(),
            "price_a": series["price_a"].round(4).tolist(),
            "price_b": series["price_b"].round(4).tolist(),
            "team_a": meta["team_a"],
            "team_b": meta["team_b"],
        },
    })


@app.route("/api/strategies")
def api_strategies():
    """All registered strategies -- add one in strategies.py and it shows up here automatically."""
    return jsonify([{"id": s.id, "name": s.name, "description": s.description} for s in STRATEGIES])


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Body: {"game_id": "...", "strategy_ids": ["strategy_1", "strategy_2", ...]}
    Runs each requested strategy independently against the same game and
    returns each one's trades, equity curve, and summary P&L.
    """
    body = request.get_json(force=True)
    game_id = body.get("game_id")
    strategy_ids = body.get("strategy_ids", [])

    meta = catalog.get_game_meta(game_id)
    series = catalog.get_series(game_id)
    if meta is None or series is None:
        return jsonify({"error": f"game not found: {game_id}"}), 404

    results = []
    for sid in strategy_ids:
        strategy = STRATEGIES_BY_ID.get(sid)
        if strategy is None:
            results.append({"strategy_id": sid, "error": "unknown strategy"})
            continue
        try:
            results.append(run_backtest(strategy, series, meta["favorite"], meta["underdog"]))
        except Exception as e:
            results.append({"strategy_id": sid, "error": str(e)})

    return jsonify({"game_id": game_id, "results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
