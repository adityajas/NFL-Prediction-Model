https://adityajas.github.io/nfl-betting-model/nfl_strategy_visualizer.html

Open the file, choose game and strategy to simulate returns
Graph data is pulled from kalshi_nfl_prices.csv

Workflow for pull_kalshi_nfl_prices.py
1. Find every settled NFL game (get_all_settled_events)
Calls GET /events?series_ticker=KXNFLGAME&status=settled, paging through with the cursor Kalshi returns until there's nothing left. This is just discovery — it gives you a list of event_tickers like KXNFLGAME-25DEC14MINDAL, one per game, with no price data yet.
2. For each game, get its two markets (get_event_markets)
Each NFL game on Kalshi has two separate binary "yes" markets — one per team, e.g. KXNFLGAME-25DEC14MINDAL-MIN and ...-DAL. This calls GET /events/{event_ticker} to get both. It tries the live endpoint first; if that comes back empty (because the game is older than Kalshi's live-data cutoff), it falls back to GET /historical/markets?event_ticker=... — that's the fix from earlier in this thread.
3. For each market, pick a time window and pull 1-minute candles (get_candlesticks)
Each market has an open_time (when betting opened, often days before kickoff) and close_time (settlement). Rather than pull the whole multi-day window, the script assumes the game itself happened in the last ~5 hours before close_time and only requests candlesticks for that slice — one GET .../candlesticks call per team per game. This also has its own live→historical fallback, same idea as step 2.
4. Flatten into rows
Each candlestick becomes one CSV row: which game, which team, the bar's timestamp, and open/close/mean "yes" price for that minute. So a single game contributes up to ~300 rows (2 teams × ~5 hours × 60 bars/hour).
5. Write incrementally
After every game's markets are processed, it rewrites the whole CSV from the rows list accumulated so far — not append-only, but cheap enough at this scale, and it means killing the script partway through doesn't lose earlier games.
