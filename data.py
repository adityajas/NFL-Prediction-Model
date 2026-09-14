"""
Loads kalshi_nfl_prices.csv (1-minute Kalshi candlestick data) once and
builds a catalog of games (both teams' aligned price series) that the API
and strategy engine both use.

NOTE ON PROVENANCE: this targets the schema confirmed directly from the
uploaded kalshi_nfl_prices.csv --
    event_ticker, game_date, team_ticker, side, end_period_ts,
    datetime_utc, yes_price_open, yes_price_close, yes_price_mean
-- rather than the original nfl_price_history.csv (Kingsets) schema this
class was first built around. If your working copy of this file differs
from what's here, diff before overwriting -- this was reconstructed from
the CSV's real columns, not copied from your actual prior file.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CSV_FILENAME = "kalshi_nfl_prices.csv"


def find_csv() -> Path:
    """
    Look for the CSV in a few sensible places rather than assuming one fixed
    folder layout -- these files might sit right next to the CSV (flat
    project folder) or in a subfolder alongside it. Raises a clear error
    listing where it looked if it can't find it, instead of a confusing
    pandas FileNotFoundError pointing at the wrong path.
    """
    candidates = [
        Path(__file__).parent / CSV_FILENAME,          # same folder as this file
        Path(__file__).parent.parent / CSV_FILENAME,    # one level up
        Path.cwd() / CSV_FILENAME,                       # wherever the script was run from
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Couldn't find {CSV_FILENAME}. Looked in:\n" +
        "\n".join(f"  - {c}" for c in candidates) +
        f"\nMake sure {CSV_FILENAME} is in the same folder as data.py, "
        f"or pass GameCatalog(csv_path=...) explicitly."
    )

# KXNFLGAME-26AUG22BALMIN-MIN -> game part is everything before the final
# "-TEAMCODE"; the date/teams are embedded in the middle segment.
GAME_ID_RE = re.compile(r"^KXNFLGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{4,8})$")
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

# NFL preseason runs in July (training camp / HOF game) and August.
PRESEASON_MONTHS = {7, 8}


class GameCatalog:
    def __init__(self, csv_path: Path | None = None):
        self.csv_path = csv_path or find_csv()
        self._raw: pd.DataFrame | None = None
        self._games: dict[str, dict] = {}
        self._series_cache: dict[str, pd.DataFrame] = {}

    def load(self):
        df = pd.read_csv(self.csv_path)

        # kalshi_nfl_prices.csv's real column names -> the generic names the
        # rest of this class (and engine.py) work with.
        df = df.rename(columns={
            "team_ticker": "market_id",
            "yes_price_close": "price",
            "datetime_utc": "timestamp",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # only real per-game moneyline markets belong here
        df = df[df["market_id"].str.match(r"^KXNFLGAME-\d{2}[A-Z]{3}\d{2}[A-Z]{4,8}-[A-Z]{2,4}$", na=False)]
        df["game_id"] = df["market_id"].str.rsplit("-", n=1).str[0]
        df["team_code"] = df["market_id"].str.rsplit("-", n=1).str[1]

        # A 1-minute candle with no trade in it has a null close price --
        # forward/back-fill within each market so those gaps carry the last
        # known price instead of poisoning downstream P&L math with NaNs.
        df = df.sort_values(["market_id", "timestamp"])
        df["price"] = df.groupby("market_id")["price"].transform(lambda s: s.ffill().bfill())
        df = df.dropna(subset=["price"])

        self._raw = df
        self._build_catalog()

    def _build_catalog(self):
        for game_id, g in self._raw.groupby("game_id"):
            teams = sorted(g["team_code"].unique())
            if len(teams) != 2:
                continue  # skip games where we're missing one side's data
            team_a, team_b = teams

            series_a = g[g["team_code"] == team_a][["timestamp", "price"]].sort_values("timestamp")
            series_b = g[g["team_code"] == team_b][["timestamp", "price"]].sort_values("timestamp")
            merged = pd.merge(series_a, series_b, on="timestamp", suffixes=(f"_{team_a}", f"_{team_b}"))
            if merged.empty:
                continue

            first = merged.iloc[0]
            price_a0, price_b0 = first[f"price_{team_a}"], first[f"price_{team_b}"]
            favorite = team_a if price_a0 > price_b0 else team_b
            underdog = team_b if favorite == team_a else team_a

            m = GAME_ID_RE.match(game_id)
            date_label = None
            month_num = None
            if m:
                yy, mon, dd, _ = m.groups()
                month_num = MONTHS.get(mon)
                if month_num:
                    date_label = f"20{yy}-{month_num:02d}-{dd}"

            self._games[game_id] = {
                "game_id": game_id,
                "team_a": team_a,
                "team_b": team_b,
                "favorite": favorite,
                "underdog": underdog,
                "date": date_label,
                "is_preseason": month_num in PRESEASON_MONTHS if month_num else False,
                "n_bars": len(merged),
                "start": merged["timestamp"].min().isoformat(),
                "end": merged["timestamp"].max().isoformat(),
            }
            self._series_cache[game_id] = merged.rename(
                columns={f"price_{team_a}": "price_a", f"price_{team_b}": "price_b"}
            ).assign(team_a=team_a, team_b=team_b)

    def list_games(self, exclude_preseason: bool = False) -> list[dict]:
        games = self._games.values()
        if exclude_preseason:
            games = [g for g in games if not g["is_preseason"]]
        return sorted(games, key=lambda g: (g["date"] or "", g["game_id"]))

    def get_game_meta(self, game_id: str) -> dict | None:
        return self._games.get(game_id)

    def get_series(self, game_id: str) -> pd.DataFrame | None:
        return self._series_cache.get(game_id)
