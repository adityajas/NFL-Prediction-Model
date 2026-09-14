"""
data_pipeline.py
 
Pulls raw NFL play-by-play data and schedule/betting-line data, then saves
both locally so we don't re-download from the network every time we want
to work with them.
"""
 
import nfl_data_py as nfl
import pandas as pd
from pathlib import Path
 
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
 
 
def fetch_pbp_data(seasons: list[int]) -> pd.DataFrame:
    """
    Pulls play-by-play data for the given seasons and returns it as a DataFrame.
 
    Example:
        fetch_pbp_data([2015, 2016, ..., 2024])
    """
    pbp = nfl.import_pbp_data(seasons)
    return pbp
 
 
def save_pbp_data(pbp: pd.DataFrame, filename: str = "pbp_raw.csv") -> None:
    """Saves a play-by-play DataFrame to data/raw/."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / filename
    pbp.to_csv(out_path, index=False)
    print(f"Saved {len(pbp):,} rows to {out_path}")
 
 
def fetch_schedules(seasons: list[int]) -> pd.DataFrame:
    """
    Pulls game-level schedule data, which includes final scores AND
    betting lines (spread_line, total_line, home/away moneyline, etc.)
    for each game. This is what we'll compare model predictions against.
 
    Note: spread_line is the CLOSING line, from the home team's perspective.
    Positive spread_line = home team favored. Negative = away team favored.
    (Source: nflverse / Pro-Football-Reference)
 
    Example:
        fetch_schedules([2015, 2016, ..., 2024])
    """
    schedules = nfl.import_schedules(seasons)
    return schedules
 
 
def save_schedules(schedules: pd.DataFrame, filename: str = "schedules_raw.csv") -> None:
    """Saves a schedules/lines DataFrame to data/raw/."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / filename
    schedules.to_csv(out_path, index=False)
    print(f"Saved {len(schedules):,} rows to {out_path}")
 
 
if __name__ == "__main__":
    seasons = list(range(2015, 2025))  # 2015 through 2024 inclusive
 
    pbp = fetch_pbp_data(seasons)
    save_pbp_data(pbp)
 
    schedules = fetch_schedules(seasons)
    save_schedules(schedules)