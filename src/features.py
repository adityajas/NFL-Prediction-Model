"""
features.py

Turns raw play-by-play data into game-level features suitable for modeling.

Pipeline:
    1. Filter out garbage-time plays (blowout situations distort EPA/success
       rate and don't reflect real team quality).
    2. Aggregate plays into one row per TEAM per GAME (offense stats when the
       team had the ball, defense stats -- i.e. what they allowed -- when the
       opponent had the ball), including situational splits (down/distance,
       red zone, pass vs. rush).
    3. Build rolling-window and season-to-date features per team, using only
       PRIOR games (shifted by one) to avoid data leakage -- a team's feature
       for their Week 5 game must never include Week 5 data.
    4. Merge home and away team features onto the schedule/spread table so
       each row = one game, with both teams' pre-game form and the closing
       spread to beat.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

ROLLING_WINDOWS = [3, 5]  # games


# ---------------------------------------------------------------------------
# 1. Garbage time filtering
# ---------------------------------------------------------------------------

def filter_garbage_time(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Removes plays that occurred in "garbage time" -- situations where the
    outcome is effectively decided and teams are playing differently than
    they would in a competitive game (e.g. running out the clock up 30,
    or throwing desperation deep balls down 30).

    Uses win probability (wp) as the filter: plays where either team's
    win probability is beyond a 95/5 threshold in the 4th quarter are
    excluded. Falls back to a simple score-differential + time heuristic
    if win probability columns aren't available.
    """
    pbp = pbp.copy()

    if "wp" in pbp.columns and "qtr" in pbp.columns:
        competitive = ~(
            (pbp["qtr"] == 4)
            & ((pbp["wp"] > 0.95) | (pbp["wp"] < 0.05))
        )
    else:
        # Fallback heuristic: 4th quarter, more than 16 point margin,
        # under 5 minutes left.
        score_diff = (pbp["posteam_score"] - pbp["defteam_score"]).abs()
        competitive = ~(
            (pbp["qtr"] == 4)
            & (score_diff > 16)
            & (pbp["quarter_seconds_remaining"] < 300)
        )

    return pbp[competitive].copy()


# ---------------------------------------------------------------------------
# 2. Team-game level aggregation with situational splits
# ---------------------------------------------------------------------------

def _agg_offense(plays: pd.DataFrame) -> pd.Series:
    """Aggregates one team's offensive plays in one game into a feature row."""
    pass_plays = plays[plays["pass"] == 1]
    rush_plays = plays[plays["rush"] == 1]
    completed_passes = plays[plays["complete_pass"] == 1]
    third_downs = plays[plays["down"] == 3]
    red_zone = plays[plays["yardline_100"] <= 20]

    return pd.Series({
        "off_epa_per_play": plays["epa"].mean(),
        "off_success_rate": plays["success"].mean(),
        "off_pass_epa": pass_plays["epa"].mean(),
        "off_rush_epa": rush_plays["epa"].mean(),
        "off_pass_rate": len(pass_plays) / len(plays) if len(plays) else np.nan,
        "off_third_down_conv_rate": third_downs["first_down"].mean() if len(third_downs) else np.nan,
        "off_red_zone_epa": red_zone["epa"].mean() if len(red_zone) else np.nan,
        "off_red_zone_td_rate": (red_zone["touchdown"] == 1).mean() if len(red_zone) else np.nan,
        "off_plays": len(plays),
        # Total yardage -- these double as prediction TARGETS (see features.py
        # ACTUAL_TARGET_COLS) and as inputs to their own rolling averages.
        "off_rush_yards": rush_plays["yards_gained"].sum(),
        "off_rec_yards": completed_passes["yards_gained"].sum(),
    })


def _agg_defense(plays: pd.DataFrame) -> pd.Series:
    """
    Aggregates what a team ALLOWED on defense in one game (i.e. the
    opponent's offensive plays against them), same situational breakdown.
    """
    pass_plays = plays[plays["pass"] == 1]
    rush_plays = plays[plays["rush"] == 1]
    completed_passes = plays[plays["complete_pass"] == 1]
    third_downs = plays[plays["down"] == 3]
    red_zone = plays[plays["yardline_100"] <= 20]

    return pd.Series({
        "def_epa_per_play_allowed": plays["epa"].mean(),
        "def_success_rate_allowed": plays["success"].mean(),
        "def_pass_epa_allowed": pass_plays["epa"].mean(),
        "def_rush_epa_allowed": rush_plays["epa"].mean(),
        "def_third_down_conv_rate_allowed": third_downs["first_down"].mean() if len(third_downs) else np.nan,
        "def_red_zone_epa_allowed": red_zone["epa"].mean() if len(red_zone) else np.nan,
        "def_red_zone_td_rate_allowed": (red_zone["touchdown"] == 1).mean() if len(red_zone) else np.nan,
        "def_rush_yards_allowed": rush_plays["yards_gained"].sum(),
        "def_rec_yards_allowed": completed_passes["yards_gained"].sum(),
    })


def build_team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses play-by-play data into one row per team per game, with
    offensive and defensive situational stats. Only regular offensive
    plays (pass/rush) are used for EPA aggregation -- special teams and
    penalties without a play type are excluded.
    """
    plays = pbp[
        (pbp["play_type"].isin(["pass", "run"]))
        & pbp["epa"].notna()
    ].copy()

    offense_rows = []
    for (game_id, posteam), group in plays.groupby(["game_id", "posteam"]):
        if pd.isna(posteam):
            continue
        stats = _agg_offense(group)
        stats["game_id"] = game_id
        stats["team"] = posteam
        offense_rows.append(stats)
    offense_df = pd.DataFrame(offense_rows)

    defense_rows = []
    for (game_id, defteam), group in plays.groupby(["game_id", "defteam"]):
        if pd.isna(defteam):
            continue
        stats = _agg_defense(group)
        stats["game_id"] = game_id
        stats["team"] = defteam
        defense_rows.append(stats)
    defense_df = pd.DataFrame(defense_rows)

    team_game = offense_df.merge(defense_df, on=["game_id", "team"], how="outer")
    return team_game


# ---------------------------------------------------------------------------
# 3. Rolling windows and season-to-date features (leakage-safe)
# ---------------------------------------------------------------------------

# Raw stats we want to predict directly (rushing/receiving yards). These get
# preserved as unshifted "actual_*" columns -- used ONLY as regression
# targets, never as model inputs -- in addition to getting their own
# rolling/season-to-date versions like every other stat.
ACTUAL_TARGET_COLS = ["off_rush_yards", "off_rec_yards"]


def add_rolling_features(
    team_game: pd.DataFrame,
    schedules: pd.DataFrame,
    windows: list[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """
    Adds rolling-window and season-to-date (expanding) versions of every
    stat column, computed PER TEAM in chronological order.

    Critically, every rolling/expanding stat is shifted by one game before
    being attached, so a team's features going INTO game N only reflect
    games 1..N-1 -- never game N itself. This prevents leakage.

    Also preserves the actual (unshifted) values of ACTUAL_TARGET_COLS as
    "actual_*" columns -- these represent what really happened in THAT game
    and are meant to be used as regression targets (e.g. by yards_model.py),
    never as model inputs.
    """
    # Attach season/week/date so we can sort chronologically per team.
    game_order = schedules[["game_id", "season", "week", "gameday"]]
    team_game = team_game.merge(game_order, on="game_id", how="left")
    team_game = team_game.sort_values(["team", "season", "gameday", "week"])

    # Preserve ground-truth values before any shifting happens.
    for col in ACTUAL_TARGET_COLS:
        if col in team_game.columns:
            team_game[f"actual_{col}"] = team_game[col]

    stat_cols = [
        c for c in team_game.columns
        if c not in ("game_id", "team", "season", "week", "gameday")
        and not c.startswith("actual_")
    ]

    grouped = team_game.groupby("team", group_keys=False)

    for col in stat_cols:
        # Season-to-date average of all prior games this season.
        team_game[f"{col}_szn_avg"] = (
            grouped[col]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(drop=True)
        )
        for w in windows:
            team_game[f"{col}_last{w}"] = (
                grouped[col]
                .apply(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
                .reset_index(drop=True)
            )

    # Drop the raw same-game stats now that we have leakage-safe versions --
    # keeping both would let the model see the outcome of the game it's
    # trying to predict. The "actual_*" columns are untouched here since
    # they're meant to be used as targets, not features.
    team_game = team_game.drop(columns=stat_cols)

    return team_game


# ---------------------------------------------------------------------------
# 4. Merge home/away team features onto the schedule
# ---------------------------------------------------------------------------

def build_game_level_features(pbp: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: filters garbage time, aggregates to team-game level,
    builds leakage-safe rolling features, then joins home and away team
    features onto the schedule table (which has scores + spread_line).

    Returns one row per game with:
        - game_id, season, week, home_team, away_team
        - home_score, away_score, spread_line, total_line
        - home_rest, away_rest
        - every rolling/season-to-date feature, prefixed home_ / away_
        - home_actual_off_rush_yards, home_actual_off_rec_yards (and away_
          equivalents) -- ground-truth yardage for that game, meant to be
          used as regression targets by yards_model.py, never as model inputs
    """
    clean_pbp = filter_garbage_time(pbp)
    team_game = build_team_game_stats(clean_pbp)
    team_features = add_rolling_features(team_game, schedules)

    home_features = team_features.add_prefix("home_").rename(
        columns={"home_game_id": "game_id", "home_team": "home_team"}
    )
    away_features = team_features.add_prefix("away_").rename(
        columns={"away_game_id": "game_id", "away_team": "away_team"}
    )

    base_cols = [
        "game_id", "season", "week", "home_team", "away_team",
        "home_score", "away_score", "spread_line", "total_line",
        "home_rest", "away_rest",
    ]
    base = schedules[base_cols].copy()

    merged = base.merge(home_features, on=["game_id", "home_team"], how="left")
    merged = merged.merge(away_features, on=["game_id", "away_team"], how="left")

    return merged


def save_features(features: pd.DataFrame, filename: str = "game_features.csv") -> None:
    """Saves the final game-level feature table to data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    features.to_csv(out_path, index=False)
    print(f"Saved {len(features):,} rows to {out_path}")


if __name__ == "__main__":
    pbp = pd.read_csv(RAW_DATA_DIR / "pbp_raw.csv", low_memory=False)
    schedules = pd.read_csv(RAW_DATA_DIR / "schedules_raw.csv")

    features = build_game_level_features(pbp, schedules)
    save_features(features)