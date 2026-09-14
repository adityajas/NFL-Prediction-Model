"""
model.py

Trains a model to predict home team margin (home_score - away_score) from
pre-game rolling features, then walk-forward validates across seasons so
each season is predicted using ONLY data from prior seasons -- never
trained on the future.

Walk-forward validation matters more than a random train/test split here:
NFL team performance drifts season to season (rosters, coaches, scheme
changes), so a random split would let the model "peek" at a team's later-
season form when predicting its earlier games, inflating accuracy in a
way that won't hold up in real, forward-looking use.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Only train on seasons where the model has at least one prior season of
# data to learn from.
FIRST_TEST_SEASON = 2018


def load_clean_features() -> pd.DataFrame:
    """Loads game_features.csv and drops rows with missing rolling stats
    (early-season games with no prior data to average -- see features.py)."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "game_features.csv")
    df = df.dropna()
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Selects only the leakage-safe rolling/season-to-date feature columns --
    excludes identifiers, raw scores, and the spread itself (the model
    should learn to predict the outcome independently, then we compare
    that prediction to the spread afterward).
    """
    exclude = {
        "game_id", "season", "week", "home_team", "away_team",
        "home_score", "away_score", "spread_line", "total_line",
        "home_gameday", "away_gameday",
        "home_season", "away_season", "home_week", "away_week",
    }
    return [c for c in df.columns if c not in exclude]


def walk_forward_predict(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    For each season from FIRST_TEST_SEASON onward, trains on all seasons
    strictly before it and predicts that season's games. Returns the
    original game info plus a new column: xgb_pred_margin.
    """
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    df["home_margin"] = df["home_score"] - df["away_score"]

    all_predictions = []
    test_seasons = sorted(s for s in df["season"].unique() if s >= FIRST_TEST_SEASON)

    for season in test_seasons:
        train = df[df["season"] < season]
        test = df[df["season"] == season]

        if len(train) == 0 or len(test) == 0:
            continue

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(train[feature_cols], train["home_margin"])

        test = test.copy()
        test["xgb_pred_margin"] = model.predict(test[feature_cols])
        all_predictions.append(test)

        print(f"Season {season}: trained on {len(train):,} games, predicted {len(test):,} games.")

    return pd.concat(all_predictions, ignore_index=True)


def save_predictions(predictions: pd.DataFrame, filename: str = "predictions.csv") -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    predictions.to_csv(out_path, index=False)
    print(f"Saved {len(predictions):,} rows to {out_path}")


if __name__ == "__main__":
    df = load_clean_features()
    feature_cols = get_feature_columns(df)
    print(f"Training on {len(feature_cols)} features.")

    predictions = walk_forward_predict(df, feature_cols)

    keep_cols = [
        "game_id", "season", "week", "home_team", "away_team",
        "home_score", "away_score", "spread_line", "home_margin",
        "xgb_pred_margin",
    ]
    save_predictions(predictions[keep_cols])