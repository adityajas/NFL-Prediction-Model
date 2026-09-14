"""
yards_model.py

Predicts total RUSHING and RECEIVING yards for both the home and away team
in a given game, using the same leakage-safe rolling features built in
features.py (season-to-date and last-3/last-5-game averages of EPA,
success rate, situational splits, and the teams' own rolling yardage).

Trains FOUR separate models (walk-forward across seasons, same approach as
model.py):
    - home_actual_off_rush_yards
    - home_actual_off_rec_yards
    - away_actual_off_rush_yards
    - away_actual_off_rec_yards

Each model is trained only on rolling/pre-game features -- never on the
actual_* columns of ANY team for that game, since those are the ground
truth outcomes we're trying to predict.
"""

import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor

PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

FIRST_TEST_SEASON = 2018

TARGET_COLS = [
    "home_actual_off_rush_yards",
    "home_actual_off_rec_yards",
    "away_actual_off_rush_yards",
    "away_actual_off_rec_yards",
]


def load_clean_features() -> pd.DataFrame:
    """Loads game_features.csv and drops rows with missing rolling stats
    (early-season games with no prior data to average -- see features.py)."""
    df = pd.read_csv(PROCESSED_DATA_DIR / "game_features.csv")
    df = df.dropna()
    return df


def engineer_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds explicit matchup-adjusted features: a team's own rolling
    offensive yardage average blended with the opponent's rolling
    yardage-ALLOWED average. Handed to the model directly rather than
    relying on it to discover this relationship on its own from 100+
    unrelated inputs -- testing showed this simple blend alone already
    beats the model's raw predictions (see conversation notes), so the
    goal here is to let the model use it as a strong starting signal.

    Built for all three rolling windows (season-to-date, last-3, last-5)
    since it's not obvious upfront which window blends best.
    """
    df = df.copy()
    windows = ["szn_avg", "last3", "last5"]

    for w in windows:
        df[f"matchup_home_rush_{w}"] = (
            df[f"home_off_rush_yards_{w}"] + df[f"away_def_rush_yards_allowed_{w}"]
        ) / 2
        df[f"matchup_home_rec_{w}"] = (
            df[f"home_off_rec_yards_{w}"] + df[f"away_def_rec_yards_allowed_{w}"]
        ) / 2
        df[f"matchup_away_rush_{w}"] = (
            df[f"away_off_rush_yards_{w}"] + df[f"home_def_rush_yards_allowed_{w}"]
        ) / 2
        df[f"matchup_away_rec_{w}"] = (
            df[f"away_off_rec_yards_{w}"] + df[f"home_def_rec_yards_allowed_{w}"]
        ) / 2

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Selects only the leakage-safe rolling/season-to-date feature columns.
    Excludes identifiers, raw scores, the spread, non-numeric date/season
    columns that leaked through the merge, and -- critically -- ALL FOUR
    actual_* target columns, since none of them should ever be used to
    predict any of the others.
    """
    exclude = {
        "game_id", "season", "week", "home_team", "away_team",
        "home_score", "away_score", "spread_line", "total_line",
        "home_gameday", "away_gameday",
        "home_season", "away_season", "home_week", "away_week",
    }
    exclude.update(TARGET_COLS)
    return [c for c in df.columns if c not in exclude]


def get_stripped_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    A deliberately narrow feature set: the matchup-blend features (offense
    rolling average blended with opponent's defense-allowed rolling
    average, all 3 windows) plus a handful of core EPA rolling averages
    and rest days. Everything else -- 3rd down rate, red zone rate, pass
    rate, etc. -- is dropped.

    The hypothesis being tested: the full 134-feature model doesn't beat
    the simple matchup-blend baseline because it has too much noisy,
    weakly-relevant input drowning out the one signal that matters. A
    much leaner model, trained on only the most plausibly-relevant
    features, should have an easier time matching or beating that baseline
    if there's any learnable improvement to be found at all.
    """
    keep_patterns = [
        "matchup_",
        "off_epa_per_play_szn_avg", "off_epa_per_play_last3",
        "off_rush_epa_szn_avg", "off_pass_epa_szn_avg",
        "def_epa_per_play_allowed_szn_avg", "def_epa_per_play_allowed_last3",
        "def_rush_epa_allowed_szn_avg", "def_pass_epa_allowed_szn_avg",
    ]
    rest_cols = ["home_rest", "away_rest"]

    stripped = [
        c for c in df.columns
        if any(p in c for p in keep_patterns) and c not in TARGET_COLS
    ]
    stripped += [c for c in rest_cols if c in df.columns]
    return sorted(set(stripped))


def walk_forward_predict_yards(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    For each season from FIRST_TEST_SEASON onward, trains one model per
    target (rushing/receiving yards, home/away) on all seasons strictly
    before it, then predicts that season. Returns game info plus four
    prediction columns, one per target, prefixed "pred_".
    """
    df = df.sort_values(["season", "week"]).reset_index(drop=True)

    all_predictions = []
    test_seasons = sorted(s for s in df["season"].unique() if s >= FIRST_TEST_SEASON)

    for season in test_seasons:
        train = df[df["season"] < season]
        test = df[df["season"] == season].copy()

        if len(train) == 0 or len(test) == 0:
            continue

        for target in TARGET_COLS:
            model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            model.fit(train[feature_cols], train[target])
            test[f"pred_{target}"] = model.predict(test[feature_cols])

        all_predictions.append(test)
        print(f"Season {season}: trained on {len(train):,} games, predicted {len(test):,} games.")

    return pd.concat(all_predictions, ignore_index=True)


def save_predictions(predictions: pd.DataFrame, filename: str = "yards_predictions.csv") -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    predictions.to_csv(out_path, index=False)
    print(f"Saved {len(predictions):,} rows to {out_path}")


if __name__ == "__main__":
    df = load_clean_features()
    df = engineer_matchup_features(df)

    baseline_cols = {
        "home_actual_off_rush_yards": ("home_off_rush_yards_szn_avg", "away_def_rush_yards_allowed_szn_avg"),
        "home_actual_off_rec_yards": ("home_off_rec_yards_szn_avg", "away_def_rec_yards_allowed_szn_avg"),
        "away_actual_off_rush_yards": ("away_off_rush_yards_szn_avg", "home_def_rush_yards_allowed_szn_avg"),
        "away_actual_off_rec_yards": ("away_off_rec_yards_szn_avg", "home_def_rec_yards_allowed_szn_avg"),
    }

    feature_sets = {
        "full (134 features)": get_feature_columns(df),
        "stripped-down": get_stripped_feature_columns(df),
    }

    results_by_set = {}
    for label, feature_cols in feature_sets.items():
        print(f"\n=== Training with {label}: {len(feature_cols)} features ===")
        predictions = walk_forward_predict_yards(df, feature_cols)
        results_by_set[label] = predictions

        if label == "stripped-down":
            keep_cols = [
                "game_id", "season", "week", "home_team", "away_team",
            ] + TARGET_COLS + [f"pred_{t}" for t in TARGET_COLS]
            save_predictions(predictions[keep_cols])

    print("\n--- MAE comparison: baseline vs. full model vs. stripped-down model ---")
    # Use the SAME rows the models were actually tested on (2018-2024 walk-
    # forward predictions), not the full df (which also includes
    # 2015-2017 train-only rows) -- otherwise the baseline comparison
    # would be computed on a different, larger set of games than the
    # models were, making it an unfair comparison.
    test_rows = results_by_set["full (134 features)"]
    for target, (off_col, def_col) in baseline_cols.items():
        baseline_mae = (
            (test_rows[target] - (test_rows[off_col] + test_rows[def_col]) / 2).abs().mean()
        )
        full_mae = (
            results_by_set["full (134 features)"][target]
            - results_by_set["full (134 features)"][f"pred_{target}"]
        ).abs().mean()
        stripped_mae = (
            results_by_set["stripped-down"][target]
            - results_by_set["stripped-down"][f"pred_{target}"]
        ).abs().mean()
        print(
            f"{target:35s} baseline={baseline_mae:6.1f}  "
            f"full_model={full_mae:6.1f}  stripped_model={stripped_mae:6.1f}"
        )