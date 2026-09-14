"""
backtest.py

Evaluates one or more models' predictions against the actual closing spread.

Expects a DataFrame with, at minimum:
    - game_id
    - home_score, away_score          (actual final result)
    - spread_line                     (closing line; positive = home favored,
                                        negative = away favored)
    - one column per model containing that model's PREDICTED home margin,
      e.g. "model_a_pred_margin", "model_b_pred_margin"

For each model, this adds three columns:
    - "{model}_beat_spread"   : "✓" or "✗" — did the model's predicted side
                                 of the spread match the actual side?
    - "{model}_margin_diff"   : how many points better/worse the model's
                                 predicted margin was vs. the actual margin
                                 relative to the spread (signed)
    - "{model}_accuracy_pct"  : that model's raw ATS win rate across all
                                 games evaluated, as a single percentage
                                 (same value repeated per row for convenience
                                 when exporting/reviewing)
"""

import pandas as pd


def evaluate_model_vs_spread(
    df: pd.DataFrame,
    model_pred_columns: list[str],
) -> pd.DataFrame:
    """
    Adds beat-spread, margin-diff, and accuracy-pct columns for each model
    listed in model_pred_columns.

    df must contain: home_score, away_score, spread_line, and each column
    named in model_pred_columns (the model's predicted home_score - away_score).
    """
    df = df.copy()

    actual_margin = df["home_score"] - df["away_score"]
    # Home team covers if actual margin beats the spread.
    # spread_line: positive = home favored, negative = away favored (nflverse convention)
    home_covered = actual_margin > df["spread_line"]
    push = actual_margin == df["spread_line"]

    for model_col in model_pred_columns:
        predicted_margin = df[model_col]

        # Which side did the model predict would cover?
        model_picked_home = predicted_margin > df["spread_line"]

        # Did the model's pick match reality? Pushes don't count as a win or loss.
        correct = (model_picked_home == home_covered) & (~push)

        beat_col = f"{model_col}_beat_spread"
        diff_col = f"{model_col}_margin_diff"
        pct_col = f"{model_col}_accuracy_pct"

        df[beat_col] = correct.map({True: "\u2713", False: "\u2717"})  # ✓ / ✗
        df.loc[push, beat_col] = "push"

        # Margin diff: how far off the model's predicted margin was from
        # the actual margin, signed so positive = model predicted a
        # stronger home performance than what actually happened.
        df[diff_col] = predicted_margin - actual_margin

        # Raw ATS accuracy across all non-push games, repeated per row.
        decided = df[~push]
        if len(decided) == 0:
            accuracy_pct = 0.0
        else:
            model_picked_home_decided = decided[model_col] > decided["spread_line"]
            accuracy_pct = (model_picked_home_decided == home_covered[~push]).mean() * 100

        df[pct_col] = round(accuracy_pct, 2)

    return df


def summarize_models(df: pd.DataFrame, model_pred_columns: list[str]) -> pd.DataFrame:
    """
    Returns a compact one-row-per-model summary table: games evaluated,
    wins, losses, pushes, and ATS win percentage.
    """
    rows = []
    for model_col in model_pred_columns:
        beat_col = f"{model_col}_beat_spread"
        wins = (df[beat_col] == "\u2713").sum()
        losses = (df[beat_col] == "\u2717").sum()
        pushes = (df[beat_col] == "push").sum()
        total_decided = wins + losses
        win_pct = round((wins / total_decided) * 100, 2) if total_decided else 0.0

        rows.append(
            {
                "model": model_col,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "ats_win_pct": win_pct,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Example usage once model.py produces predictions:
    #
    # df = pd.read_csv("data/processed/predictions.csv")
    # results = evaluate_model_vs_spread(df, ["xgb_pred_margin", "linear_pred_margin"])
    # summary = summarize_models(results, ["xgb_pred_margin", "linear_pred_margin"])
    # print(summary)
    raise NotImplementedError(
        "Load real predictions from model.py before running this script directly."
    )