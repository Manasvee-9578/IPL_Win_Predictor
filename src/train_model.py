"""
IPL Win Probability Predictor — Model Training
================================================
Trains Logistic Regression (baseline) and XGBoost classifiers on
the feature-engineered IPL dataset, evaluates both, and saves the
best model for downstream prediction.

Usage:
    python src/train_model.py
"""

import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")                       # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR    = "models"
INPUT_CSV     = os.path.join(PROCESSED_DIR, "ipl_features.csv")
MODEL_PKL     = os.path.join(MODELS_DIR, "ipl_model.pkl")
FEAT_COLS_PKL = os.path.join(MODELS_DIR, "feature_columns.pkl")
FI_PNG        = os.path.join(MODELS_DIR, "feature_importance.png")

FEATURE_COLS = [
    "runs_remaining",
    "balls_remaining",
    "wickets_remaining",
    "required_run_rate",
    "current_run_rate",
    "run_rate_diff",
    "toss_advantage",
    "venue_encoded",
    "batting_team_form",
    "head_to_head_ratio",
]
TARGET = "batting_team_wins"

# Season ranges for train / test
TRAIN_SEASONS = set(range(2008, 2023))       # 2008 – 2022
TEST_SEASONS  = {2023, 2024}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_season_year(season_val) -> int:
    try:
        s = str(season_val).strip()
        return int(float(s[:4]))
    except (ValueError, IndexError, TypeError):
        return -1


def _print_table(results: dict) -> None:
    """Pretty-print a comparison table of model metrics."""
    header = f"  {'Model':<25s} {'Accuracy':>10s} {'Brier':>10s} {'ROC-AUC':>10s}"
    sep    = "  " + "-" * 57
    print(header)
    print(sep)
    for name, metrics in results.items():
        print(
            f"  {name:<25s}"
            f" {metrics['accuracy']:>10.4f}"
            f" {metrics['brier']:>10.4f}"
            f" {metrics['roc_auc']:>10.4f}"
        )
    print(sep)


def _save_feature_importance(model, feature_names: list, path: str) -> None:
    """Save a horizontal bar chart of XGBoost feature importances."""
    importances = model.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in order],
        importances[order],
        color="#2563eb",
        edgecolor="#1e40af",
    )
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost — Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===========================  MAIN PIPELINE  ==============================

def run() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 — Loading feature data")
    print("=" * 60)

    if not os.path.isfile(INPUT_CSV):
        sys.exit(f"  ✖ File not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  ✔ Loaded {INPUT_CSV}  →  {df.shape}")

    # ------------------------------------------------------------------
    # 2. Season-based train / test split
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Train / Test split by season")
    print("=" * 60)

    df["_season_year"] = df["season"].apply(_parse_season_year)

    train_mask = df["_season_year"].isin(TRAIN_SEASONS)
    test_mask  = df["_season_year"].isin(TEST_SEASONS)

    X_train = df.loc[train_mask, FEATURE_COLS]
    y_train = df.loc[train_mask, TARGET]
    X_test  = df.loc[test_mask, FEATURE_COLS]
    y_test  = df.loc[test_mask, TARGET]

    print(f"  Train seasons : {sorted(TRAIN_SEASONS)}")
    print(f"  Test  seasons : {sorted(TEST_SEASONS)}")
    print(f"  Train size    : {X_train.shape}")
    print(f"  Test  size    : {X_test.shape}")

    if len(X_test) == 0:
        sys.exit("  ✖ No test data found — check season values.")

    # ------------------------------------------------------------------
    # 3. Train models
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 — Training models")
    print("=" * 60)

    # 3a  Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    print("  ✔ Logistic Regression trained")

    # 3b  XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    print("  ✔ XGBoost trained")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4 — Evaluation on test set")
    print("=" * 60)

    results = {}
    for name, model in [("Logistic Regression", lr), ("XGBoost", xgb)]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "brier":    brier_score_loss(y_test, y_prob),
            "roc_auc":  roc_auc_score(y_test, y_prob),
        }

    _print_table(results)

    # ------------------------------------------------------------------
    # 5. Feature importance chart
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5 — Feature importance chart")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)
    _save_feature_importance(xgb, FEATURE_COLS, FI_PNG)
    print(f"  ✔ Saved to {FI_PNG}")

    # ------------------------------------------------------------------
    # 6. Save model & feature columns
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6 — Saving model artefacts")
    print("=" * 60)

    joblib.dump(xgb, MODEL_PKL)
    print(f"  ✔ Best model (XGBoost) → {MODEL_PKL}")

    joblib.dump(FEATURE_COLS, FEAT_COLS_PKL)
    print(f"  ✔ Feature columns      → {FEAT_COLS_PKL}")

    # ------------------------------------------------------------------
    # 7. Sample prediction at over 15
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7 — Sample prediction (over 15 scenario)")
    print("=" * 60)

    # Pick a real test-set row near over 15
    if "over_number" in df.columns:
        sample_pool = df.loc[test_mask & (df["over_number"] == 15)]
    else:
        sample_pool = df.loc[test_mask]

    if len(sample_pool) == 0:
        sample_pool = df.loc[test_mask]

    sample_row = sample_pool.iloc[0]
    sample_X   = sample_row[FEATURE_COLS].values.reshape(1, -1)
    win_prob   = xgb.predict_proba(sample_X)[0, 1]

    print("  Scenario:")
    for feat in FEATURE_COLS:
        print(f"    {feat:>25s}  {sample_row[feat]}")

    # Print context columns if available
    for ctx in ["team_batting", "team_bowling", "match_winner",
                "over_number", "season"]:
        if ctx in sample_row.index:
            print(f"    {ctx:>25s}  {sample_row[ctx]}")

    actual = int(sample_row[TARGET])
    print(f"\n  Predicted win probability : {win_prob:.4f}")
    print(f"  Actual result             : {'WIN ✔' if actual == 1 else 'LOSS ✖'}")

    print("\n✅ Model training complete!")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run()
