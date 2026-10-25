"""
IPL Win Probability Predictor - Model Training & Evaluation (stage 3)
=====================================================================
Evaluation protocol (chronological, never random):

  1. MODEL SELECTION  fit on seasons <= 2019, validate on 2020-2022.
                      The candidate with the lowest validation log-loss wins.
  2. FINAL EVALUATION refit every candidate on seasons <= 2022 and score it on
                      the untouched hold-out 2023-2025.
  3. The selected model (trained on <= 2022) is saved for the API, together
     with a ``metrics.json`` the dashboard renders in its Model Insights tab.

Metrics: accuracy, ROC-AUC, Brier score and log-loss. For a *probability*
product log-loss / Brier (calibration quality) matter more than accuracy.

Usage:
    python -m src.train_model
"""

import json
import sys
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config as C
from src.models import BlendModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

# Domain knowledge baked into the model: more runs needed can never help the
# chasing team; more wickets in hand and a stronger team never hurt.
MONOTONE = {"runs_remaining": -1, "required_run_rate": -1,
            "wickets_remaining": 1, "elo_diff": 1}


def candidates() -> dict:
    cons = tuple(MONOTONE.get(c, 0) for c in C.FEATURE_COLS)
    original = dict(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric="logloss",
                    random_state=42, verbosity=0, n_jobs=-1)
    # Heavily regularised: ~110 rows per match are strongly correlated, so deep
    # trees memorise individual matches instead of learning cricket.
    tuned = dict(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.7,
                 colsample_bytree=0.7, min_child_weight=500, reg_lambda=20,
                 monotone_constraints=cons, eval_metric="logloss", random_state=42,
                 verbosity=0, n_jobs=-1)
    return {
        "Logistic Regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000)),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=100, n_jobs=-1, random_state=42),
        "XGBoost (original settings)": lambda: XGBClassifier(**original),
        "XGBoost (tuned + monotone)": lambda: XGBClassifier(**tuned),
        "Blend (LR + XGBoost)": lambda: BlendModel(xgb_params=tuned),
    }


def score(y, p) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
    }


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() >= 30:
            rows.append({"predicted": float(p[mask].mean()), "actual": float(y[mask].mean()),
                         "count": int(mask.sum())})
    return rows


def accuracy_by_stage(df: pd.DataFrame, p: np.ndarray) -> list[dict]:
    stages = [("Overs 0-5", 0, 36), ("Overs 6-10", 36, 66), ("Overs 11-15", 66, 96),
              ("Overs 16-20", 96, 120)]
    rows = []
    for label, lo, hi in stages:
        m = ((df["balls_bowled"] >= lo) & (df["balls_bowled"] < hi)).to_numpy()
        y = df[C.TARGET].to_numpy()[m]
        rows.append({"stage": label, "accuracy": float(accuracy_score(y, p[m] >= 0.5)),
                     "brier": float(brier_score_loss(y, p[m])), "rows": int(m.sum())})
    return rows


def run() -> None:
    print("=" * 60 + "\nSTAGE 3 - MODEL TRAINING & EVALUATION\n" + "=" * 60)
    if not C.FEATURES_CSV.is_file():
        sys.exit("  x features.csv missing - run `python -m src.feature_engineering` first")
    df = pd.read_csv(C.FEATURES_CSV)

    fit_a = df[df["season"] < C.VALID_START]
    valid = df[(df["season"] >= C.VALID_START) & (df["season"] <= C.TRAIN_END)]
    fit_b = df[df["season"] <= C.TRAIN_END]
    test = df[df["season"] >= C.TEST_START]
    print(f"  fit(<=2019) {len(fit_a):,} | validation(2020-22) {len(valid):,} | "
          f"final fit(<=2022) {len(fit_b):,} | test(2023-25) {len(test):,}")
    print(f"  test matches: {test['match_id'].nunique()}")

    X = C.FEATURE_COLS
    results, fitted = {}, {}
    for name, make in candidates().items():
        m_val = make().fit(fit_a[X], fit_a[C.TARGET])
        val = score(valid[C.TARGET].to_numpy(), m_val.predict_proba(valid[X])[:, 1])
        m_fin = make().fit(fit_b[X], fit_b[C.TARGET])
        p_test = m_fin.predict_proba(test[X])[:, 1]
        results[name] = {"validation": val, "test": score(test[C.TARGET].to_numpy(), p_test)}
        fitted[name] = (m_fin, p_test)

    best = min(results, key=lambda n: results[n]["validation"]["log_loss"])

    print(f"\n  {'Model':<22}{'val logloss':>12}{'test acc':>10}{'test AUC':>10}"
          f"{'test Brier':>12}{'test logloss':>14}")
    for n, r in results.items():
        mark = "  <-- selected" if n == best else ""
        print(f"  {n:<22}{r['validation']['log_loss']:>12.4f}{r['test']['accuracy']:>10.4f}"
              f"{r['test']['roc_auc']:>10.4f}{r['test']['brier']:>12.4f}"
              f"{r['test']['log_loss']:>14.4f}{mark}")

    model, p_test = fitted[best]
    y_test = test[C.TARGET].to_numpy()

    # -- per-match view: how early does the model "know" the winner? ----------
    checkpoints = {}
    for ball in (0, 30, 60, 90):
        snap = test[test["balls_bowled"] == ball]
        if len(snap):
            ps = model.predict_proba(snap[X])[:, 1]
            checkpoints[f"after_{ball // 6}_overs"] = {
                "accuracy": float(accuracy_score(snap[C.TARGET], ps >= 0.5)),
                "matches": int(len(snap))}

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = dict(zip(X, map(float, model.feature_importances_)))

    metrics = {
        "selected_model": best,
        "protocol": {"model_selection": "fit seasons <=2019, validate 2020-2022",
                     "final_evaluation": "fit <=2022, test 2023-2025 (never used for tuning)"},
        "rows": {"train": int(len(fit_b)), "test": int(len(test)),
                 "test_matches": int(test["match_id"].nunique())},
        "candidates": results,
        "test_by_stage": accuracy_by_stage(test.reset_index(drop=True), p_test),
        "test_by_checkpoint": checkpoints,
        "calibration": calibration_table(y_test, p_test),
        "feature_importance": importances,
        "monotone_constraints": MONOTONE,
        "baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
    }

    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, C.MODEL_PKL)
    C.FEATURES_JSON.write_text(json.dumps(C.FEATURE_COLS, indent=2))
    C.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    _plot_report(metrics, y_test, p_test)

    print(f"\n  stage accuracy: " + ", ".join(
        f"{s['stage']} {s['accuracy']:.1%}" for s in metrics["test_by_stage"]))
    print(f"  saved model -> {C.MODEL_PKL.name}, metrics -> {C.METRICS_JSON.name}")


def _plot_report(metrics: dict, y, p) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cal = metrics["calibration"]
    ax[0].plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax[0].plot([c["predicted"] for c in cal], [c["actual"] for c in cal], "o-",
               color="#2563eb", label="model")
    ax[0].set(title="Calibration (2023-25 hold-out)", xlabel="Predicted win prob",
              ylabel="Observed win rate")
    ax[0].legend()
    st = metrics["test_by_stage"]
    ax[1].bar([s["stage"] for s in st], [s["accuracy"] for s in st], color="#22d3ee")
    ax[1].axhline(metrics["baseline_accuracy"], ls="--", color="grey")
    ax[1].set(title="Accuracy by stage of the chase", ylim=(0.4, 1))
    ax[1].tick_params(axis="x", rotation=20)
    fi = metrics.get("feature_importance") or {}
    if fi:
        items = sorted(fi.items(), key=lambda kv: kv[1])
        ax[2].barh([k for k, _ in items], [v for _, v in items], color="#6366f1")
        ax[2].set(title="Feature importance")
    fig.tight_layout()
    fig.savefig(C.MODELS_DIR / "model_report.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
