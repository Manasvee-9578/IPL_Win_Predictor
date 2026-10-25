"""
Custom estimators.

``BlendModel`` averages, in log-odds space, an interpretable Logistic
Regression and a regularised, monotone-constrained XGBoost:

    logit(p) = 0.5 * logit_LR(x) + 0.5 * logit_XGB(x)

Why log-odds space?  Both components are *additive* in log-odds (LR
coefficients, tree SHAP values), so the blend is too.  That lets us explain
any prediction as "base rate + one contribution per feature" with no extra
library - see ``contributions``.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class BlendModel(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_params: dict | None = None, weight_lr: float = 0.5):
        self.xgb_params = xgb_params or {}
        self.weight_lr = weight_lr

    # ---- training -------------------------------------------------------
    def fit(self, X: pd.DataFrame, y):
        self.feature_names_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X)
        self.lr_ = LogisticRegression(max_iter=3000).fit(self.scaler_.transform(X), y)
        self.xgb_ = xgb.XGBClassifier(**self.xgb_params).fit(X, y)
        self.classes_ = self.lr_.classes_
        return self

    # ---- inference ------------------------------------------------------
    def _logit(self, X: pd.DataFrame) -> np.ndarray:
        z_lr = self.lr_.decision_function(self.scaler_.transform(X))
        z_xgb = self.xgb_.predict(X, output_margin=True)
        return self.weight_lr * z_lr + (1 - self.weight_lr) * z_xgb

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = _sigmoid(self._logit(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ---- explanation ----------------------------------------------------
    def contributions(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Per-feature log-odds contributions.

        Returns (frame, base) such that for every row
        ``base + frame.sum(axis=1) == logit(p)``.
        """
        w = self.weight_lr
        lr_part = self.scaler_.transform(X) * self.lr_.coef_[0]
        shap = self.xgb_.get_booster().predict(
            xgb.DMatrix(X, feature_names=self.feature_names_), pred_contribs=True)
        xgb_part, xgb_base = shap[:, :-1], shap[:, -1]

        parts = w * lr_part + (1 - w) * xgb_part
        base = w * self.lr_.intercept_[0] + (1 - w) * xgb_base
        return pd.DataFrame(parts, columns=self.feature_names_, index=X.index), base

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean |contribution|-style importance: blend of normalised LR and XGB importances."""
        lr_imp = np.abs(self.lr_.coef_[0])
        lr_imp = lr_imp / lr_imp.sum()
        xgb_imp = self.xgb_.feature_importances_
        return self.weight_lr * lr_imp + (1 - self.weight_lr) * xgb_imp
