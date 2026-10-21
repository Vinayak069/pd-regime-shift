"""Monthly cause-specific hazard models for default and prepayment.

Each model is a binary classifier on landmark pairs (dataset.landmark_pairs): with
the loan state seen at landmark month t, it predicts default (y_def) or prepayment
(y_pre) in month t+k+1 for a loan still at risk at t+k.
Together they give the competing-risks lifetime PD in lifetime.py.

  logit_hazard     impute + missing indicators, scaling, splines on loan age and
                   credit score, one-hot categoricals, L2 logistic regression
  lightgbm_hazard  gradient-boosted trees, native categoricals, early stopping
No resampling or class weights: both would bias the predicted probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

SPLINE_COLS = ["loan_age", "credit_score", "k"]

LGB_DEFAULTS = dict(objective="binary", n_estimators=2000, learning_rate=0.05, num_leaves=63,
                    min_child_samples=500, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                    reg_lambda=1.0, verbose=-1, n_jobs=-1)


@dataclass
class Hazard:
    kind: str
    numeric: list[str]
    categorical: list[str]
    params: dict = field(default_factory=dict)
    seed: int = 0
    model: object = None

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X[self.numeric + self.categorical].copy()
        for c in self.categorical:
            Z[c] = Z[c].astype("string").fillna("NA").astype("category")
        return Z

    def fit(self, X: pd.DataFrame, y: np.ndarray, X_val=None, y_val=None) -> "Hazard":
        Z = self._frame(X)
        if self.kind == "logit_hazard":
            self.model = self._logit().fit(Z, y)
        elif self.kind == "lightgbm_hazard":
            p = {**LGB_DEFAULTS, **self.params, "random_state": self.seed}
            self.model = lgb.LGBMClassifier(**p)
            kw = {}
            if X_val is not None:
                Zv = self._frame(X_val)
                for c in self.categorical:
                    Zv[c] = pd.Categorical(Zv[c].astype("string"), categories=Z[c].cat.categories)
                kw = dict(eval_X=(Zv,), eval_y=(y_val,), callbacks=[lgb.early_stopping(100, verbose=False)])
            self.model.fit(Z, y, **kw)
            self._cats = {c: Z[c].cat.categories for c in self.categorical}
        else:
            raise ValueError(f"unknown model type {self.kind!r}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._frame(X)
        if self.kind == "lightgbm_hazard":
            for c in self.categorical:
                Z[c] = pd.Categorical(Z[c].astype("string"), categories=self._cats[c])
        return self.model.predict_proba(Z)[:, 1]

    def _logit(self):
        spl = [c for c in SPLINE_COLS if c in self.numeric]
        lin = [c for c in self.numeric if c not in spl]
        tf = ColumnTransformer([
            ("lin", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), lin),
            ("spl", make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=6, degree=3)), spl),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.001), self.categorical),
        ])
        C = self.params.get("C", 1.0)
        return make_pipeline(tf, LogisticRegression(C=C, max_iter=500, tol=1e-5))


@dataclass
class CompetingRisks:
    """A default hazard and a prepayment hazard fitted on the same rows."""
    default: Hazard
    prepay: Hazard

    def fit(self, X, y_def, y_pre, X_val=None, yv_def=None, yv_pre=None) -> "CompetingRisks":
        self.default.fit(X, y_def, X_val, yv_def)
        self.prepay.fit(X, y_pre, X_val, yv_pre)
        return self

    def hazards(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        hd, hp = self.default.predict(X), self.prepay.predict(X)
        tot = hd + hp
        scale = np.where(tot > 0.999, 0.999 / np.maximum(tot, 1e-12), 1.0)  # keep survival positive
        return hd * scale, hp * scale


def build(spec: dict, numeric: list[str], categorical: list[str], seed: int = 0,
          params: dict | None = None, prepay_params: dict | None = None) -> CompetingRisks:
    kind = spec["type"]
    return CompetingRisks(
        Hazard(kind, numeric, categorical, dict(params or spec.get("params", {})), seed),
        Hazard(kind, numeric, categorical, dict(prepay_params or spec.get("params", {})), seed),
    )
