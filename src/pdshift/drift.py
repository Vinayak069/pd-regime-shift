"""Covariate drift between a reference sample and each later month.

  PSI per feature on reference-decile bins (> 0.25 is the usual "major shift" flag)
  C2ST: a domain classifier separating reference rows from month-t rows; its
        cross-validated AUC is 0.5 with no detectable shift
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def psi(ref, cur, bins: int = 10) -> float:
    ref, cur = pd.Series(ref).dropna().to_numpy(float), pd.Series(cur).dropna().to_numpy(float)
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def c2st_auc(ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str], folds: int = 3, seed: int = 0) -> float:
    X = pd.concat([ref[cols], cur[cols]], ignore_index=True).astype(float)
    d = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    oof = np.zeros(len(d))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, d):
        m = lgb.LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1, verbose=-1,
                               random_state=seed).fit(X.iloc[tr], d[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return float(roc_auc_score(d, oof))


def drift_table(ref: pd.DataFrame, snaps: pd.DataFrame, cols: list[str], c2st_cols: list[str] | None = None,
                seed: int = 0) -> pd.DataFrame:
    rows = []
    for t, cur in snaps.groupby("t"):
        r = {"t": t, **{f"psi_{c}": psi(ref[c], cur[c]) for c in cols}}
        if c2st_cols:
            r["c2st_auc"] = c2st_auc(ref, cur, c2st_cols, seed=seed)
        rows.append(r)
    return pd.DataFrame(rows)
