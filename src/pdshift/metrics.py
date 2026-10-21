"""Discrimination and calibration metrics, DeLong test and loan-clustered bootstrap.

ECE uses equal-mass bins, which suits PDs that sit mostly below a few percent.
Calibration intercept and slope come from a logistic regression of the outcome
on logit(p) (slope 1, intercept 0 when calibrated); the intercept is estimated
with the slope fixed at 1 (calibration-in-the-large on the log-odds scale).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in parts if len(b)) / len(y))


def _loglik(eta, y):
    return float(np.sum(y * eta - np.logaddexp(0, eta)))


def _newton_logit(z, y, offset=None, fit_slope=True, iters=100):
    """Logistic regression of y on [1, z] (or [1] with z as offset): Newton with step halving."""
    X = np.column_stack([np.ones_like(z), z]) if fit_slope else np.ones((len(z), 1))
    off = np.zeros_like(z) if offset is None else offset
    b = np.zeros(X.shape[1])
    if not fit_slope:
        # intercept only: solve mean(expit(off + a)) = mean(y) by bracketing; always converges
        from scipy.optimize import brentq
        ybar = y.mean()
        if ybar <= 0 or ybar >= 1:
            return b
        return np.array([brentq(lambda a: expit(off + a).mean() - ybar, -30, 30)])
    ll = _loglik(X @ b + off, y)
    for _ in range(iters):
        mu = expit(X @ b + off)
        W = mu * (1 - mu) + 1e-12
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1]), X.T @ (y - mu))
        lam = 1.0
        while lam > 1e-6:
            nb = b + lam * step
            nll = _loglik(X @ nb + off, y)
            if nll >= ll - 1e-12:
                break
            lam /= 2
        b, done = nb, abs(nll - ll) < 1e-10
        ll = nll
        if done:
            break
    return b


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """(intercept with slope fixed at 1, slope) of the logistic recalibration curve."""
    y = np.asarray(y, float)
    z = logit(np.clip(np.asarray(p, float), EPS, 1 - EPS))
    a = _newton_logit(z, y, offset=z, fit_slope=False)[0]
    slope = _newton_logit(z, y)[1]
    return float(a), float(slope)


def summary(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    a, b = calibration_slope_intercept(y, p)
    out = dict(n=len(y), events=int(y.sum()), obs_rate=y.mean(), pred_rate=p.mean(),
               citl=y.mean() - p.mean(), ratio=p.mean() / max(y.mean(), EPS),
               brier=np.mean((p - y) ** 2), ece=ece(y, p), cal_intercept=a, cal_slope=b)
    out["auc"] = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return out


def monthly_citl(df: pd.DataFrame, y="y", p="p", month="t") -> pd.Series:
    g = df.groupby(month)
    return g[y].mean() - g[p].mean()


def by_month(df: pd.DataFrame, y="y", p="p", month="t") -> pd.DataFrame:
    return pd.DataFrame([{month: t, **summary(g[y], g[p])} for t, g in df.groupby(month)])


# ---------------------------------------------------------------- DeLong (Sun & Xu 2014 fast version)

def _midrank(x):
    return stats.rankdata(x, method="average")


def delong(y, p1, p2) -> dict:
    """AUCs of two scores on the same cases, their difference and its two-sided p-value."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([_midrank(r) for r in pos])
    ty = np.vstack([_midrank(r) for r in neg])
    tz = np.vstack([_midrank(r) for r in np.hstack([pos, neg])])
    auc = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (auc[0] - auc[1]) / np.sqrt(var) if var > 0 else 0.0
    return dict(auc1=auc[0], auc2=auc[1], diff=auc[0] - auc[1], se=float(np.sqrt(max(var, 0))),
                p_value=float(2 * stats.norm.sf(abs(z))))


# ---------------------------------------------------------------- clustered bootstrap

def cluster_bootstrap(df: pd.DataFrame, stat, cluster="loan_id", n_boot=1000, seed=0,
                      alpha=0.05) -> dict:
    """Percentile CI of stat(df) resampling whole clusters (a loan contributes all its rows)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(df[cluster])
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    k = len(uniq)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        ln = sizes[pick]
        # positions start[pick] + 0..len-1 for every picked cluster, without a Python loop
        offs = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
        idx = order[np.repeat(starts[pick], ln) + offs]
        vals.append(stat(df.iloc[idx]))
    vals = np.asarray(vals, float)
    return dict(estimate=float(stat(df)), lo=float(np.nanquantile(vals, alpha / 2)),
                hi=float(np.nanquantile(vals, 1 - alpha / 2)), se=float(np.nanstd(vals, ddof=1)))
