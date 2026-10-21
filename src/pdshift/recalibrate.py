"""Post-hoc recalibration of 12-month PDs that respects label delay.

A 12-month label for a snapshot at month s is known only at s + delay
(delay = label_delay_months = 12). So at test month t a recalibrator may be
fitted only on snapshots s <= t - delay:

  static   fitted once, on snapshots [T0 - delay - window + 1, T0 - delay]
           (T0 = first test month) and applied to every test month
  rolling  refitted at every test month t on [t - delay - window + 1, t - delay]

Methods map an input PD p to a recalibrated PD:
  none       p
  intercept  logit p + a                      (slope fixed at 1)
  platt      a + b logit p
  isotonic   monotone step function of p
  iw_platt   platt with importance weights w(x) = P(current | x) / P(history | x),
             from a domain classifier on loan features at month t (no labels needed)

Early-label recalibration (recalibrate_early) shortens the delay. A 3-month label is
known after 3 months, so at test month t the intercept shift a is fitted on the
model's 3-month PD against 3-month outcomes of snapshots s in [t - 3 - w + 1, t - 3],
and applied to the 12-month PD as logit p12 + a. For small hazards, multiplying
every monthly default hazard by e^a moves both cumulative PDs by about the same
logit shift, which is what carries the correction from 3 to 12 months.

Drift-triggered recalibration uses the early correction only in months where a
detector fires, and the static recalibration otherwise:
  trig_z    the early labels disagree with the model: |O - E| / sqrt(sum p(1 - p)) > z
            over the early window (O, E observed and expected 3-month defaults)
  trig_psi  the 12-month PD distribution at t has moved from the static reference
            window: PSI > psi (label-free)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .metrics import EPS, _newton_logit

METHODS = ["none", "intercept", "platt", "isotonic", "iw_platt"]
IW_FEATURES = ["credit_score", "orig_ltv", "orig_dti", "mtm_ltv", "loan_age", "dlq_now", "refi_incentive"]


def _z(p):
    return logit(np.clip(np.asarray(p, float), EPS, 1 - EPS))


def _weighted_platt(z, y, w):
    X = np.column_stack([np.ones_like(z), z])
    b = np.zeros(2)
    for _ in range(50):
        mu = expit(X @ b)
        W = w * mu * (1 - mu) + 1e-12
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2), X.T @ (w * (y - mu)))
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def importance_weights(hist: pd.DataFrame, cur: pd.DataFrame, cols=IW_FEATURES, clip=(0.05, 20.0)):
    cols = [c for c in cols if c in hist and c in cur]
    X = pd.concat([hist[cols], cur[cols]], ignore_index=True).to_numpy(float)
    d = np.r_[np.zeros(len(hist)), np.ones(len(cur))]
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))
    clf.fit(X, d)
    q = np.clip(clf.predict_proba(X[: len(hist)])[:, 1], 1e-4, 1 - 1e-4)
    w = q / (1 - q) * len(hist) / len(cur)
    w = np.clip(w, *clip)
    return w / w.mean()


def fit_apply(method: str, hist: pd.DataFrame, cur: pd.DataFrame, p="p", y="y") -> np.ndarray:
    """Fit on hist (labelled) and return recalibrated PDs for cur."""
    pc = cur[p].to_numpy(float)
    hist = hist[hist[y].notna()]
    if method == "none" or len(hist) == 0 or hist[y].sum() == 0:
        return pc
    zh, yh, zc = _z(hist[p]), hist[y].to_numpy(float), _z(pc)
    if method == "intercept":
        a = _newton_logit(zh, yh, offset=zh, fit_slope=False)[0]
        return expit(zc + a)
    if method == "platt":
        a, b = _newton_logit(zh, yh)
        return expit(a + b * zc)
    if method == "iw_platt":
        a, b = _weighted_platt(zh, yh, importance_weights(hist, cur))
        return expit(a + b * zc)
    if method == "isotonic":
        iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(hist[p], yh)
        return iso.predict(pc)
    raise ValueError(f"unknown recalibration method {method!r}")


def recalibrate(snaps: pd.DataFrame, test_months, method: str, mode: str, delay: int = 12,
                window: int = 12, p="p", y="y") -> pd.Series:
    """Recalibrated PDs for the snapshot rows in test_months; label-delay safe by construction."""
    test_months = sorted(set(test_months))
    out = pd.Series(np.nan, index=snaps.index)
    t0 = test_months[0]
    static_hist = snaps[snaps["t"].between(t0 - delay - window + 1, t0 - delay)]
    for t in test_months:
        cur = snaps[snaps["t"] == t]
        if cur.empty:
            continue
        hist = static_hist if mode == "static" else snaps[snaps["t"].between(t - delay - window + 1, t - delay)]
        assert hist.empty or hist["t"].max() + delay <= t, "label-delay violation"
        out.loc[cur.index] = fit_apply(method, hist, cur, p, y)
    return out


EARLY_MODES = ["early", "trig_z", "trig_psi"]


def _intercept(hist, p, y):
    hist = hist[hist[y].notna()]
    if len(hist) == 0 or hist[y].sum() == 0:
        return 0.0
    z = _z(hist[p])
    return float(_newton_logit(z, hist[y].to_numpy(float), offset=z, fit_slope=False)[0])


def recalibrate_early(snaps: pd.DataFrame, test_months, mode: str, delay: int = 12, window: int = 12,
                      early_delay: int = 3, early_window: int = 6, z: float = 2.0, psi_max: float = 0.10,
                      p="p", y="y", p_early="p3", y_early="y3") -> tuple[pd.Series, pd.DataFrame]:
    """Recalibrated 12-month PDs for test_months, and a per-month log (shift, z, psi, fired)."""
    from .drift import psi as _psi
    test_months = sorted(set(test_months))
    out = pd.Series(np.nan, index=snaps.index)
    t0 = test_months[0]
    ref = snaps[snaps["t"].between(t0 - delay - window + 1, t0 - delay)]
    a_static = _intercept(ref, p, y)
    log = []
    for t in test_months:
        cur = snaps[snaps["t"] == t]
        if cur.empty:
            continue
        hist = snaps[snaps["t"].between(t - early_delay - early_window + 1, t - early_delay)]
        assert hist.empty or hist["t"].max() + early_delay <= t, "label-delay violation"
        h = hist[hist[y_early].notna()]
        pe = h[p_early].to_numpy(float)
        zstat = (h[y_early].sum() - pe.sum()) / np.sqrt(max((pe * (1 - pe)).sum(), 1e-12)) if len(h) else 0.0
        s_psi = _psi(ref[p], cur[p]) if len(ref) else np.nan
        fired = {"early": True, "trig_z": abs(zstat) > z, "trig_psi": bool(s_psi > psi_max)}[mode]
        a = _intercept(hist, p_early, y_early) if fired else a_static
        out.loc[cur.index] = expit(_z(cur[p]) + a)
        log.append(dict(t=t, mode=mode, fired=bool(fired), shift=a, z=float(zstat), psi=float(s_psi)))
    return out, pd.DataFrame(log)
