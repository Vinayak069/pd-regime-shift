"""Monthly hazards -> cumulative (lifetime) PD with prepayment as a competing risk.

For a loan at risk at month t, step k = 0..H-1 predicts the hazards for month
t+k+1 with loan age advanced by k and the macro inputs taken from a path:

  oracle    the realised macro series at t+k (not knowable at t; an upper bound
            on what a perfect macro forecast would give)
  scenario  the values known at t, moved by shocks that ramp in linearly over
            ramp_months; with no shocks this is the persistence forecast.
            The house price index grows at the scenario's yoy rate.

Loan state (delinquency, modification, balance ratio) stays at its time-t value,
and the hazards were fitted that way: on landmark pairs with k = months since t.

  S_0 = 1,  S_{k+1} = S_k (1 - hd_k - hp_k),  PD(H) = sum_{k<H} S_k hd_k
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import assemble
from .models import CompetingRisks

MACRO_COLS = ["unrate", "unrate_d12", "mortgage30us", "csushpinsa_yoy", "csushpinsa_idx", "gdpc1_yoy"]


def _derived(X: pd.DataFrame, hpi_orig: np.ndarray) -> None:
    X["refi_incentive"] = X["current_interest_rate"] - X["mortgage30us"]
    X["mtm_ltv"] = X["orig_ltv"] * hpi_orig / X["csushpinsa_idx"]


def macro_step(base: pd.DataFrame, macro: pd.DataFrame, k: int, path: dict) -> dict[str, np.ndarray]:
    if path.get("kind") == "oracle":
        m = macro.reindex(base["t"].to_numpy() + k)
        return {c: m[c].to_numpy() for c in MACRO_COLS}
    ramp = min(k / path.get("ramp_months", 12), 1.0)
    shocks = path.get("shocks", {})
    out = {c: base[c].to_numpy() + ramp * shocks.get(c, 0.0) for c in MACRO_COLS if c != "csushpinsa_idx"}
    out["unrate_d12"] = base["unrate_d12"].to_numpy() + ramp * shocks.get("unrate", 0.0)
    # integrate monthly HPI growth along the path: sum_{j<k} yoy_j / 12
    r = path.get("ramp_months", 12)
    j = np.arange(k)
    cum_ramp = np.minimum(j / r, 1.0).sum() if k else 0.0
    growth = (k * base["csushpinsa_yoy"].to_numpy() + cum_ramp * shocks.get("csushpinsa_yoy", 0.0)) / 12
    out["csushpinsa_idx"] = base["csushpinsa_idx"].to_numpy() * np.exp(growth)
    return out


def marginal_pd(model: CompetingRisks, rows: pd.DataFrame, loans: pd.DataFrame, macro: pd.DataFrame,
                horizon: int, path: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(m, S): m[:, k] = P(default in month t+k+1), S[:, k] = P(still at risk at t+k)."""
    path = path or {"kind": "scenario"}
    base = assemble(rows, loans, macro).reset_index(drop=True)
    hpi_orig = (base["mtm_ltv"] * base["csushpinsa_idx"] / base["orig_ltv"]).to_numpy()
    n = len(base)
    m, S = np.zeros((n, horizon)), np.ones((n, horizon))
    s = np.ones(n)
    X = base.copy()
    for k in range(horizon):
        X["loan_age"] = base["loan_age"] + k
        X["k"] = k
        for c, v in macro_step(base, macro, k, path).items():
            X[c] = v
        _derived(X, hpi_orig)
        hd, hp = model.hazards(X)
        S[:, k] = s
        m[:, k] = s * hd
        s = s * (1 - hd - hp)
    return m, S


def cumulative_pd(model, rows, loans, macro, horizons: list[int], path: dict | None = None) -> pd.DataFrame:
    m, _ = marginal_pd(model, rows, loans, macro, max(horizons), path)
    c = np.cumsum(m, axis=1)
    return pd.DataFrame({f"pd_{h}": c[:, h - 1] for h in horizons})
