"""Expected credit loss from the lifetime PD paths, and a realised-vs-predicted check.

  ECL_H = sum_{k<H} m_k * LGD * EAD_k * DF_k
    m_k   marginal default probability for month t+k+1 (lifetime.marginal_pd)
    EAD_k scheduled balance after k payments (level-payment amortisation from the
          current balance, note rate and remaining term)
    DF_k  (1 + r/12)^-(k+1), r = the loan's own rate (IFRS 9 effective interest rate)
    LGD   one pooled value from defaults resolved before the test window starts

The realised counterpart of ECL_12 is def_12 * EAD_0 * LGD, which uses the same
LGD and exposure so that the comparison isolates the PD model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pooled_lgd(losses: pd.DataFrame, known_by: int) -> dict:
    """Loss / UPB at default over defaults disposed of (with a loss figure) by month known_by."""
    d = losses[(losses["disposal_t"] <= known_by) & losses["actual_loss"].notna()
               & (losses["upb_before_default"] > 0)]
    if len(d) == 0:
        return dict(lgd=np.nan, n=0)
    loss = d["actual_loss"].to_numpy(float)
    if np.median(loss) < 0:      # guide: Actual Loss is positive for a loss; guard against a sign flip
        loss = -loss
    lgd = float(np.clip(loss.sum() / d["upb_before_default"].sum(), 0, 1))
    return dict(lgd=lgd, n=len(d), median_loss_rate=float(np.median(loss / d["upb_before_default"])))


def exposure(rows: pd.DataFrame, loans: pd.DataFrame) -> np.ndarray:
    orig = loans["orig_upb"].reindex(rows["loan_id"]).to_numpy(float)
    cur = rows["current_upb"].to_numpy(float)
    return np.where(cur > 0, cur, orig)


def amortisation(rows: pd.DataFrame, loans: pd.DataFrame, horizon: int) -> np.ndarray:
    """Balance after k scheduled payments as a fraction of today's, k = 0..horizon-1."""
    term = loans["orig_loan_term"].reindex(rows["loan_id"]).to_numpy(float)
    n = np.maximum(term - rows["loan_age"].to_numpy(float), 1)
    r = rows["current_interest_rate"].to_numpy(float) / 1200
    k = np.arange(horizon)[None, :]
    g = (1 + r)[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        f = (g ** n[:, None] - g ** k) / (g ** n[:, None] - 1)
    f = np.where(r[:, None] > 0, f, 1 - k / n[:, None])
    return np.clip(np.where(k < n[:, None], f, 0.0), 0, 1)


def ecl(m: np.ndarray, rows: pd.DataFrame, loans: pd.DataFrame, lgd: float, horizons: list[int]) -> pd.DataFrame:
    H = m.shape[1]
    ead = exposure(rows, loans)[:, None] * amortisation(rows, loans, H)
    r = rows["current_interest_rate"].to_numpy(float) / 1200
    df = (1 + r)[:, None] ** -(np.arange(H)[None, :] + 1)
    loss = np.cumsum(m * lgd * ead * df, axis=1)
    return pd.DataFrame({f"ecl_{h}": loss[:, min(h, H) - 1] for h in horizons})


def realised_vs_predicted(rows: pd.DataFrame, loans: pd.DataFrame, ecl12: np.ndarray, lgd: float,
                          label="def_12") -> pd.DataFrame:
    ok = rows[label].notna().to_numpy()
    d = pd.DataFrame({"t": rows["t"].to_numpy()[ok],
                      "ead": exposure(rows, loans)[ok],
                      "pred": ecl12[ok],
                      "real": rows[label].to_numpy(float)[ok] * exposure(rows, loans)[ok] * lgd})
    g = d.groupby("t").sum()
    g["pred_rate"] = g["pred"] / g["ead"]
    g["real_rate"] = g["real"] / g["ead"]
    g["ratio"] = g["pred"] / g["real"].replace(0, np.nan)
    return g.reset_index()
