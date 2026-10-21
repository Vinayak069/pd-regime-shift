"""Model matrix for a set of (loan, month) rows.

The macro values for a row are looked up at `macro_at` (default: the row's own
month t), which is how lifetime.py rolls features forward along a macro path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def feature_names(fcfg: dict, sets: list[str]) -> tuple[list[str], list[str]]:
    """(numeric, categorical) feature lists for a model spec's feature sets."""
    num, cat = [], []
    for s in sets:
        if s == "loan_static":
            num += fcfg["loan_static"]["numeric"]
            cat += fcfg["loan_static"]["categorical"]
        elif s == "loan_dynamic":
            num += fcfg["loan_dynamic"]
        elif s == "macro":
            num += fcfg["macro_features"] + fcfg["derived"]
        else:
            raise ValueError(f"unknown feature set {s!r}")
    return num, cat


def assemble(rows: pd.DataFrame, loans: pd.DataFrame, macro: pd.DataFrame,
             macro_at: np.ndarray | None = None, age_shift: np.ndarray | int = 0) -> pd.DataFrame:
    """Join loan attributes and macro values onto rows (needs loan_id, t and the dynamic columns).

    loans must be indexed by loan_id. macro is the table from macro.load_macro().
    """
    X = rows.join(loans.drop(columns=[c for c in loans.columns if c in rows.columns]), on="loan_id")
    at = rows["t"].to_numpy() if macro_at is None else np.asarray(macro_at)
    m = macro.reindex(at)
    for c in macro.columns:
        X[c] = m[c].to_numpy()
    X["loan_age"] = X["loan_age"] + age_shift
    if "k" not in X:
        X["k"] = 0
    X["upb_ratio"] = (X["current_upb"] / X["orig_upb"]).where(X["current_upb"] > 0)
    X["refi_incentive"] = X["current_interest_rate"] - X["mortgage30us"]
    fp_idx = cfg.ym_to_idx(X["first_payment_date"].fillna(0).astype("int64"))
    hpi_orig = macro["csushpinsa_idx"].reindex(fp_idx.to_numpy()).to_numpy()
    X["mtm_ltv"] = X["orig_ltv"] * hpi_orig / X["csushpinsa_idx"]
    return X
