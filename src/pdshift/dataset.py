"""Read the processed panel back with row filters, and split it by loan bucket.

Loans in buckets [0, train_buckets) are the only ones a model is fitted on.
Every evaluation, recalibration and drift sample comes from the remaining
buckets, so recalibration is never fitted on in-sample predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import config as cfg

PROCESSED = cfg.ROOT / "data/processed"


def load_loans(root: Path = PROCESSED) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "loans").set_index("loan_id")


def load_losses(root: Path = PROCESSED) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "losses")


def load_panel(t_lo: int | None, t_hi: int | None, buckets: tuple[int, int],
               columns: list[str] | None = None, root: Path = PROCESSED) -> pd.DataFrame:
    """Rows with t_lo <= t <= t_hi and buckets[0] <= bucket < buckets[1]."""
    d = ds.dataset(Path(root) / "panel", format="parquet")
    f = (ds.field("bucket") >= buckets[0]) & (ds.field("bucket") < buckets[1])
    if t_lo is not None:
        f &= ds.field("t") >= t_lo
    if t_hi is not None:
        f &= ds.field("t") <= t_hi
    return d.to_table(filter=f, columns=columns).to_pandas()


def bucket_counts(t_lo: int | None, t_hi: int | None, root: Path = PROCESSED) -> pd.Series:
    """Number of panel rows per loan bucket with t_lo <= t <= t_hi (reads two narrow columns)."""
    b = load_panel(t_lo, t_hi, (0, 100), columns=["bucket"], root=root)["bucket"]
    return b.value_counts().reindex(range(100), fill_value=0).sort_index()


def load_limited(t_lo, t_hi, buckets: tuple[int, int], max_rows: int | None, seed: int,
                 root: Path = PROCESSED) -> pd.DataFrame:
    """Like load_panel, but reads only as many whole buckets as max_rows needs, then thins by loan."""
    if max_rows:
        n = bucket_counts(t_lo, t_hi, root).loc[buckets[0]: buckets[1] - 1]
        cum = n.cumsum()
        last = int(cum.index[min(int((cum < max_rows).sum()), len(cum) - 1)])
        buckets = (buckets[0], last + 1)
    return thin(load_panel(t_lo, t_hi, buckets, root=root), max_rows, seed)


LANDMARK_STATE = ["current_upb", "current_interest_rate", "dlq_now", "dlq_max_12", "mod_ever", "ever_30dpd"]


def landmark_pairs(rows: pd.DataFrame, every: int, K: int, seed: int) -> pd.DataFrame:
    """Expand at-risk loan-months into (landmark t, later month t+k) pairs, 0 <= k < K.

    Each loan gets a landmark every `every` at-risk months (random phase per loan). A pair carries
    the loan state observed at the landmark (delinquency, balance, rate, modification) and the
    outcome, loan age and calendar month of the later month, plus k. A hazard fitted on these
    pairs predicts P(event in month t+k+1 | at risk at t+k, state at t), which is what a lifetime
    PD made at t needs: the loan state is never assumed frozen at a value the model never saw.
    """
    r = rows.sort_values(["loan_id", "t"], kind="stable").reset_index(drop=True)
    codes, uniq = pd.factorize(r["loan_id"])
    sizes = np.bincount(codes)
    start = np.r_[0, np.cumsum(sizes)[:-1]]
    pos = np.arange(len(r)) - start[codes]
    off = np.random.default_rng(seed).integers(0, every, len(uniq))[codes]
    lm = np.flatnonzero((pos >= off) & ((pos - off) % every == 0))
    n = np.minimum(K, start[codes[lm]] + sizes[codes[lm]] - lm)
    i = np.repeat(lm, n)
    j = i + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    t_i, t_j = r["t"].to_numpy()[i], r["t"].to_numpy()[j]
    keep = (t_j - t_i) < K                      # guards against gaps in a loan's monthly record
    i, j = i[keep], j[keep]
    out = r.iloc[j][["loan_id", "t", "loan_age", "y_def", "y_pre"]].reset_index(drop=True)
    for c in LANDMARK_STATE:
        out[c] = r[c].to_numpy()[i]
    out["k"] = (r["t"].to_numpy()[j] - r["t"].to_numpy()[i]).astype("int16")
    out["t_landmark"] = r["t"].to_numpy()[i]
    return out


def thin(df: pd.DataFrame, max_rows: int | None, seed: int, by: str | None = "loan_id") -> pd.DataFrame:
    """Keep at most max_rows, sampling whole loans (by='loan_id') or whole months (by='t') at random."""
    if not max_rows or len(df) <= max_rows:
        return df
    if by is None:
        return df.sample(max_rows, random_state=seed)
    keys = df[by].unique()
    rng = np.random.default_rng(seed)
    keep = rng.choice(keys, size=max(1, int(len(keys) * max_rows / len(df))), replace=False)
    return df[df[by].isin(keep)]


def snapshots(df: pd.DataFrame, per_month: int, seed: int) -> pd.DataFrame:
    """At most per_month loans in each month, sampled independently month by month."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("t", sort=True):
        parts.append(g if len(g) <= per_month else g.iloc[rng.choice(len(g), per_month, replace=False)])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]
