"""Read raw Freddie Mac origination and monthly performance files.

Accepts the files as downloaded (sample_YYYY.zip, historical_data_YYYY*.zip) or
already extracted .txt files anywhere under the raw directory.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ORIG_USED = [
    "credit_score", "first_payment_date", "first_time_homebuyer", "mi_pct", "num_units",
    "occupancy_status", "orig_cltv", "orig_dti", "orig_upb", "orig_ltv", "orig_interest_rate",
    "channel", "property_state", "property_type", "loan_id", "loan_purpose", "orig_loan_term",
    "num_borrowers",
]
PERF_USED = [
    "loan_id", "period", "current_upb", "delinquency_status", "loan_age", "modification_flag",
    "zero_balance_code", "current_interest_rate", "actual_loss", "zero_balance_removal_upb",
    "borrower_assistance_status", "delinquency_due_to_disaster",
]
ORIG_NUMERIC = ["credit_score", "first_payment_date", "mi_pct", "num_units", "orig_cltv", "orig_dti",
                "orig_upb", "orig_ltv", "orig_interest_rate", "orig_loan_term", "num_borrowers"]
PERF_NUMERIC = ["period", "current_upb", "loan_age", "current_interest_rate", "actual_loss",
                "zero_balance_removal_upb"]


class LayoutError(ValueError):
    """Raised when a file does not look like the configured Freddie Mac layout."""


def find_vintages(raw_dir: Path) -> dict[str, dict[str, object]]:
    """Map a year tag (e.g. '2005' or '2005Q1') to its origination and performance sources.

    A source is either a Path to a .txt file or a (zip Path, member name) pair.
    """
    found: dict[str, dict[str, object]] = {}
    # sample_orig_YYYY.txt / sample_perf_YYYY.txt (older releases: sample_svcg_YYYY.txt),
    # historical_data_YYYYQn.txt / historical_data_time_YYYYQn.txt
    sample = re.compile(r"sample_(orig|svcg|perf)_(\d{4})\.txt$", re.I)
    full = re.compile(r"historical_data_(time_)?(\d{4}Q\d)\.txt$", re.I)

    def add(name, src):
        if m := sample.search(name):
            kind, tag = ("orig" if m.group(1).lower() == "orig" else "perf"), m.group(2)
        elif m := full.search(name):
            kind, tag = ("perf" if m.group(1) else "orig"), m.group(2)
        else:
            return
        found.setdefault(tag.upper(), {})[kind] = src

    for p in sorted(Path(raw_dir).rglob("*")):
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for member in z.namelist():
                    add(member, (p, member))
        elif p.suffix.lower() == ".txt":
            add(p.name, p)
    return {k: v for k, v in sorted(found.items()) if "orig" in v and "perf" in v}


def _open(src) -> io.TextIOBase:
    if isinstance(src, tuple):
        zpath, member = src
        z = zipfile.ZipFile(zpath)
        return io.TextIOWrapper(z.open(member), encoding="latin-1")
    return open(src, encoding="latin-1")


def _read(src, columns: list[str], used: list[str], chunksize: int | None = None):
    idx = [columns.index(c) for c in used]
    return pd.read_csv(
        _open(src), sep="|", header=None, usecols=idx, names=None, dtype=str,
        keep_default_na=False, na_values=[""], chunksize=chunksize,
    )


def _name(df: pd.DataFrame, columns: list[str], used: list[str]) -> pd.DataFrame:
    idx = sorted(columns.index(c) for c in used)
    df.columns = [columns[i] for i in idx]
    return df


def read_origination(src, layout: dict) -> pd.DataFrame:
    cols = layout["origination"]
    df = _name(_read(src, cols, ORIG_USED), cols, ORIG_USED)
    for c in ORIG_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, codes in layout.get("missing_codes", {}).items():
        if c in df:
            df.loc[df[c].isin(codes), c] = np.nan
    _check_origination(df)
    return df


def read_performance(src, layout: dict, loan_ids: set[str] | None = None,
                     chunksize: int = 2_000_000) -> pd.DataFrame:
    cols = layout["performance"]
    parts = []
    for chunk in _read(src, cols, PERF_USED, chunksize=chunksize):
        chunk = _name(chunk, cols, PERF_USED)
        if loan_ids is not None:
            chunk = chunk[chunk["loan_id"].isin(loan_ids)]
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PERF_USED)
    for c in PERF_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["period"] = df["period"].astype("int64")
    _check_performance(df)
    return df


def _check_origination(df: pd.DataFrame) -> None:
    ids = df["loan_id"].dropna().head(1000)
    if len(ids) and not ids.str.match(r"^[A-Z]\d{2}Q[1-4]\w+$").mean() > 0.95:
        raise LayoutError("loan_id column does not look like Freddie Mac loan sequence numbers; "
                          "check configs/freddie_layout.yaml against the file layout")
    cs = df["credit_score"].dropna()
    if len(cs) and not cs.between(300, 850).mean() > 0.95:
        raise LayoutError("credit_score values out of range; origination column order is wrong")
    fp = df["first_payment_date"].dropna()
    if len(fp) and not fp.between(199001, 210012).mean() > 0.99:
        raise LayoutError("first_payment_date is not YYYYMM; origination column order is wrong")


def _check_performance(df: pd.DataFrame) -> None:
    if df.empty:
        return
    if not df["period"].between(199001, 210012).all():
        raise LayoutError("monthly reporting period is not YYYYMM; performance column order is wrong")
    dlq = df["delinquency_status"].dropna().head(10000)
    ok = dlq.str.fullmatch(r"\d{1,3}|RA|XX").mean()
    if ok < 0.99:
        raise LayoutError("delinquency status values unexpected; performance column order is wrong")
