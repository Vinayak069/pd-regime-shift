"""Build the loan-month panel and event labels.

Default = first month 90+ days delinquent, REO acquisition, or a credit-related
zero balance (codes in configs/freddie_layout.yaml). Prepayment is a competing
event; a cause-specific default hazard treats it as censoring.

For every month t at which a loan is at risk (not yet defaulted, not paid off,
fewer than 3 months delinquent) the panel stores:
  y_def, y_pre   event in month t+1 (targets for the monthly hazards)
  def_H          default within H months of t, NaN when censored before t+H
  def_3          the same over 3 months (early labels for recalibration, known at t+3)
  def_H_xf       def_H with relief-driven delinquency not counted: a 90+ day delinquency
                 reported while the loan is in forbearance (borrower assistance code F) or
                 flagged as disaster-related does not count as default; credit zero-balance
                 codes still do. Rows end at the first default under the main definition,
                 so a loan that cures after forbearance leaves the panel there.
  relief_now     loan in forbearance or disaster relief at t
"""
from __future__ import annotations

import argparse
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data import find_vintages, read_origination, read_performance

log = logging.getLogger(__name__)

STATIC = ["loan_id", "vintage", "first_payment_date", "credit_score", "orig_ltv", "orig_cltv", "orig_dti", "orig_upb",
          "orig_interest_rate", "mi_pct", "num_units", "num_borrowers", "orig_loan_term",
          "first_time_homebuyer", "occupancy_status", "channel", "property_type", "loan_purpose",
          "property_state"]


def _dlq_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s.where(~s.isin(["RA", "XX"])), errors="coerce")
    return out.mask(s == "RA", 99)


def label_events(perf: pd.DataFrame, events: dict, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (at-risk panel rows with labels, one row per defaulted loan with loss data)."""
    p = perf.sort_values(["loan_id", "period"]).reset_index(drop=True)
    p["t"] = (p["period"] // 100) * 12 + (p["period"] % 100) - 1
    p["dlq"] = _dlq_numeric(p["delinquency_status"].astype("string"))
    zb = p["zero_balance_code"].astype("string").str.zfill(2)

    is_def = (p["dlq"] >= events["default_dlq_months"]) | zb.isin(events["default_zero_balance_codes"])
    is_pre = zb.isin(events["prepay_zero_balance_codes"])
    is_end = zb.notna()

    g = p.groupby("loan_id", sort=False)["t"]
    first_def = p["t"].where(is_def).groupby(p["loan_id"]).transform("min")
    first_pre = p["t"].where(is_pre).groupby(p["loan_id"]).transform("min")
    first_end = p["t"].where(is_end).groupby(p["loan_id"]).transform("min")
    last_obs = g.transform("max")

    # A prepayment that happens after default is not a competing exit.
    first_pre = first_pre.where(~(first_def <= first_pre))

    at_risk = (p["dlq"].fillna(0) < events["default_dlq_months"]) & ~is_end
    at_risk &= ~(p["t"] >= first_def)          # NaN comparisons are False, so loans never defaulting stay
    at_risk &= ~(p["t"] >= first_pre)
    at_risk &= ~(p["t"] >= first_end)

    p["y_def"] = (first_def == p["t"] + 1).astype("int8")
    p["y_pre"] = (first_pre == p["t"] + 1).astype("int8")

    def _flag(col, code):
        return (p[col].astype("string") == code).fillna(False) if col in p else pd.Series(False, index=p.index)

    relief = _flag("borrower_assistance_status", "F") | _flag("delinquency_due_to_disaster", "Y")
    is_def_xf = ((p["dlq"] >= events["default_dlq_months"]) & ~relief) | zb.isin(events["default_zero_balance_codes"])
    first_def_xf = p["t"].where(is_def_xf).groupby(p["loan_id"]).transform("min")
    first_pre_xf = first_pre.where(~(first_def_xf <= first_pre))

    def _label(fd, fp, h):
        hit = (fd - p["t"]).between(1, h)
        exited_clean = (fp - p["t"]).between(1, h)   # prepaid first: known non-default
        seen = last_obs >= p["t"] + h
        lab = pd.Series(np.nan, index=p.index, dtype="float32")
        lab[seen | exited_clean] = 0.0
        lab[hit] = 1.0
        return lab

    for h in sorted(set(horizons) | {3}):
        p[f"def_{h}"] = _label(first_def, first_pre, h)
    for h in (3, 12):
        p[f"def_{h}_xf"] = _label(first_def_xf, first_pre_xf, h)
    p["relief_now"] = relief.astype("int8")

    # Dynamic features, computed on the full history before rows are dropped.
    p["dlq_now"] = p["dlq"].clip(upper=2).fillna(0).astype("int8")
    p["dlq_c"] = p["dlq"].clip(upper=3)
    p["dlq_max_12"] = (p.groupby("loan_id", sort=False)["dlq_c"].rolling(12, min_periods=1).max()
                       .reset_index(level=0, drop=True).fillna(0).astype("int8"))
    p["mod_ever"] = p["modification_flag"].astype("string").isin(["Y", "P"]).fillna(False).groupby(p["loan_id"]).cummax().astype("int8")
    p["ever_30dpd"] = (p["dlq"].fillna(0) >= 1).groupby(p["loan_id"]).cummax().astype("int8")

    losses = _loss_records(p, first_def, zb, events)

    keep = ["loan_id", "t", "loan_age", "current_upb", "current_interest_rate", "dlq_now",
            "dlq_max_12", "mod_ever", "ever_30dpd", "relief_now", "y_def", "y_pre"] + \
        [f"def_{h}" for h in sorted(set(horizons) | {3})] + ["def_3_xf", "def_12_xf"]
    out = p.loc[at_risk, keep].reset_index(drop=True)
    out = out.astype({"t": "int32", "loan_age": "int16", "current_upb": "float32",
                      "current_interest_rate": "float32"})
    return out, losses


def _loss_records(p, first_def, zb, events) -> pd.DataFrame:
    """Per defaulted loan: UPB at default and the realised loss if the loan was disposed of."""
    d = p[first_def.notna()].copy()
    d["first_def"] = first_def[first_def.notna()]
    at_def = d[d["t"] <= d["first_def"]].groupby("loan_id").agg(
        default_t=("first_def", "first"),
        upb_before_default=("current_upb", lambda s: s[s > 0].iloc[-1] if (s > 0).any() else np.nan),
    )
    term = p[zb.isin(events["default_zero_balance_codes"])].groupby("loan_id").agg(
        disposal_t=("t", "first"), actual_loss=("actual_loss", "first"),
        removal_upb=("zero_balance_removal_upb", "first"),
    )
    return at_def.join(term, how="left").reset_index()


def build_vintage(orig_src, perf_src, layout: dict, horizons: list[int],
                  loans_per_vintage: int | None, seed: int) -> tuple[pd.DataFrame, ...]:
    orig = read_origination(orig_src, layout)
    if loans_per_vintage and len(orig) > loans_per_vintage:
        orig = orig.sample(loans_per_vintage, random_state=seed)
    orig["vintage"] = (orig["first_payment_date"] // 100).astype("int16")
    perf = read_performance(perf_src, layout, loan_ids=set(orig["loan_id"]))
    panel, losses = label_events(perf, layout["events"], horizons)
    # Stable loan bucket 0..99: lets every step draw disjoint, reproducible loan samples.
    orig["bucket"] = loan_bucket(orig["loan_id"])
    panel = panel.merge(orig[["loan_id", "bucket"]], on="loan_id", how="left")
    return orig[STATIC + ["bucket"]].reset_index(drop=True), panel, losses


def loan_bucket(ids: pd.Series) -> pd.Series:
    return ids.map(lambda s: zlib.crc32(s.encode()) % 100).astype("int8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Raw Freddie Mac files -> loan-month panel (parquet)")
    ap.add_argument("--raw", default=str(cfg.ROOT / "data/raw/freddie"))
    ap.add_argument("--out", default=str(cfg.ROOT / "data/processed"))
    ap.add_argument("--loans-per-vintage", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    c = cfg.load_all()
    horizons = c["splits"]["horizons_months"]
    out = Path(args.out)
    for sub in ("loans", "panel", "losses"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    vintages = find_vintages(Path(args.raw))
    if not vintages:
        raise SystemExit(f"no origination/performance file pairs found under {args.raw}")
    for tag, src in vintages.items():
        if (out / "panel" / f"{tag}.parquet").exists():
            log.info("vintage %s already built, skipping", tag)
            continue
        log.info("building vintage %s", tag)
        loans, panel, losses = build_vintage(src["orig"], src["perf"], c["freddie_layout"], horizons,
                                             args.loans_per_vintage, args.seed)
        loans.to_parquet(out / "loans" / f"{tag}.parquet", index=False)
        panel.to_parquet(out / "panel" / f"{tag}.parquet", index=False)
        losses.to_parquet(out / "losses" / f"{tag}.parquet", index=False)
        log.info("vintage %s: %d loans, %d at-risk loan-months, %d defaults",
                 tag, len(loans), len(panel), int(panel["y_def"].sum()))


if __name__ == "__main__":
    main()
