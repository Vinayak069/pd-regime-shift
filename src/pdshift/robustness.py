"""Robustness study: seeds, a forbearance-free default definition, early-label and drift-triggered recalibration.

For each seed the training sample, landmark phases and evaluation snapshots are drawn
again (prepare(seed=...)) and B0 and M1 are refitted, so the spread across seeds covers
both sampling and model randomness. The 3- and 12-month PDs come from the persistence
(baseline) macro path, the one a lender could deploy. Each model is scored against two
default definitions:

  all  def_12      the main definition (90+ days delinquent or a credit zero balance)
  xf   def_12_xf   delinquency reported under forbearance or disaster relief not counted

and recalibrated with the label-delay-safe methods of recalibrate.py (static, rolling,
early, trig_z, trig_psi). Recalibration fitted for one definition uses that definition's
labels.

  python -m pdshift.robustness --exp covid [--seeds 5] [--boot 1000]

Writes results/<exp>/robustness.csv (one row per seed, model, definition, method),
robustness_monthly.csv and robustness_triggers.csv (first seed, M1), robustness_bootstrap.csv
(first seed, M1, loan-clustered) and robustness_info.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as cfg
from . import dataset, lifetime, metrics, recalibrate
from .experiment import evaluate, fit_model, prepare

log = logging.getLogger("pdshift")
LABELS = {"all": ("def_12", "def_3"), "xf": ("def_12_xf", "def_3_xf")}
STANDARD = [("none", "static"), ("intercept", "static"), ("intercept", "rolling"),
            ("platt", "static"), ("platt", "rolling")]


def recalibrations(d: pd.DataFrame, test_months, delay: int, window: int,
                   hz: tuple | None = None) -> tuple[dict, pd.DataFrame]:
    """{(method, mode): recalibrated PDs on test rows} and the early/trigger log."""
    out, logs = {}, []
    for method, mode in STANDARD:
        out[(method, mode)] = recalibrate.recalibrate(d, test_months, method, mode, delay, window)
    for mode in recalibrate.EARLY_MODES + recalibrate.TREND_MODES:
        out[("intercept", mode)], lg = recalibrate.recalibrate_early(d, test_months, mode, delay, window)
        logs.append(lg)
    if hz is not None:
        for mode in recalibrate.HZ_MODES:
            out[("hazard", mode)], lg = recalibrate.recalibrate_early(d, test_months, mode, delay, window, hz=hz)
            logs.append(lg)
    return out, pd.concat(logs, ignore_index=True)


def run(name: str, n_seeds: int = 5, n_boot: int = 1000, models_: tuple = ("B0", "M1"),
        out_dir: Path | None = None, **prep_kw) -> Path:
    t_start = time.time()
    c = cfg.load_all()
    out = Path(out_dir or cfg.ROOT / "results" / name)
    out.mkdir(parents=True, exist_ok=True)
    delay = c["splits"]["label_delay_months"]
    window = c["models"]["recalibration"]["window_months"]
    seeds = c["models"]["seeds"][:n_seeds]
    rows, monthly, trig, boot, info = [], [], [], [], {}
    for seed in seeds:
        t0 = time.time()
        ctx = prepare(name, c, seed=seed, **prep_kw)
        snaps, T0 = ctx.snaps, ctx.T0
        test = (snaps["t"] >= T0).to_numpy()
        test_months = sorted(snaps.loc[test, "t"].unique())
        info[seed] = dict(ctx.info, train_vintages=None)
        for key in models_:
            model = fit_model(ctx, key, seed)
            m, S = lifetime.marginal_pd(model, snaps, ctx.loans, ctx.macro, 13)
            pdf = pd.DataFrame({"pd_3": m[:, :3].sum(1), "pd_12": m[:, :12].sum(1)})
            # monthly hazards back out of the path: hd_k = m_k / S_k, hp_k = 1 - hd_k - S_{k+1} / S_k
            hd = (m[:, :12] / S[:, :12]).astype("float32")
            hz = (hd, (1 - hd - S[:, 1:13] / S[:, :12]).clip(0, 1).astype("float32"))
            for lab, (y12, y3) in LABELS.items():
                d = snaps[["loan_id", "t"]].assign(p=pdf["pd_12"].to_numpy(), p3=pdf["pd_3"].to_numpy(),
                                                   y=snaps[y12].to_numpy(), y3=snaps[y3].to_numpy())
                rec, lg = recalibrations(d, test_months, delay, window, hz)
                for (method, mode), pr in rec.items():
                    dd = d[test].assign(p_recal=pr[test])
                    rows.append(dict(seed=seed, model=key, label=lab, method=method, mode=mode,
                                     **evaluate(dd, "y", "p_recal")))
                if seed == seeds[0] and key == "M1":
                    trig.append(lg.assign(label=lab))
                    dt = d[test]
                    mt = dt.groupby("t").agg(obs=("y", "mean"), n=("y", "count"))
                    for (method, mode), pr in rec.items():
                        mt[f"{method}|{mode}"] = pr[test].groupby(dt["t"]).mean()
                    monthly.append(mt.reset_index().assign(label=lab))
                    if n_boot:
                        base = rec[("intercept", "rolling")][test]
                        for (method, mode) in [("none", "static"), ("intercept", "early"), ("intercept", "trig_z"),
                                               ("intercept", "trig_trend")]:
                            # paired: AUC(method) - AUC(rolling intercept) on the same resampled loans
                            db = dt.assign(p_recal=rec[(method, mode)][test], p_ref=base)
                            db = db[db["y"].notna()]
                            fn = lambda x: roc_auc_score(x["y"], x["p_recal"]) - roc_auc_score(x["y"], x["p_ref"])
                            boot.append(dict(label=lab, method=method, mode=mode, stat="auc_minus_rolling",
                                             **metrics.cluster_bootstrap(db, fn, n_boot=n_boot, seed=seed)))
                        for (method, mode) in [("none", "static"), ("intercept", "rolling"),
                                               ("intercept", "early"), ("intercept", "trig_z"),
                                               ("intercept", "trig_trend")]:
                            db = dt.assign(p_recal=rec[(method, mode)][test])
                            db = db[db["y"].notna()]
                            for stat, fn in (("auc", lambda x: roc_auc_score(x["y"], x["p_recal"])),
                                             ("ratio", lambda x: x["p_recal"].mean() / x["y"].mean())):
                                boot.append(dict(label=lab, method=method, mode=mode, stat=stat,
                                                 **metrics.cluster_bootstrap(db, fn, n_boot=n_boot, seed=seed)))
            log.info("%s seed %d %s done (%.0fs since seed start)", name, seed, key, time.time() - t0)
        pd.DataFrame(rows).to_csv(out / "robustness.csv", index=False)   # partial results survive a crash
    pd.concat(monthly).to_csv(out / "robustness_monthly.csv", index=False)
    pd.concat(trig).to_csv(out / "robustness_triggers.csv", index=False)
    if boot:
        pd.DataFrame(boot).to_csv(out / "robustness_bootstrap.csv", index=False)
    (out / "robustness_info.json").write_text(json.dumps(dict(
        experiment=name, seeds=seeds, models=list(models_), n_boot=n_boot,
        runtime_s=round(time.time() - t_start), per_seed=info), indent=2, default=str))
    log.info("robustness for %s done in %.0fs", name, time.time() - t_start)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="seeds, forbearance-free labels, early and triggered recalibration")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--models", nargs="+", default=["B0", "M1"])
    ap.add_argument("--processed", default=str(dataset.PROCESSED))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.exp, args.seeds, args.boot, tuple(args.models), Path(args.out) if args.out else None,
        processed=Path(args.processed))


if __name__ == "__main__":
    main()
