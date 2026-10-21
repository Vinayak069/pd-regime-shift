"""Run one regime experiment end to end: fit, predict lifetime PD, evaluate, recalibrate, ECL.

  python -m pdshift.experiment --exp crisis_2008 [--models B0 B1 M1] [--seeds 1] [--boot 200]

Writes results/<exp>/*.csv and run_info.json. Every number in the paper comes from these files.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as cfg
from . import dataset, drift, ecl, lifetime, metrics, models, recalibrate
from .features import assemble, feature_names
from .macro import load_macro

log = logging.getLogger("pdshift")


@dataclass
class Context:
    name: str
    c: dict
    win: dict
    T0: int
    loans: pd.DataFrame
    macro: pd.DataFrame
    losses: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    snaps: pd.DataFrame
    horizons: list[int]
    info: dict = field(default_factory=dict)


def windows(c: dict, name: str) -> dict:
    e = c["splits"]["experiments"][name]
    return {k: cfg.window(v) for k, v in e.items()}


def prepare(name: str, c: dict, seed: int = 7, processed: Path = dataset.PROCESSED,
            fred: Path | None = None) -> Context:
    win = windows(c, name)
    smp = c["models"]["sampling"]
    tb = smp["train_buckets"]
    delay = c["splits"]["label_delay_months"]
    rwin = c["models"]["recalibration"]["window_months"]
    horizons = c["splits"]["horizons_months"]
    T0, t_hi = win["test"]
    loans = dataset.load_loans(processed)
    macro = load_macro(fred)
    losses = dataset.load_losses(processed)

    # Hazard rows: the month-(t+1) outcome must be observed inside the window. On the Freddie Mac
    # sample, landmark pairs number about 1.35x the loan-months they come from (loans exit early),
    # so reading max_train_rows loan-months and thinning the pairs keeps the budget.
    K = max(c["models"]["ecl"]["lifetime_months"], max(horizons))
    every = smp["landmark_every"]
    rows = dataset.load_limited(None, win["train"][1] - 1, (0, tb), smp["max_train_rows"], seed, processed)
    train = dataset.thin(dataset.landmark_pairs(rows, every, K, seed), smp["max_train_rows"], seed)
    rows = dataset.load_limited(win["validate"][0], win["validate"][1] - 1, (0, tb), smp["max_val_rows"], seed,
                                processed)
    val = dataset.thin(dataset.landmark_pairs(rows, every, K, seed), smp["max_val_rows"], seed)
    del rows
    # From here on, snapshots cover what static recalibration needs for both the validation window
    # (used by selection.py) and the test window.
    lo = win["validate"][0] - delay - rwin + 1
    # Read only enough evaluation buckets for ~3x loans_per_snapshot loans per month on average.
    n = dataset.bucket_counts(lo, t_hi, processed).loc[tb:99]
    per_bucket_month = max(n.mean() / (t_hi - lo + 1), 1)
    k = int(min(len(n), np.ceil(3 * smp["loans_per_snapshot"] / per_bucket_month)))
    snaps = dataset.snapshots(dataset.load_panel(lo, t_hi, (tb, tb + k), root=processed),
                              smp["loans_per_snapshot"], seed)
    # Labels not yet known at T0 are hidden from everything fitted or selected before T0.
    for h in horizons:
        hide = (snaps["t"] < T0) & (snaps["t"] + h > T0 - 1)
        snaps[f"known_{h}"] = snaps[f"def_{h}"].where(~hide)
    ctx = Context(name, c, win, T0, loans, macro, losses,
                  assemble(train, loans, macro), assemble(val, loans, macro), snaps, horizons)
    ctx.info = dict(train_pairs=len(train), train_loans=train["loan_id"].nunique(),
                    train_defaults=int(train["y_def"].sum()), train_prepays=int(train["y_pre"].sum()),
                    val_rows=len(val), snapshot_rows=len(snaps),
                    test_snapshot_rows=int((snaps["t"] >= T0).sum()),
                    train_vintages=sorted(map(int, train["loan_id"].map(loans["vintage"]).unique())))
    return ctx


def fit_model(ctx: Context, key: str, seed: int, params: dict | None = None,
              prepay: models.Hazard | None = None, max_rows: int | None = None) -> models.CompetingRisks:
    spec = ctx.c["models"]["models"][key]
    num, cat = feature_names(ctx.c["features"], spec["features"])
    m = models.build(spec, num, cat, seed, params=params)
    tr = ctx.train if not max_rows else dataset.thin(ctx.train, max_rows, seed)
    m.default.fit(tr, tr["y_def"].to_numpy(), ctx.val, ctx.val["y_def"].to_numpy())
    if prepay is not None:
        m.prepay = prepay
    else:
        m.prepay.fit(tr, tr["y_pre"].to_numpy(), ctx.val, ctx.val["y_pre"].to_numpy())
    return m


def scenario_paths(c: dict) -> dict[str, dict]:
    e = c["models"]["ecl"]
    paths = {"oracle": {"kind": "oracle"}}
    for n, s in e["scenarios"].items():
        paths[n] = {"kind": "scenario", "shocks": s["shocks"], "ramp_months": e["ramp_months"]}
    return paths


def evaluate(df: pd.DataFrame, label: str, pcol: str) -> dict:
    d = df[df[label].notna()]
    out = metrics.summary(d[label], d[pcol])
    citl = metrics.monthly_citl(d, label, pcol)
    out.update(citl_monthly_std=float(citl.std()), citl_monthly_maxabs=float(citl.abs().max()),
               months=int(citl.size))
    return out


def run(name: str, model_keys: list[str], n_seeds: int = 1, n_boot: int = 200,
        lifetime_months: int | None = None, out_dir: Path | None = None, **prep_kw) -> Path:
    t_start = time.time()
    c = cfg.load_all()
    out = Path(out_dir or cfg.ROOT / "results" / name)
    out.mkdir(parents=True, exist_ok=True)
    ctx = prepare(name, c, **prep_kw)
    log.info("%s: %s", name, ctx.info)
    H = max(lifetime_months or c["models"]["ecl"]["lifetime_months"], max(c["splits"]["horizons_months"]))
    delay = c["splits"]["label_delay_months"]
    rc = c["models"]["recalibration"]
    snaps, T0 = ctx.snaps, ctx.T0
    test = snaps["t"] >= T0
    val_mask = cfg.in_window(snaps["t"], ctx.win["validate"])
    known_by = T0 - 1
    lgd = ecl.pooled_lgd(ctx.losses, known_by)
    paths = scenario_paths(c)
    weights = {n: s["weight"] for n, s in c["models"]["ecl"]["scenarios"].items()}

    rows_metrics, rows_month, rows_recal, rows_ecl, preds = [], [], [], [], {}
    seeds = c["models"]["seeds"][:n_seeds]
    for key in model_keys:
        for seed in (seeds if c["models"]["models"][key]["type"] == "lightgbm_hazard" else seeds[:1]):
            t0 = time.time()
            model = fit_model(ctx, key, seed)
            log.info("fitted %s seed %d in %.0fs", key, seed, time.time() - t0)
            uses_macro = "macro" in c["models"]["models"][key]["features"]
            ecl_paths = {}
            ti = np.flatnonzero(test.to_numpy())
            pi = np.flatnonzero(~test.to_numpy())
            todo = {n: p for n, p in paths.items() if uses_macro or n == "baseline"}
            t1 = time.time()
            path_out = _all_paths(model, snaps, ti, pi, ctx, H, todo)
            log.info("%s seed %d: %d macro paths scored in %.0fs", key, seed, len(todo), time.time() - t1)
            for pname, path in todo.items():
                # Test rows need the full lifetime path (ECL); earlier rows only feed 12-month
                # validation metrics and recalibration, whose longer labels are hidden anyway.
                m, m_pre = path_out[pname]
                d = snaps[["loan_id", "t"] + [f"def_{h}" for h in ctx.horizons]
                          + [f"known_{h}" for h in ctx.horizons]].copy()
                for h in ctx.horizons:
                    col = np.full(len(d), np.nan)
                    col[ti] = np.cumsum(m[:, :h], axis=1)[:, -1]
                    if h <= 12:
                        col[pi] = np.cumsum(m_pre[:, :h], axis=1)[:, -1]
                    d[f"pd_{h}"] = col
                tag = f"{key}|{seed}|{pname}"
                preds[tag] = d
                for h in ctx.horizons:
                    for split, mask in (("test", test), ("val", val_mask)):
                        lab = f"def_{h}" if split == "test" else f"known_{h}"
                        if d.loc[mask, lab].notna().sum() == 0:
                            continue
                        rows_metrics.append(dict(model=key, seed=seed, path=pname, horizon=h, split=split,
                                                 **evaluate(d[mask], lab, f"pd_{h}")))
                mt = metrics.by_month(d[test & d["def_12"].notna()], "def_12", "pd_12")
                rows_month.append(mt.assign(model=key, seed=seed, path=pname))
                if pname in weights or pname == "oracle":
                    st = snaps.iloc[ti]
                    e = ecl.ecl(m, st, ctx.loans, lgd["lgd"], [12, H])
                    ecl_paths[pname] = e
                    rv = ecl.realised_vs_predicted(st, ctx.loans, e["ecl_12"].to_numpy(), lgd["lgd"])
                    rows_ecl.append(rv.assign(model=key, seed=seed, path=pname,
                                              lifetime_ecl=e[f"ecl_{H}"].sum()))
            if not uses_macro:
                for pname in weights:
                    if pname != "baseline":
                        rows_ecl.append(rows_ecl[-1].assign(path=pname))
            # Recalibration of the deployable (baseline-path) 12-month PD.
            d = preds[f"{key}|{seed}|baseline"].copy()
            d["y"], d["p"] = d["def_12"], d["pd_12"]   # recalibrate() only uses labels with s + delay <= t
            feats = assemble(snaps, ctx.loans, ctx.macro)
            for col in recalibrate.IW_FEATURES:
                d[col] = feats[col].to_numpy()
            test_months = sorted(d.loc[test, "t"].unique())
            for method in rc["methods"]:
                for mode in (rc["modes"] if method != "none" else ["static"]):
                    pr = recalibrate.recalibrate(d, test_months, method, mode, delay, rc["window_months"])
                    dd = d[test].assign(p_recal=pr[test])
                    rows_recal.append(dict(model=key, seed=seed, method=method, mode=mode,
                                           **evaluate(dd, "def_12", "p_recal")))
                    preds[f"{key}|{seed}|recal|{method}|{mode}"] = dd[["loan_id", "t", "def_12", "p_recal"]]
            log.info("%s seed %d done in %.0fs", key, seed, time.time() - t0)

    pd.DataFrame(rows_metrics).to_csv(out / "metrics.csv", index=False)
    pd.concat(rows_month).to_csv(out / "monthly.csv", index=False)
    pd.DataFrame(rows_recal).to_csv(out / "recalibration.csv", index=False)
    if rows_ecl:
        pd.concat(rows_ecl).to_csv(out / "ecl_monthly.csv", index=False)
        ecl_summary(pd.concat(rows_ecl), weights).to_csv(out / "ecl_summary.csv", index=False)

    # DeLong and bootstrap CIs on the 12-month test PD, first seed, baseline path.
    s0 = seeds[0]
    base = {k: preds[f"{k}|{s0}|baseline"] for k in model_keys}
    rows_dl = []
    for i, a in enumerate(model_keys):
        for b in model_keys[i + 1:]:
            da, db = base[a][test], base[b][test]
            ok = da["def_12"].notna().to_numpy()
            rows_dl.append(dict(a=a, b=b, **metrics.delong(da["def_12"][ok], da["pd_12"][ok], db["pd_12"][ok])))
    pd.DataFrame(rows_dl).to_csv(out / "delong.csv", index=False)
    if n_boot:
        rows_bs = []
        for k, d in base.items():
            dt = d[test & d["def_12"].notna()]
            for stat_name, fn in (("auc", lambda x: roc_auc_score(x["def_12"], x["pd_12"])),
                                  ("ece", lambda x: metrics.ece(x["def_12"], x["pd_12"])),
                                  ("citl", lambda x: x["def_12"].mean() - x["pd_12"].mean())):
                rows_bs.append(dict(model=k, stat=stat_name,
                                    **metrics.cluster_bootstrap(dt, fn, n_boot=n_boot, seed=s0)))
        pd.DataFrame(rows_bs).to_csv(out / "bootstrap.csv", index=False)

    # Drift of loan features from the validation window to each later month.
    feats = assemble(snaps, ctx.loans, ctx.macro)
    ref = feats[feats["t"].between(*ctx.win["validate"])]
    cols = ["credit_score", "orig_ltv", "orig_dti", "mtm_ltv", "loan_age", "refi_incentive", "dlq_now",
            "current_interest_rate"]
    later = feats[feats["t"] > ctx.win["validate"][1]]
    later = later[later["t"] % 3 == 2]          # quarter-end months keep the C2ST affordable
    drift.drift_table(ref, later, cols, c2st_cols=cols).to_csv(out / "drift.csv", index=False)
    mac = ctx.macro.loc[snaps["t"].min(): snaps["t"].max()].copy()
    mac.index = [cfg.idx_to_ym(i) for i in mac.index]
    mac.to_csv(out / "macro_path.csv", index_label="yyyymm")

    info = dict(experiment=name, models=model_keys, seeds=seeds, lifetime_months=H, lgd=lgd,
                windows={k: [cfg.idx_to_ym(x) if x is not None else None for x in v] if v else None
                         for k, v in ctx.win.items()},
                runtime_s=round(time.time() - t_start), **ctx.info)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    pd.concat([d.assign(tag=k) for k, d in preds.items() if "|recal|" not in k]).to_parquet(out / "predictions.parquet")
    log.info("results in %s (%.0fs)", out, time.time() - t_start)
    return out


_JOB: dict = {}


def _one_path(pname):
    j = _JOB
    rows_t, rows_p = j["snaps"].iloc[j["ti"]], j["snaps"].iloc[j["pi"]]
    m, _ = lifetime.marginal_pd(j["model"], rows_t, j["loans"], j["macro"], j["H"], j["paths"][pname])
    m_pre, _ = lifetime.marginal_pd(j["model"], rows_p, j["loans"], j["macro"], 12, j["paths"][pname])
    return pname, m.astype("float32"), m_pre.astype("float32")


def _all_paths(model, snaps, ti, pi, ctx, H, paths) -> dict:
    """Score every macro path; logit models use one forked worker per path (inputs shared, not pickled)."""
    import multiprocessing as mp
    _JOB.update(model=model, snaps=snaps, ti=ti, pi=pi, loans=ctx.loans, macro=ctx.macro, H=H, paths=paths)
    # LightGBM is multithreaded already, and OpenMP deadlocks in forked children, so trees run in-process.
    if len(paths) == 1 or model.default.kind == "lightgbm_hazard":
        res = [_one_path(n) for n in paths]
    else:
        with mp.get_context("fork").Pool(min(len(paths), 4)) as pool:
            res = pool.map(_one_path, list(paths))
    _JOB.clear()
    return {n: (m.astype(float), mp_.astype(float)) for n, m, mp_ in res}


def ecl_summary(e: pd.DataFrame, weights: dict) -> pd.DataFrame:
    g = e.groupby(["model", "seed", "path"]).agg(pred=("pred", "sum"), real=("real", "sum"),
                                                  ead=("ead", "sum"), lifetime_ecl=("lifetime_ecl", "first"))
    g = g.reset_index()
    rows = [g]
    for (mk, sd), x in g[g["path"].isin(weights)].groupby(["model", "seed"]):
        w = x["path"].map(weights).to_numpy()
        rows.append(pd.DataFrame([dict(model=mk, seed=sd, path="weighted", pred=(x["pred"] * w).sum(),
                                       real=x["real"].iloc[0], ead=x["ead"].iloc[0],
                                       lifetime_ecl=(x["lifetime_ecl"] * w).sum())]))
    out = pd.concat(rows, ignore_index=True)
    out["ratio_pred_real"] = out["pred"] / out["real"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", nargs="+", default=["B0", "B1", "M1"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--lifetime-months", type=int, default=None)
    ap.add_argument("--processed", default=str(dataset.PROCESSED))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.exp, args.models, args.seeds, args.boot, args.lifetime_months,
        Path(args.out) if args.out else None, processed=Path(args.processed))


if __name__ == "__main__":
    main()
