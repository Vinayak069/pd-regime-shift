"""Multi-objective selection of the LightGBM default hazard and its recalibration (NSGA-II).

Decision variables: num_leaves, learning_rate, min_child_samples, feature_fraction
(ranges from configs/models.yaml) and the recalibration method, including the
early-label methods of recalibrate.recalibrate_early. --label xf scores against the
forbearance-free default definition. Objectives, all
measured on validation snapshots with labels that close before the test window:

  maximise AUC of the 12-month PD
  minimise ECE of the 12-month PD
  minimise the std over months of calibration-in-the-large

The prepayment hazard is fitted once and held fixed. Static recalibration for a
validation month uses snapshots 12-23 months earlier (out-of-sample loans).
Afterwards every evaluated candidate is scored on the test window, so the paper
can compare the Pareto set and its knee against single-objective picks.

  python -m pdshift.selection --exp crisis_2008
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from . import config as cfg
from . import dataset, lifetime, recalibrate
from .experiment import evaluate, fit_model, prepare
from .features import assemble

log = logging.getLogger("pdshift")
METHODS = ["none", "intercept", "platt", "isotonic", "iw_platt", "early", "trig_z"]
LABELS = {"all": ("def_12", "def_3"), "xf": ("def_12_xf", "def_3_xf")}


def decode(x, space: dict) -> tuple[dict, str]:
    nl, lr, mcs, ff = space["num_leaves"], space["learning_rate"], space["min_child_samples"], space["feature_fraction"]
    params = dict(num_leaves=int(round(np.exp(np.interp(x[0], [0, 1], np.log(nl))))),
                  learning_rate=float(np.exp(np.interp(x[1], [0, 1], np.log(lr)))),
                  min_child_samples=int(round(np.exp(np.interp(x[2], [0, 1], np.log(mcs))))),
                  feature_fraction=float(np.interp(x[3], [0, 1], ff)))
    return params, METHODS[min(int(x[4] * len(METHODS)), len(METHODS) - 1)]


class Selector:
    def __init__(self, ctx, seed: int, max_rows: int, label: str = "all"):
        self.ctx, self.seed, self.max_rows, self.label = ctx, seed, max_rows, label
        self.delay = ctx.c["splits"]["label_delay_months"]
        self.rwin = ctx.c["models"]["recalibration"]["window_months"]
        self.space = ctx.c["models"]["models"]["M1"]["search"]
        base = fit_model(ctx, "M1", seed, max_rows=max_rows)
        self.prepay = base.prepay
        # a smaller snapshot sample per month keeps each candidate's evaluation affordable
        s = dataset.snapshots(ctx.snaps, ctx.c["models"]["selection"]["loans_per_snapshot"], seed)
        self.snaps = s.reset_index(drop=True)
        s = self.snaps
        feats = assemble(s, ctx.loans, ctx.macro)
        y12, y3 = LABELS[label]
        self.frame = s[["loan_id", "t"]].copy()
        for name, col, h in (("y12", y12, 12), ("y3", y3, 3)):
            self.frame[name] = s[col].to_numpy()
            # labels not yet closed at the start of the test window are hidden from selection
            hide = (s["t"] < ctx.T0) & (s["t"] + h > ctx.T0 - 1)
            self.frame[f"{name}_known"] = s[col].where(~hide).to_numpy()
        for col in recalibrate.IW_FEATURES:
            self.frame[col] = feats[col].to_numpy()
        v0, v1 = ctx.win["validate"]
        self.val_months = list(range(v0, v1 + 1))
        self.test_months = sorted(s.loc[s["t"] >= ctx.T0, "t"].unique())
        self.cache: dict[tuple, tuple[np.ndarray, int]] = {}
        self.log: list[dict] = []

    def predictions(self, params: dict) -> np.ndarray:
        key = tuple(sorted(params.items()))
        if key not in self.cache:
            m = fit_model(self.ctx, "M1", self.seed, params=params, prepay=self.prepay, max_rows=self.max_rows)
            pdf = lifetime.cumulative_pd(m, self.snaps, self.ctx.loans, self.ctx.macro, [3, 12])
            self.cache[key] = (pdf["pd_12"].to_numpy(), pdf["pd_3"].to_numpy(),
                               m.default.model.best_iteration_ or m.default.model.n_estimators)
        return self.cache[key]

    def score(self, params: dict, method: str, months, label: str, mode: str = "static") -> dict:
        p, p3, n_iter = self.predictions(params)
        sfx = "_known" if label == "val" else ""
        d = self.frame.assign(p=p, p3=p3, y=self.frame[f"y12{sfx}"], y3=self.frame[f"y3{sfx}"])
        if method in recalibrate.EARLY_MODES:
            pr, _ = recalibrate.recalibrate_early(d, months, method, self.delay, self.rwin)
        else:
            pr = recalibrate.recalibrate(d, months, method, mode, self.delay, self.rwin)
        dd = d[d["t"].isin(months)].assign(p_recal=pr)
        return dict(n_trees=n_iter, **evaluate(dd, "y", "p_recal"))


class Problem(ElementwiseProblem):
    def __init__(self, sel: Selector):
        super().__init__(n_var=5, n_obj=3, xl=np.zeros(5), xu=np.ones(5) - 1e-9)
        self.sel = sel

    def _evaluate(self, x, out, *args, **kwargs):
        params, method = decode(x, self.sel.space)
        r = self.sel.score(params, method, self.sel.val_months, "val")
        self.sel.log.append(dict(eval=len(self.sel.log), method=method, **params,
                                 **{f"val_{k}": v for k, v in r.items()}))
        out["F"] = [-r["auc"], r["ece"], r["citl_monthly_std"]]
        log.info("eval %d: %s %s val auc %.4f ece %.5f citl_std %.5f", len(self.sel.log), method, params,
                 r["auc"], r["ece"], r["citl_monthly_std"])


def pareto_mask(F: np.ndarray) -> np.ndarray:
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        keep[i] = not dominated.any()
    return keep


def knee(F: np.ndarray) -> int:
    """Point closest to the ideal after min-max scaling each objective."""
    lo, hi = F.min(0), F.max(0)
    Z = (F - lo) / np.where(hi > lo, hi - lo, 1)
    return int(np.argmin(np.linalg.norm(Z, axis=1)))


def run(name: str, out_dir: Path | None = None, pop: int | None = None, gens: int | None = None,
        seed: int | None = None, label: str = "all", **prep_kw) -> Path:
    t_start = time.time()
    c = cfg.load_all()
    sc = c["models"]["selection"]
    seed = seed if seed is not None else c["models"]["seeds"][0]
    out = Path(out_dir or cfg.ROOT / "results" / name)
    out.mkdir(parents=True, exist_ok=True)
    ctx = prepare(name, c, **prep_kw)
    sel = Selector(ctx, seed, sc["max_train_rows"], label)
    res = minimize(Problem(sel), NSGA2(pop_size=pop or sc["population"]),
                   ("n_gen", gens or sc["generations"]), seed=seed, verbose=False)
    ev = pd.DataFrame(sel.log)
    F = np.column_stack([-ev["val_auc"], ev["val_ece"], ev["val_citl_monthly_std"]])
    ev["pareto"] = pareto_mask(F)
    ev["pick"] = ""
    ev.loc[ev["val_auc"].idxmax(), "pick"] += "max_auc;"
    ev.loc[ev["val_ece"].idxmin(), "pick"] += "min_ece;"
    ev.loc[ev["val_brier"].idxmin(), "pick"] += "min_brier;"
    par = ev.index[ev["pareto"]]
    ev.loc[par[knee(F[ev["pareto"].to_numpy()])], "pick"] += "knee;"
    keys = ["num_leaves", "learning_rate", "min_child_samples", "feature_fraction"]
    test_rows = []
    for i, r in ev[ev["pareto"] | (ev["pick"] != "")].iterrows():
        params = {k: (int(r[k]) if k in ("num_leaves", "min_child_samples") else float(r[k])) for k in keys}
        for mode in ("static", "rolling"):
            s = sel.score(params, r["method"], sel.test_months, "test", mode)
            test_rows.append(dict(eval=r["eval"], mode=mode, **{f"test_{k}": v for k, v in s.items()}))
    ev = ev.merge(pd.DataFrame(test_rows), on="eval", how="left")
    ev.to_csv(out / "selection.csv", index=False)
    (out / "selection_info.json").write_text(json.dumps(dict(
        evaluations=len(sel.log), distinct_models=len(sel.cache), seed=seed, label=label,
        population=pop or sc["population"], generations=gens or sc["generations"],
        runtime_s=round(time.time() - t_start)), indent=2))
    log.info("selection for %s done: %d evaluations in %.0fs", name, len(sel.log), time.time() - t_start)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="NSGA-II model + recalibration selection")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--pop", type=int)
    ap.add_argument("--gens", type=int)
    ap.add_argument("--out")
    ap.add_argument("--label", choices=list(LABELS), default="all")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args.exp, Path(args.out) if args.out else None, args.pop, args.gens, label=args.label)


if __name__ == "__main__":
    main()
