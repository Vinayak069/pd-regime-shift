"""Collect results/<experiment>/ outputs into results/summary.md and paper figures.

  python -m pdshift.report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import config as cfg
from . import recalibrate

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = cfg.ROOT / "results"
EXPS = ["control", "crisis_2008", "covid"]
# reference palette (dataviz skill), categorical slots 1-3; observed series in primary ink
COLORS = {"observed": "#0b0b0b", "none": "#2a78d6", "platt static": "#eb6834", "platt rolling": "#1baf7a"}


def _fmt(x, d=3):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def headline(d: Path) -> pd.DataFrame:
    m = pd.read_csv(d / "metrics.csv")
    m = m[(m.split == "test") & (m.horizon == 12) & (m.path == "baseline")]
    bs = pd.read_csv(d / "bootstrap.csv") if (d / "bootstrap.csv").exists() else None
    v = pd.read_csv(d / "metrics.csv")
    v = v[(v.split == "val") & (v.horizon == 12) & (v.path == "baseline")].set_index("model")
    rows = []
    for _, r in m.iterrows():
        ci = ""
        if bs is not None:
            b = bs[(bs.model == r.model) & (bs.stat == "auc")]
            if len(b):
                ci = f" [{b.lo.iloc[0]:.3f}, {b.hi.iloc[0]:.3f}]"
        rows.append({"model": r.model, "val obs/pred": f"{v.loc[r.model, 'obs_rate']:.4f} / {v.loc[r.model, 'pred_rate']:.4f}",
                     "val AUC": _fmt(v.loc[r.model, "auc"]),
                     "test events": int(r.events), "test obs rate": _fmt(r.obs_rate, 4),
                     "test pred rate": _fmt(r.pred_rate, 4), "pred/obs": _fmt(r.ratio, 2),
                     "test AUC [95% CI]": _fmt(r.auc) + ci, "ECE": _fmt(r.ece, 4),
                     "cal slope": _fmt(r.cal_slope, 2), "std monthly CITL": _fmt(r.citl_monthly_std, 4)})
    return pd.DataFrame(rows)


def recal_table(d: Path, model: str) -> pd.DataFrame:
    r = pd.read_csv(d / "recalibration.csv")
    r = r[r.model == model]
    return pd.DataFrame({"method": r.method, "mode": r["mode"], "pred/obs": r.ratio.map(lambda x: _fmt(x, 2)),
                         "AUC": r.auc.map(_fmt), "ECE": r.ece.map(lambda x: _fmt(x, 4)),
                         "cal slope": r.cal_slope.map(lambda x: _fmt(x, 2)),
                         "std monthly CITL": r.citl_monthly_std.map(lambda x: _fmt(x, 4)),
                         "max abs monthly CITL": r.citl_monthly_maxabs.map(lambda x: _fmt(x, 4))})


def path_table(d: Path) -> pd.DataFrame:
    m = pd.read_csv(d / "metrics.csv")
    m = m[(m.split == "test") & (m.horizon == 12)]
    p = m.pivot_table(index="model", columns="path", values="ratio")
    return p.reset_index().round(3)


def monthly_figure(d: Path, name: str, model: str = "M1") -> Path | None:
    pr = d / "predictions.parquet"
    if not pr.exists():
        return None
    p = pd.read_parquet(pr)
    p = p[p["tag"].str.startswith(f"{model}|") & p["tag"].str.endswith("|baseline")].reset_index(drop=True)
    info = json.loads((d / "run_info.json").read_text())
    T0 = cfg.ym_to_idx(int(str(info["windows"]["test"][0])))
    p["y"], p["p"] = p["def_12"], p["pd_12"]
    months = sorted(p.loc[p["t"] >= T0, "t"].unique())
    series = {"none": p["p"]}
    for mode in ("static", "rolling"):
        series[f"platt {mode}"] = recalibrate.recalibrate(p, months, "platt", mode, 12, 12)
    test = p[p["t"] >= T0]
    g = test.groupby("t")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    x = [pd.Timestamp(year=t // 12, month=t % 12 + 1, day=1) for t in g.size().index]
    ax.plot(x, g["y"].mean() * 100, color=COLORS["observed"], lw=2, label="observed")
    for k, s in series.items():
        ax.plot(x, s[test.index].groupby(test["t"]).mean() * 100, color=COLORS[k], lw=2,
                label={"none": f"{model}, no recalibration", "platt static": "Platt, static",
                       "platt rolling": "Platt, rolling (12-month delay)"}[k])
    ax.set_ylabel("12-month default rate (%)")
    ax.set_title(f"{name}: observed vs predicted 12-month PD by snapshot month", fontsize=10, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = RESULTS / f"fig_{name}_monthly.png"
    fig.savefig(out)
    plt.close(fig)
    return out


ROBUST_COLORS = {"none|static": "#2a78d6", "intercept|rolling": "#eb6834", "intercept|early": "#1baf7a",
                 "intercept|trig_z": "#8e4ec6"}
ROBUST_NAMES = {"none|static": "M1, no recalibration", "intercept|rolling": "intercept, rolling (12-month delay)",
                "intercept|early": "intercept from 3-month labels", "intercept|trig_z": "drift-triggered (z)"}


def has_relief(d: Path) -> bool:
    """False when the forbearance-free definition changes no label in this test window."""
    r = pd.read_csv(d / "robustness.csv").set_index(["seed", "model", "method", "mode"])
    return not np.allclose(r[r["label"] == "all"]["ratio"], r[r["label"] == "xf"]["ratio"].reindex(
        r[r["label"] == "all"].index))


def robust_table(d: Path) -> pd.DataFrame:
    r = pd.read_csv(d / "robustness.csv")
    if not has_relief(d):
        r = r[r["label"] == "all"]
    g = r.groupby(["model", "label", "method", "mode"], sort=False)
    out = []
    for (mk, lab, method, mode), x in g:
        out.append({"model": mk, "default def.": lab, "recalibration": f"{method} {mode}",
                    "seeds": x["seed"].nunique(),
                    "pred/obs": f"{x['ratio'].mean():.2f} ± {x['ratio'].std(ddof=1):.2f}",
                    "AUC": f"{x['auc'].mean():.3f} ± {x['auc'].std(ddof=1):.3f}",
                    "ECE": f"{x['ece'].mean():.4f}",
                    "std monthly CITL": f"{x['citl_monthly_std'].mean():.4f}"})
    return pd.DataFrame(out)


def robust_figure(d: Path, name: str, label: str) -> Path | None:
    f = d / "robustness_monthly.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    m = m[m["label"] == label]
    if m.empty:
        return None
    x = [pd.Timestamp(year=int(t) // 12, month=int(t) % 12 + 1, day=1) for t in m["t"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    ax.plot(x, m["obs"] * 100, color=COLORS["observed"], lw=2, label="observed")
    for k, col in ROBUST_COLORS.items():
        ax.plot(x, m[k] * 100, color=col, lw=2, label=ROBUST_NAMES[k])
    ax.set_ylabel("12-month default rate (%)")
    what = "main default definition" if label == "all" else "forbearance delinquency not counted"
    ax.set_title(f"{name}, {what}: observed vs recalibrated 12-month PD", fontsize=10, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e4e0", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = RESULTS / f"fig_{name}_robust_{label}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


METHOD_COLORS = {"none": "#2a78d6", "intercept": "#eb6834", "platt": "#1baf7a", "isotonic": "#8e4ec6",
                 "iw_platt": "#c9a227", "early": "#d6457a", "trig_z": "#3a9ba8"}


def pareto_figure(sel_csv: Path, title: str, out: Path) -> Path | None:
    """Left: validation objectives of every candidate, Pareto set outlined. Right: the same Pareto
    candidates scored on the test window afterwards (static mode), pred/obs against AUC."""
    if not sel_csv.exists():
        return None
    s = pd.read_csv(sel_csv)
    ev = s.drop_duplicates("eval")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=200)
    for m, col in METHOD_COLORS.items():
        x = ev[ev["method"] == m]
        if x.empty:
            continue
        a.scatter(x["val_auc"], x["val_ece"], s=14, color=col, alpha=0.35, linewidths=0)
        par = x[x["pareto"]]
        a.scatter(par["val_auc"], par["val_ece"], s=26, color=col, edgecolors="#0b0b0b", linewidths=0.6,
                  label=m.replace("_", " "))
        t = s[(s["method"] == m) & s["pareto"] & (s["mode"] == "static")]
        b.scatter(t["test_auc"], t["test_ratio"], s=26, color=col, edgecolors="#0b0b0b", linewidths=0.6)
    for pick, mk in (("knee", "*"), ("max_auc", "^")):
        r = ev[ev["pick"].fillna("").str.contains(pick)]
        a.scatter(r["val_auc"], r["val_ece"], s=110, marker=mk, facecolors="none", edgecolors="#0b0b0b", lw=1.2)
        t = s[s["pick"].fillna("").str.contains(pick) & (s["mode"] == "static")]
        b.scatter(t["test_auc"], t["test_ratio"], s=110, marker=mk, facecolors="none", edgecolors="#0b0b0b",
                  lw=1.2, label={"knee": "Pareto knee", "max_auc": "max-AUC pick"}[pick])
    b.axhline(1.0, color="#9a9893", lw=1, ls="--")
    a.set_xlabel("validation AUC")
    a.set_ylabel("validation ECE")
    b.set_xlabel("test AUC")
    b.set_ylabel("test predicted / observed")
    a.set_title(f"{title}: candidates on validation (Pareto set outlined)", fontsize=9, loc="left")
    b.set_title("Pareto set on the test window", fontsize=9, loc="left")
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color="#e5e4e0", lw=0.8)
    a.legend(frameon=False, fontsize=7, title="recalibration", title_fontsize=7)
    b.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    parts = ["# Results summary\n", "All numbers below come from the Freddie Mac SFLLD standard sample files "
             "(1999-2026 vintages) and FRED macro series, produced by `make experiments` / `make select`. "
             "12-month PD at monthly snapshots; test windows are out of time.\n"]
    for e in EXPS:
        d = RESULTS / e
        if not (d / "run_info.json").exists():
            parts.append(f"\n## {e}\n\nNot run yet.\n")
            continue
        info = json.loads((d / "run_info.json").read_text())
        parts.append(f"\n## {e}\n")
        parts.append(f"Windows: {info['windows']}. Training: {info.get('train_pairs', info.get('train_rows'))} "
                     f"landmark pairs from {info['train_loans']} loans (vintages {info['train_vintages'][0]}-"
                     f"{info['train_vintages'][-1]}), {info['train_defaults']} defaults. Test snapshots: "
                     f"{info['test_snapshot_rows']} loan-months. Pooled LGD {info['lgd']['lgd']:.3f} "
                     f"from {info['lgd']['n']} resolved defaults.\n")
        parts.append("\n**Models, deployable (persistence) macro path**\n\n" + md_table(headline(d)))
        parts.append("\n\n**Predicted / observed 12-month default rate by macro path**\n\n" + md_table(path_table(d)))
        for mdl in ("M1", "B1"):
            parts.append(f"\n\n**Recalibration of {mdl}**\n\n" + md_table(recal_table(d, mdl)))
        es = pd.read_csv(d / "ecl_summary.csv")
        es = es[es.path.isin(["baseline", "weighted", "oracle"])]
        parts.append("\n\n**12-month ECL, predicted / realised (same LGD and EAD)**\n\n" + md_table(
            es[["model", "path"]].assign(ratio=es.ratio_pred_real.round(3))))
        if (d / "selection.csv").exists():
            s = pd.read_csv(d / "selection.csv")
            picks = s[s["pick"].notna() & (s["pick"] != "")]
            cols = ["pick", "mode", "method", "num_leaves", "learning_rate", "min_child_samples", "feature_fraction",
                    "val_auc", "val_ece", "val_citl_monthly_std", "test_auc", "test_ece", "test_ratio",
                    "test_citl_monthly_std"]
            parts.append("\n\n**NSGA-II selection: single-objective picks vs Pareto knee (test scored after "
                         "selection)**\n\n" + md_table(picks[cols].round(4)))
            parts.append(f"\n\nPareto set: {int(s.drop_duplicates('eval')['pareto'].sum())} of "
                         f"{s['eval'].nunique()} evaluated candidates.\n")
        f = monthly_figure(d, e)
        if f:
            parts.append(f"\n\n![{e} monthly]({f.name})\n")
        if (d / "robustness.csv").exists():
            parts.append("\n\n**Robustness: mean ± sd over seeds (sample and model redrawn), test window**\n\n"
                         + md_table(robust_table(d)))
            if (d / "robustness_bootstrap.csv").exists():
                b = pd.read_csv(d / "robustness_bootstrap.csv").round(4)
                parts.append("\n\n**Loan-clustered bootstrap, first seed, M1**\n\n" + md_table(b))
            for lab in (("all", "xf") if has_relief(d) else ("all",)):
                f = robust_figure(d, e, lab)
                if f:
                    parts.append(f"\n\n![{e} {lab}]({f.name})\n")
    parts.append("\n## Second NSGA-II search (early-label methods in the search space)\n")
    for sub, title in (("crisis_2008", "Crisis 2008"), ("covid_xf", "COVID, forbearance-free")):
        f = RESULTS / "sel2" / sub / "selection.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        picks = s[s["pick"].notna() & (s["pick"] != "")]
        parts.append(f"\n**{title}**\n\n" + md_table(picks[["pick", "mode", "method", "val_auc", "val_ece",
                                                              "test_auc", "test_ratio", "test_citl_monthly_std"]].round(4)))
        fig = pareto_figure(f, title, RESULTS / f"fig_pareto_{sub}.png")
        parts.append(f"\n\n![{title} Pareto front]({fig.name})\n")
    for e in EXPS:
        f = RESULTS / e / "selection.csv"
        if f.exists():
            pareto_figure(f, f"{e} (first search)", RESULTS / f"fig_pareto_{e}_first.png")
    (RESULTS / "summary.md").write_text("\n".join(parts))
    print(RESULTS / "summary.md")


if __name__ == "__main__":
    main()
