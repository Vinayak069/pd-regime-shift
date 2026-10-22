"""Paired, loan-clustered bootstrap of test AUC differences between models (replaces DeLong).

  python -m pdshift.auc_diff [--boot 1000]

Reads results/<exp>/predictions.parquet (first seed, baseline path) and writes
results/auc_diff.csv: AUC(a) - AUC(b) with a percentile 95% interval.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as cfg
from . import metrics

PAIRS = [("M1", "B0"), ("M1", "B1"), ("B0", "B1")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args(argv)
    rows = []
    for e in ["control", "crisis_2008", "covid"]:
        d = cfg.ROOT / "results" / e
        info = json.loads((d / "run_info.json").read_text())
        seed, T0 = info["seeds"][0], cfg.ym_to_idx(int(str(info["windows"]["test"][0])))
        p = pd.read_parquet(d / "predictions.parquet", columns=["loan_id", "t", "def_12", "pd_12", "tag"])
        p = p[p["t"] >= T0]
        wide = None
        for m in {x for pair in PAIRS for x in pair}:
            x = p[p["tag"] == f"{m}|{seed}|baseline"][["loan_id", "t", "def_12", "pd_12"]].rename(columns={"pd_12": m})
            wide = x if wide is None else wide.merge(x.drop(columns="def_12"), on=["loan_id", "t"])
        wide = wide[wide["def_12"].notna()].reset_index(drop=True)
        for a, b in PAIRS:
            fn = lambda x, a=a, b=b: roc_auc_score(x["def_12"], x[a]) - roc_auc_score(x["def_12"], x[b])
            rows.append(dict(experiment=e, a=a, b=b, **metrics.cluster_bootstrap(wide, fn, n_boot=args.boot, seed=seed)))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(cfg.ROOT / "results" / "auc_diff.csv", index=False)


if __name__ == "__main__":
    main()
