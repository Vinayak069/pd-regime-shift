"""The whole pipeline on tiny synthetic files: checks that it runs and writes its outputs.
The numbers it produces mean nothing; real results come only from the Freddie Mac data."""
import json

import pandas as pd

from pdshift import experiment, panel, robustness, selection

import synth


def test_pipeline_runs_end_to_end(tmp_path):
    raw, proc, fred = tmp_path / "raw", tmp_path / "proc", tmp_path / "fred"
    synth.freddie(raw, years=range(2000, 2015), loans=150)
    synth.fred(fred)
    panel.main(["--raw", str(raw), "--out", str(proc)])
    out = experiment.run("control", ["B0", "B1", "M1"], n_seeds=1, n_boot=5, lifetime_months=24,
                         out_dir=tmp_path / "res", processed=proc, fred=fred)
    m = pd.read_csv(out / "metrics.csv")
    assert {"B0", "B1", "M1"} <= set(m["model"])
    assert {"test", "val"} <= set(m["split"])
    r = pd.read_csv(out / "recalibration.csv")
    assert set(r["method"]) == {"none", "intercept", "platt", "isotonic", "iw_platt"}
    for f in ["monthly.csv", "delong.csv", "bootstrap.csv", "drift.csv", "ecl_summary.csv", "predictions.parquet"]:
        assert (out / f).exists(), f
    info = json.loads((out / "run_info.json").read_text())
    assert info["train_defaults"] > 0
    s = selection.run("control", out_dir=tmp_path / "res", pop=4, gens=2, processed=proc, fred=fred)
    sel = pd.read_csv(s / "selection.csv")
    assert sel["pareto"].any() and sel["test_auc"].notna().any()
    rb = robustness.run("control", n_seeds=2, n_boot=5, out_dir=tmp_path / "res", processed=proc, fred=fred)
    r = pd.read_csv(rb / "robustness.csv")
    assert set(r["seed"]) == {11, 22} and set(r["label"]) == {"all", "xf"}
    assert {"early", "trig_z", "trig_psi", "rolling"} <= set(r["mode"])
