"""Guards on the evaluation protocol: no look-ahead, label delay, split hygiene."""
import numpy as np
import pandas as pd
import pytest

from pdshift import config as cfg
from pdshift import macro, recalibrate

import synth


def test_splits_do_not_overlap_and_gap_covers_label_delay():
    c = cfg.load("splits")
    delay = c["label_delay_months"]
    for name, e in c["experiments"].items():
        w = {k: cfg.window(v) for k, v in e.items()}
        assert w["train"][1] < w["validate"][0]
        assert w["validate"][1] < w["gap"][0] <= w["gap"][1] < w["test"][0]
        # 12-month labels of the last validation month close before the test window
        assert w["validate"][1] + delay <= w["test"][0] - 1, name


def test_macro_values_respect_publication_lag(tmp_path):
    synth.fred(tmp_path)
    spec = cfg.load("features")["macro"]
    tab = macro.load_macro(tmp_path, spec)
    raw = macro.read_fred_csv(tmp_path / "UNRATE.csv")
    raw_m = raw.groupby(raw.index.year * 12 + raw.index.month - 1).mean()
    t = cfg.ym_to_idx(201004)
    assert tab.loc[t, "unrate"] == pytest.approx(raw_m[t - spec["UNRATE"]["lag"]])
    # quarterly GDP: Q1 (dated January) becomes visible only from April
    g = macro.read_fred_csv(tmp_path / "GDPC1.csv")
    g_m = g.groupby(g.index.year * 12 + g.index.month - 1).mean()
    q1, q4 = cfg.ym_to_idx(201001), cfg.ym_to_idx(200910)
    yoy = lambda a, b: np.log(g_m[a] / g_m[b])
    assert tab.loc[cfg.ym_to_idx(201004), "gdpc1_yoy"] == pytest.approx(yoy(q1, q1 - 12))
    assert tab.loc[cfg.ym_to_idx(201003), "gdpc1_yoy"] == pytest.approx(yoy(q4, q4 - 12))


def test_macro_table_ignores_future_observations(tmp_path):
    synth.fred(tmp_path)
    spec = cfg.load("features")["macro"]
    series = {n: macro.read_fred_csv(tmp_path / f"{n}.csv") for n in spec}
    full = macro.macro_table(series, spec)
    cut = pd.Timestamp("2008-06-30")
    trunc = macro.macro_table({n: s[s.index <= cut] for n, s in series.items()}, spec)
    t = cfg.ym_to_idx(200806)
    common = trunc.loc[:t].dropna()
    pd.testing.assert_frame_equal(full.loc[common.index], common, check_freq=False)


def test_recalibration_never_uses_labels_before_delay():
    rng = np.random.default_rng(0)
    months = np.arange(100, 160)
    df = pd.DataFrame({"t": np.repeat(months, 200)})
    df["p"] = rng.uniform(0.01, 0.05, len(df))
    df["y"] = (rng.random(len(df)) < df["p"] * 2).astype(float)
    seen = []
    orig = recalibrate.fit_apply

    def spy(method, hist, cur, p="p", y="y"):
        seen.append((hist["t"].max() if len(hist) else -1, cur["t"].iloc[0]))
        return orig(method, hist, cur, p, y)

    recalibrate.fit_apply = spy
    try:
        for mode in ("static", "rolling"):
            recalibrate.recalibrate(df, months[30:], "platt", mode, delay=12, window=12)
    finally:
        recalibrate.fit_apply = orig
    assert seen and all(h + 12 <= t for h, t in seen)


def test_intercept_recalibration_fixes_a_level_shift():
    rng = np.random.default_rng(1)
    n = 40000
    p = rng.uniform(0.005, 0.05, n)
    y = (rng.random(n) < np.clip(2 * p, 0, 1)).astype(float)
    hist = pd.DataFrame({"p": p[: n // 2], "y": y[: n // 2]})
    cur = pd.DataFrame({"p": p[n // 2:], "y": y[n // 2:]})
    out = recalibrate.fit_apply("intercept", hist, cur)
    assert abs(out.mean() - cur["y"].mean()) < 0.1 * cur["y"].mean()


def test_early_recalibration_uses_only_closed_short_labels_and_tracks_a_jump():
    rng = np.random.default_rng(2)
    months = np.arange(100, 160)
    df = pd.DataFrame({"t": np.repeat(months, 2000)})
    df["p3"] = rng.uniform(0.002, 0.01, len(df))
    df["p"] = 4 * df["p3"]
    mult = np.where(df["t"] >= 130, 3.0, 1.0)          # risk triples from month 130 on
    df["y3"] = (rng.random(len(df)) < df["p3"] * mult).astype(float)
    df["y"] = (rng.random(len(df)) < df["p"] * mult).astype(float)
    df.loc[df["t"] >= 150, ["y", "y3"]] = np.nan       # labels a lender could not have yet
    test = months[30:]
    out, log = recalibrate.recalibrate_early(df, test, "early", delay=12, window=12)
    # 12-month labels before month 130 were all the static fit could see; the early fit adapts
    late = df["t"].between(140, 159)
    assert out[late].mean() > 2.5 * df.loc[late, "p"].mean()
    assert log["fired"].all()
    st, _ = recalibrate.recalibrate_early(df, test, "trig_psi", delay=12, window=12)
    assert st[late].mean() < 1.5 * df.loc[late, "p"].mean()     # no score drift: stays static


def test_hazard_shift_matches_early_labels_and_is_identity_at_zero():
    rng = np.random.default_rng(3)
    n = 20000
    hd = np.tile(np.linspace(0.0005, 0.003, 12), (n, 1)) * rng.uniform(0.5, 1.5, (n, 1))
    hp = np.full((n, 12), 0.01)
    s, ref = np.ones(n), np.zeros(n)
    for k in range(12):
        ref += s * hd[:, k]
        s *= 1 - hd[:, k] - hp[:, k]
    assert np.allclose(recalibrate.shifted_pd(hd, hp, 0.0, 12), ref, rtol=1e-6)
    y3 = (rng.random(n) < 2 * recalibrate.shifted_pd(hd, hp, 0.0, 3)).astype(float)
    a = recalibrate._hazard_shift(hd, hp, y3)
    assert abs(recalibrate.shifted_pd(hd, hp, a, 3).mean() - y3.mean()) < 1e-9
    assert 0.4 < a < 1.0                                   # about log 2
