import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pdshift import drift, metrics
from pdshift.lifetime import marginal_pd


def test_ece_and_calibration_of_a_calibrated_score():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.001, 0.2, 200000)
    y = (rng.random(len(p)) < p).astype(float)
    s = metrics.summary(y, p)
    assert s["ece"] < 0.003
    assert s["cal_slope"] == pytest.approx(1, abs=0.05)
    assert s["cal_intercept"] == pytest.approx(0, abs=0.03)
    assert s["auc"] == pytest.approx(roc_auc_score(y, p))


def test_delong_matches_sklearn_and_detects_difference():
    rng = np.random.default_rng(1)
    y = rng.random(5000) < 0.1
    good = y + rng.normal(0, 0.8, len(y))
    bad = y + rng.normal(0, 2.0, len(y))
    r = metrics.delong(y, good, bad)
    assert r["auc1"] == pytest.approx(roc_auc_score(y, good))
    assert r["auc2"] == pytest.approx(roc_auc_score(y, bad))
    assert r["p_value"] < 1e-6
    assert metrics.delong(y, good, good)["p_value"] == pytest.approx(1.0)


def test_cluster_bootstrap_covers_truth():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"loan_id": np.repeat(np.arange(2000), 5), "x": np.repeat(rng.normal(0, 1, 2000), 5)})
    r = metrics.cluster_bootstrap(df, lambda d: d["x"].mean(), n_boot=300)
    assert r["lo"] < r["estimate"] < r["hi"]
    # clustered SE ~ 1/sqrt(2000), not 1/sqrt(10000)
    assert r["se"] == pytest.approx(1 / np.sqrt(2000), rel=0.2)


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 50000)
    assert drift.psi(a, rng.normal(0, 1, 50000)) < 0.01
    assert drift.psi(a, rng.normal(1, 1, 50000)) > 0.25


class ConstantHazards:
    def __init__(self, hd, hp):
        self.hd, self.hp = hd, hp

    def hazards(self, X):
        return np.full(len(X), self.hd), np.full(len(X), self.hp)


def test_cumulative_incidence_matches_closed_form():
    hd, hp, H = 0.002, 0.01, 60
    rows = pd.DataFrame({"loan_id": ["x"], "t": [24000], "loan_age": [10], "current_upb": [1.0],
                         "current_interest_rate": [5.0], "dlq_now": [0]})
    loans = pd.DataFrame({"orig_upb": [1.0], "orig_ltv": [80.0], "first_payment_date": [199901]},
                         index=pd.Index(["x"], name="loan_id"))
    mac = pd.DataFrame({c: [1.0] for c in ["unrate", "unrate_d12", "mortgage30us", "csushpinsa_yoy",
                                           "csushpinsa_idx", "gdpc1_yoy"]}, index=[24000])
    mac = mac.reindex(range(23000, 24100)).ffill().bfill()
    m, S = marginal_pd(ConstantHazards(hd, hp), rows, loans, mac, H)
    expected = hd / (hd + hp) * (1 - (1 - hd - hp) ** H)
    assert m.sum() == pytest.approx(expected, rel=1e-9)
    assert S[0, -1] == pytest.approx((1 - hd - hp) ** (H - 1))
