"""Event labelling and Freddie Mac file discovery on hand-built examples."""
import numpy as np
import pandas as pd

from pdshift import config as cfg
from pdshift.data import find_vintages
from pdshift.panel import label_events

EVENTS = cfg.load("freddie_layout")["events"]


def perf(loan, start, dlq, zb_last=None):
    t0 = cfg.ym_to_idx(start)
    rows = []
    for k, d in enumerate(dlq):
        rows.append(dict(loan_id=loan, period=cfg.idx_to_ym(t0 + k), current_upb=1000.0, delinquency_status=str(d),
                         loan_age=k + 1, modification_flag=None, zero_balance_code=None, current_interest_rate=5.0,
                         actual_loss=np.nan, zero_balance_removal_upb=np.nan))
    if zb_last:
        rows[-1]["zero_balance_code"] = zb_last
        if zb_last == "09":
            rows[-1].update(actual_loss=300.0, zero_balance_removal_upb=1000.0)
    return rows


def test_default_prepay_and_censoring_labels():
    p = pd.DataFrame(perf("A", 201001, [0, 0, 1, 2, 3, 4, "RA", "RA"], "09")
                     + perf("B", 201001, [0, 0, 0, 0], "01")
                     + perf("C", 201001, [0, 0, 0], "96"))
    panel, losses = label_events(p, EVENTS, [1, 3, 12])
    a = panel[panel.loan_id == "A"].set_index("t")
    t0 = cfg.ym_to_idx(201001)
    assert list(a.index - t0) == [0, 1, 2, 3]            # at risk until 90+ dpd in month 4
    assert a["y_def"].tolist() == [0, 0, 0, 1]
    assert a.loc[t0 + 1, "def_3"] == 1 and a.loc[t0, "def_3"] == 0
    assert a.loc[t0, "def_12"] == 1
    b = panel[panel.loan_id == "B"].set_index("t")
    assert list(b.index - t0) == [0, 1, 2]
    assert b["y_pre"].tolist() == [0, 0, 1]
    assert (b["def_12"] == 0).all()                      # prepaid first: known non-default
    c = panel[panel.loan_id == "C"].set_index("t")
    assert c["def_12"].isna().all()                      # removed by a defect: censored
    assert losses.set_index("loan_id").loc["A", "actual_loss"] == 300.0
    assert losses.set_index("loan_id").loc["A", "default_t"] == t0 + 4


def test_forbearance_delinquency_excluded_from_xf_labels():
    rows = perf("F", 202001, [0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for r in rows[2:6]:
        r["borrower_assistance_status"] = "F"             # delinquent only while in forbearance
    rows += perf("G", 202001, [0, 0, 1, 2, 3, 4, 5, 6], "09")
    for r in rows[-8:][2:]:
        r["borrower_assistance_status"] = "F"             # forborne, then foreclosed: still a default
    panel, _ = label_events(pd.DataFrame(rows), EVENTS, [12])
    t0 = cfg.ym_to_idx(202001)
    f = panel[panel.loan_id == "F"].set_index("t")
    assert f.loc[t0, "def_12"] == 1 and f.loc[t0, "def_12_xf"] == 0
    assert f.loc[t0 + 2, "relief_now"] == 1 and f.loc[t0, "relief_now"] == 0
    g = panel[panel.loan_id == "G"].set_index("t")
    assert g.loc[t0, "def_12"] == 1 and g.loc[t0, "def_12_xf"] == 1


def test_find_vintages_recognises_sample_and_full_names(tmp_path):
    for n in ["sample_orig_2005.txt", "sample_perf_2005.txt", "sample_orig_2006.txt", "sample_svcg_2006.txt", "historical_data_2007Q1.txt",
              "historical_data_time_2007Q1.txt", "historical_data_2008Q2.txt"]:
        (tmp_path / n).write_text("")
    v = find_vintages(tmp_path)
    assert set(v) == {"2005", "2006", "2007Q1"}
    assert v["2007Q1"]["perf"].name == "historical_data_time_2007Q1.txt"


def test_landmark_pairs_link_state_at_t_to_outcomes_at_t_plus_k():
    from pdshift.dataset import landmark_pairs
    rows = pd.DataFrame({"loan_id": "A", "t": np.arange(100, 106), "loan_age": np.arange(1, 7),
                         "y_def": [0, 0, 0, 0, 0, 1], "y_pre": 0, "current_upb": 1.0,
                         "current_interest_rate": 5.0, "dlq_now": [0, 0, 1, 2, 2, 2], "dlq_max_12": 0,
                         "mod_ever": 0, "ever_30dpd": 0})
    p = landmark_pairs(rows, every=2, K=3, seed=0)
    assert (p["k"] < 3).all() and (p["t"] - p["t_landmark"] == p["k"]).all()
    # the state comes from the landmark month, the outcome from the later month
    for _, r in p.iterrows():
        assert r["dlq_now"] == rows.set_index("t").loc[r["t_landmark"], "dlq_now"]
        assert r["y_def"] == rows.set_index("t").loc[r["t"], "y_def"]
    assert p["t_landmark"].nunique() == len(set(p["t_landmark"]))
    assert set(np.diff(sorted(p["t_landmark"].unique()))) <= {2}
