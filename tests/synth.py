"""Tiny synthetic Freddie-Mac-shaped files for tests only. Never used for results."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pdshift import config as cfg


def fred(dirpath: Path, start=1995, end=2024, seed=0):
    rng = np.random.default_rng(seed)
    dirpath.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    t = np.arange(len(months))
    un = 5 + 2.5 * np.exp(-((months.year + months.month / 12 - 2010) ** 2) / 4) + rng.normal(0, .1, len(t))
    pd.DataFrame({"observation_date": months.strftime("%Y-%m-%d"), "UNRATE": un.round(1)}).to_csv(dirpath / "UNRATE.csv", index=False)
    hpi = 100 * np.exp(0.003 * t - 0.25 * (months.year >= 2007) * np.minimum((months.year - 2007) * 12 + months.month, 48) / 48)
    pd.DataFrame({"observation_date": months.strftime("%Y-%m-%d"), "CSUSHPINSA": hpi.round(3)}).to_csv(dirpath / "CSUSHPINSA.csv", index=False)
    weeks = pd.date_range(f"{start}-01-05", f"{end}-12-31", freq="W-THU")
    pd.DataFrame({"observation_date": weeks.strftime("%Y-%m-%d"),
                  "MORTGAGE30US": (7 - 0.15 * (weeks.year - start)).round(2)}).to_csv(dirpath / "MORTGAGE30US.csv", index=False)
    q = pd.date_range(f"{start}-01-01", f"{end}-10-01", freq="QS")
    pd.DataFrame({"observation_date": q.strftime("%Y-%m-%d"),
                  "GDPC1": (10000 * np.exp(0.006 * np.arange(len(q)))).round(1)}).to_csv(dirpath / "GDPC1.csv", index=False)


def freddie(dirpath: Path, years=range(2000, 2013), loans=300, end=201712, seed=0):
    rng = np.random.default_rng(seed)
    layout = cfg.load("freddie_layout")
    oc, pc = layout["origination"], layout["performance"]
    dirpath.mkdir(parents=True, exist_ok=True)
    end_idx = cfg.ym_to_idx(end)
    for y in years:
        orig, perf = [], []
        for i in range(loans):
            lid = f"F{y % 100:02d}Q{1 + i % 4}{i:07d}"
            fp = y * 100 + 1 + i % 12
            cs = int(np.clip(rng.normal(730, 50), 450, 840))
            rate = round(7 - 0.2 * (y - 2000) + rng.normal(0, .3), 3)
            upb = int(rng.uniform(80, 400)) * 1000
            r = dict.fromkeys(oc, "")
            r.update(credit_score=cs, first_payment_date=fp, first_time_homebuyer=rng.choice(["Y", "N"]),
                     mi_pct=0, num_units=1, occupancy_status="P", orig_cltv=80, orig_dti=int(rng.uniform(15, 50)),
                     orig_upb=upb, orig_ltv=int(rng.uniform(50, 97)), orig_interest_rate=rate, channel="R",
                     property_state="CA", property_type="SF", loan_id=lid, loan_purpose="P",
                     orig_loan_term=360, num_borrowers=rng.choice([1, 2]))
            orig.append("|".join(str(r[c]) for c in oc))
            t, dlq, bal = cfg.ym_to_idx(fp), 0, float(upb)
            age = 0
            while t <= end_idx:
                yr = t // 12
                stress = 3.0 if 2008 <= yr <= 2010 else 1.0
                h_def = 0.0015 * stress * np.exp(-(cs - 730) / 60)
                rec = dict.fromkeys(pc, "")
                age += 1
                rec.update(loan_id=lid, period=cfg.idx_to_ym(t), current_upb=f"{bal:.2f}", loan_age=age,
                           current_interest_rate=rate, delinquency_status=dlq)
                u = rng.random()
                if dlq > 0:
                    dlq = dlq + 1 if u < 0.6 else 0
                elif u < h_def * 4:
                    dlq = 1
                elif u < h_def * 4 + 0.012:
                    rec.update(zero_balance_code="01", current_upb="0", zero_balance_date=cfg.idx_to_ym(t))
                    perf.append("|".join(str(rec[c]) for c in pc))
                    break
                if dlq >= 6:
                    rec.update(zero_balance_code="09", delinquency_status=dlq, current_upb="0",
                               actual_loss=f"{0.3 * bal:.2f}", zero_balance_removal_upb=f"{bal:.2f}")
                    perf.append("|".join(str(rec[c]) for c in pc))
                    break
                perf.append("|".join(str(rec[c]) for c in pc))
                bal *= 0.998
                t += 1
        with zipfile.ZipFile(dirpath / f"sample_{y}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"sample_orig_{y}.txt", "\n".join(orig) + "\n")
            z.writestr(f"sample_svcg_{y}.txt", "\n".join(perf) + "\n")
