# pd-regime-shift

Calibration of lifetime PD models under macroeconomic regime shifts (IFRS 9), on the Freddie Mac Single-Family Loan-Level Dataset.

The research design, splits and metrics are frozen in `configs/` and in the project workbook. Change them only with a dated note in `CHANGELOG.md`.

## Layout
| Path | What it does |
|---|---|
| `configs/splits.yaml` | Experiment windows (control, crisis 2008, COVID) and label delay |
| `configs/features.yaml` | Loan and macro features, macro publication lags |
| `configs/models.yaml` | Model specs, sampling sizes, recalibration, NSGA-II, ECL scenarios |
| `configs/freddie_layout.yaml` | Column order of the Freddie Mac files and event codes (checked against the 2026 user guide) |
| `src/pdshift/data.py` | Read raw origination and performance files (zipped or extracted), with layout checks |
| `src/pdshift/panel.py` | Loan-month panel: at-risk rows, default / prepayment events, horizon labels, loss records |
| `src/pdshift/macro.py` | FRED series to monthly features with publication lags |
| `src/pdshift/features.py` | Model matrix: loan + macro + refi incentive + mark-to-market LTV |
| `src/pdshift/dataset.py` | Filtered reads of the panel; loan buckets keep fitting and evaluation loans disjoint |
| `src/pdshift/models.py` | Cause-specific monthly hazards: logit (B0, B1) and LightGBM (M1) |
| `src/pdshift/lifetime.py` | Hazards to 12/36/60-month and lifetime PD along oracle or scenario macro paths |
| `src/pdshift/metrics.py` | AUC, Brier, ECE, calibration intercept/slope, CITL, DeLong, loan-clustered bootstrap |
| `src/pdshift/drift.py` | PSI and classifier two-sample test |
| `src/pdshift/recalibrate.py` | none / intercept / Platt / isotonic / importance-weighted Platt, static or rolling, label-delay safe |
| `src/pdshift/selection.py` | NSGA-II over LightGBM hyperparameters and recalibration method |
| `src/pdshift/ecl.py` | 12-month and lifetime ECL, scenario weighting, realised vs predicted |
| `src/pdshift/experiment.py` | One experiment end to end; writes `results/<experiment>/` |
| `tests/` | Protocol guards and unit tests; `tests/synth.py` makes tiny synthetic files for tests only |

## Protocol in one paragraph
Monthly default and prepayment hazards are fitted on loan-months whose outcome month is inside the
training window, using loans in buckets 0-49 only. Hyperparameters and recalibration are chosen on
validation snapshots whose 12-month labels close before the test window (hence the 12-month gap).
At test month t a recalibrator may use only snapshots s <= t - 12. Evaluation, recalibration and
drift use loans in buckets 50-99. The deployable forecast holds macro inputs at their time-t values
(persistence); the oracle path, which uses realised macro data, is reported only as a bound.

## Getting the data
1. Register at Freddie Mac (Single-Family Loan-Level Dataset) and download the **standard sample**
   files `sample_YYYY.zip` (50,000 loans per origination year), 1999 onwards.
2. Put the zips in `data/raw/freddie/` unchanged.
3. `make fred` downloads the FRED series into `data/raw/fred/`.

## Reproduce
```
make test         # unit tests on synthetic fixtures
make fred         # FRED series
make panel        # raw files -> data/processed/{loans,panel,losses}/
make smoke        # quick check on the control experiment
make experiments  # B0, B1, M1 on control, crisis_2008, covid (5 seeds, 1000 bootstrap draws)
make select       # NSGA-II selection per experiment
```
Outputs per experiment in `results/<experiment>/`: `metrics.csv`, `monthly.csv`, `recalibration.csv`,
`delong.csv`, `bootstrap.csv`, `drift.csv`, `ecl_monthly.csv`, `ecl_summary.csv`, `selection.csv`,
`run_info.json`, `predictions.parquet`.
