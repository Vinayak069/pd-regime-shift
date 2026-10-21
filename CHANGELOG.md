# Changelog

- 2026-10-20: protocol frozen (splits, metrics, models).
- 2026-10-20 (before any model was fitted): crisis and COVID splits moved back one year and given a
  12-month gap like the control split, so 12-month validation labels close before the test window.
  Freddie Mac layout and zero-balance codes checked against the official file layout and user guide;
  codes 16 (RPL securitisation) and 96 (defect) treated as censoring.
- 2026-10-21 (after the first results): added, as robustness checks rather than protocol changes,
  a forbearance-free default definition (def_12_xf: 90+ dpd under forbearance or disaster relief not
  counted), 3-month early labels, early-label and drift-triggered recalibration
  (recalibrate.recalibrate_early) and a 5-seed study (pdshift.robustness). The panel was rebuilt to
  add these columns; existing columns are unchanged (same 71,021,713 rows).
