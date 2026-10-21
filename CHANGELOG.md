# Changelog

- 2026-10-20: protocol frozen (splits, metrics, models).
- 2026-10-20 (before any model was fitted): crisis and COVID splits moved back one year and given a
  12-month gap like the control split, so 12-month validation labels close before the test window.
  Freddie Mac layout and zero-balance codes checked against the official file layout and user guide;
  codes 16 (RPL securitisation) and 96 (defect) treated as censoring.
