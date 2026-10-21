"""FRED macro series as monthly features, with publication lags applied.

Every value placed at month t is one that was published by the end of month t:
the series is moved forward by its lag (configs/features.yaml), so month t sees
the observation dated t - lag. Weekly series are averaged within the month and
quarterly series are carried forward from the first month of their quarter
before the lag is applied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def read_fred_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
    val_col = next(c for c in df.columns if c != date_col)
    s = pd.to_numeric(df[val_col], errors="coerce")  # FRED writes "." for missing
    s.index = pd.to_datetime(df[date_col])
    return s.dropna().rename(val_col)


def to_monthly(s: pd.Series, freq: str = "m") -> pd.Series:
    """Index by month index (cfg.ym_to_idx). Weekly -> mean of the month's weeks;
    quarterly (dated at quarter start) -> the same value in each of the quarter's three months."""
    m = s.groupby(s.index.year * 12 + s.index.month - 1).mean()
    if freq == "q":
        m = pd.concat([m.rename(lambda i, k=k: i + k) for k in range(3)]).sort_index()
    return m.reindex(np.arange(m.index.min(), m.index.max() + 1))


def macro_table(series: dict[str, pd.Series], spec: dict) -> pd.DataFrame:
    """Monthly table of macro features keyed by month index, publication lags applied."""
    out = {}
    for name, opts in spec.items():
        m = to_monthly(series[name], opts.get("freq", "m"))
        m.index = m.index + int(opts["lag"])          # value dated t-lag becomes usable at t
        for tr in opts["transform"]:
            if tr == "level":
                out[f"{name.lower()}"] = m
            elif tr == "diff_12":
                out[f"{name.lower()}_d12"] = m - m.shift(12)
            elif tr == "yoy":
                out[f"{name.lower()}_yoy"] = np.log(m / m.shift(12))
            elif tr == "index":
                out[f"{name.lower()}_idx"] = m
            else:
                raise ValueError(f"unknown transform {tr!r} for {name}")
    tab = pd.DataFrame(out)
    tab.index.name = "t"
    return tab


def load_macro(fred_dir: Path | None = None, spec: dict | None = None) -> pd.DataFrame:
    fred_dir = Path(fred_dir or cfg.ROOT / "data/raw/fred")
    spec = spec or cfg.load("features")["macro"]
    series = {n: read_fred_csv(fred_dir / f"{n}.csv") for n in spec}
    return macro_table(series, spec)


def download(fred_dir: Path, names) -> None:
    import urllib.request
    fred_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        urllib.request.urlretrieve(FRED_URL.format(n), fred_dir / f"{n}.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download FRED series and print the lagged macro table")
    ap.add_argument("--fred", default=str(cfg.ROOT / "data/raw/fred"))
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args(argv)
    spec = cfg.load("features")["macro"]
    if args.download:
        download(Path(args.fred), spec)
    tab = load_macro(Path(args.fred), spec)
    tab.index = [cfg.idx_to_ym(i) for i in tab.index]
    print(tab.dropna().iloc[[0, -1]].T)


if __name__ == "__main__":
    main()
