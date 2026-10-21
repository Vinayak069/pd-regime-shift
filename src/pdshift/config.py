"""Config loading and month arithmetic shared by every step."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, config_dir: Path | None = None) -> dict:
    path = (config_dir or ROOT / "configs") / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_all(config_dir: Path | None = None) -> dict:
    return {n: load(n, config_dir) for n in ("splits", "features", "models", "freddie_layout")}


def ym_to_idx(yyyymm: int) -> int:
    """YYYYMM -> months since year 0, so differences are month counts."""
    return (yyyymm // 100) * 12 + (yyyymm % 100) - 1


def idx_to_ym(idx: int) -> int:
    return (idx // 12) * 100 + idx % 12 + 1


def ym_str_to_idx(s: str | None) -> int | None:
    """'2011-12' -> month index; None stays None (open-ended window)."""
    if s is None:
        return None
    y, m = str(s).split("-")
    return int(y) * 12 + int(m) - 1


def window(bounds: list | None) -> tuple[int | None, int | None] | None:
    if bounds is None:
        return None
    lo, hi = bounds
    return ym_str_to_idx(lo), ym_str_to_idx(hi)


def in_window(idx, win):
    """Vectorised membership test for a (lo, hi) window, either end may be None."""
    lo, hi = win
    mask = idx == idx
    if lo is not None:
        mask = mask & (idx >= lo)
    if hi is not None:
        mask = mask & (idx <= hi)
    return mask
