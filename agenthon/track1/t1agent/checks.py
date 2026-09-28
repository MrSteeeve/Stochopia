"""Deterministic checks on the deliverables, run after every script execution.

Errors drive a repair call. Warnings are shown to the model once, in a review
call, because some tasks legitimately contain NaN (e.g. a rolling window's
first rows) and the contract extraction can be wrong.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import Deliverable

_MAX_SCAN_BYTES = 20_000_000


@dataclass
class Problem:
    severity: str  # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.message}"


def _non_finite_paths(obj: Any, prefix: str = "", limit: int = 8) -> list[str]:
    bad: list[str] = []

    def walk(x: Any, path: str) -> None:
        if len(bad) >= limit:
            return
        if isinstance(x, float) and not math.isfinite(x):
            bad.append(path or "<root>")
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj, prefix)
    return bad


def _check_table(path: Path, d: Deliverable) -> list[Problem]:
    import pandas as pd

    fmt = d.fmt
    try:
        if fmt == "csv":
            df = pd.read_csv(path)
        elif fmt in {"parquet", "pqt"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_excel(path)
    except Exception as exc:  # noqa: BLE001 - any read failure is a problem to report
        return [Problem("error", f"{d.name}: cannot be read as {fmt}: {type(exc).__name__}: {exc}")]
    out: list[Problem] = []
    if df.empty:
        out.append(Problem("error", f"{d.name}: table has no rows"))
    missing = [c for c in d.columns if c not in df.columns]
    if missing:
        out.append(
            Problem("error", f"{d.name}: missing columns {missing}; present columns {list(df.columns)[:40]}")
        )
    nulls = {c: int(n) for c, n in df.isna().sum().items() if n}
    if nulls:
        out.append(Problem("warning", f"{d.name}: null values per column {dict(list(nulls.items())[:15])} of {len(df)} rows"))
    num = df.select_dtypes("number")
    if not num.empty:
        import numpy as np

        infs = {c: int(n) for c, n in np.isinf(num).sum().items() if n}
        if infs:
            out.append(Problem("error", f"{d.name}: infinite values per column {infs}"))
    return out


def _check_json(path: Path, d: Deliverable) -> list[Problem]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [Problem("error", f"{d.name}: invalid JSON: {exc}")]
    out: list[Problem] = []
    if d.keys:
        if not isinstance(obj, dict):
            out.append(Problem("error", f"{d.name}: expected a JSON object with keys {d.keys}, got {type(obj).__name__}"))
        else:
            missing = [k for k in d.keys if k not in obj]
            if missing:
                out.append(Problem("error", f"{d.name}: missing top-level keys {missing}; present keys {list(obj)[:40]}"))
    bad = _non_finite_paths(obj)
    if bad:
        out.append(Problem("error", f"{d.name}: NaN/Infinity at {bad} (JSON has no NaN; the checker will fail)"))
    return out


def check_outputs(deliverables: list[Deliverable], out_dir: Path, canaries: list[str]) -> list[Problem]:
    problems: list[Problem] = []
    written = sorted(p for p in out_dir.rglob("*") if p.is_file())
    if not deliverables:
        if not written:
            problems.append(Problem("error", "no output files were written"))
    for d in deliverables:
        path = out_dir / d.name
        if not path.is_file():
            problems.append(Problem("error", f"{d.name}: required output file was not written"))
            continue
        if path.stat().st_size == 0:
            problems.append(Problem("error", f"{d.name}: file is empty"))
            continue
        if d.fmt == "json":
            problems += _check_json(path, d)
        elif d.fmt in {"csv", "parquet", "pqt", "xlsx"}:
            problems += _check_table(path, d)
    for f in written:
        if f.name in {"reward.json", "reward.txt", "pytest_report.json"}:
            problems.append(Problem("error", f"{f.name}: must not be written by the solution"))
        if canaries and f.stat().st_size < _MAX_SCAN_BYTES:
            text = f.read_bytes().decode("utf-8", errors="ignore").lower()
            if any(c.lower() in text for c in canaries):
                problems.append(Problem("error", f"{f.name}: contains the task's canary GUID; remove it"))
    return problems


def score(problems: list[Problem]) -> tuple[int, int]:
    """Lower is better: (errors, warnings)."""
    return (
        sum(p.severity == "error" for p in problems),
        sum(p.severity == "warning" for p in problems),
    )
