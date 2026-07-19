"""Build leakage-safe month-end snapshots from public daily close files."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import pstdev

from .benchmark import BenchmarkError, MarketSnapshot


@dataclass(frozen=True)
class DailyClose:
    as_of: date
    underlying: str
    close: float
    source: str


def load_daily_closes(path: str | Path) -> list[DailyClose]:
    """Load date, underlying, close, source; duplicate dates are rejected."""
    p = Path(path)
    with p.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "underlying", "close", "source"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise BenchmarkError(f"daily close CSV missing columns: {sorted(missing)}")
        rows: list[DailyClose] = []
        for line, row in enumerate(reader, start=2):
            try:
                item = DailyClose(
                    as_of=date.fromisoformat(row["date"].strip()),
                    underlying=row["underlying"].strip(),
                    close=float(row["close"]),
                    source=row["source"].strip(),
                )
            except (TypeError, ValueError) as exc:
                raise BenchmarkError(f"invalid daily close at {p}:{line}: {exc}") from exc
            if item.close <= 0 or not item.underlying or not item.source:
                raise BenchmarkError(f"invalid daily close at {p}:{line}")
            rows.append(item)
    keys = [(row.underlying, row.as_of) for row in rows]
    if len(keys) != len(set(keys)):
        raise BenchmarkError("duplicate underlying/date in daily close CSV")
    return sorted(rows, key=lambda row: (row.underlying, row.as_of))


def _return(closes: list[float], lookback: int) -> float | None:
    if len(closes) <= lookback:
        return None
    return closes[-1] / closes[-lookback - 1] - 1.0


def _realized_vol(closes: list[float], lookback: int) -> float | None:
    if len(closes) <= lookback:
        return None
    selected = closes[-lookback - 1:]
    returns = [math.log(selected[i] / selected[i - 1]) for i in range(1, len(selected))]
    return pstdev(returns) * math.sqrt(252.0)


def _drawdown(closes: list[float], lookback: int = 126) -> float | None:
    if len(closes) < 2:
        return None
    selected = closes[-min(len(closes), lookback):]
    return selected[-1] / max(selected) - 1.0


def _regime(return_20d: float | None, vol_20d: float | None) -> str:
    if return_20d is None or vol_20d is None:
        return "insufficient_warmup"
    vol = "high_vol" if vol_20d >= 0.25 else "low_vol"
    trend = "uptrend" if return_20d >= 0.03 else "downtrend" if return_20d <= -0.03 else "sideways"
    return f"{vol}_{trend}"


def build_monthly_snapshots(
    daily: list[DailyClose],
    *,
    evaluation_start: date = date(2023, 1, 1),
    evaluation_end: date = date(2025, 12, 31),
    risk_free_rate: float = 0.02,
    atm_iv: dict[tuple[str, date], dict[str, float]] | None = None,
) -> list[MarketSnapshot]:
    """Build past-only month-end features and six-round half-year episodes."""
    atm_iv = atm_iv or {}
    grouped: dict[str, list[DailyClose]] = {}
    for row in daily:
        grouped.setdefault(row.underlying, []).append(row)
    output: list[MarketSnapshot] = []
    for underlying, rows in grouped.items():
        month_ends: dict[tuple[int, int], int] = {}
        for index, row in enumerate(rows):
            if evaluation_start <= row.as_of <= evaluation_end:
                month_ends[(row.as_of.year, row.as_of.month)] = index
        for (year, month), index in sorted(month_ends.items()):
            row = rows[index]
            history = [item.close for item in rows[: index + 1]]
            half = "H1" if month <= 6 else "H2"
            round_num = month if month <= 6 else month - 6
            iv = atm_iv.get((underlying, row.as_of), {})
            rv20 = _realized_vol(history, 20)
            rv60 = _realized_vol(history, 60)
            snapshot = MarketSnapshot(
                episode_id=f"{underlying}_{year}{half}",
                round_num=round_num,
                as_of=row.as_of,
                underlying=underlying,
                spot=row.close,
                risk_free_rate=risk_free_rate,
                return_20d=_return(history, 20),
                realized_vol_20d=rv20,
                realized_vol_60d=rv60,
                drawdown_6m=_drawdown(history),
                atm_iv_1m=iv.get("atm_iv_1m"),
                atm_iv_3m=iv.get("atm_iv_3m"),
                atm_iv_6m=iv.get("atm_iv_6m"),
                carry_rate=iv.get("carry_rate", 0.0),
                regime=_regime(_return(history, 20), rv20),
                source=row.source,
            )
            snapshot.pricing_volatility()
            output.append(snapshot)
    return sorted(output, key=lambda item: (item.episode_id, item.round_num))


def write_market_snapshots(path: str | Path, snapshots: list[MarketSnapshot]) -> None:
    """Write the public schema with explicit provenance."""
    fields = [
        "episode_id", "round", "date", "underlying", "spot", "risk_free_rate",
        "return_20d", "realized_vol_20d", "realized_vol_60d", "drawdown_6m",
        "atm_iv_1m", "atm_iv_3m", "atm_iv_6m", "carry_rate", "regime", "source",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in snapshots:
            writer.writerow({
                "episode_id": item.episode_id,
                "round": item.round_num,
                "date": item.as_of.isoformat(),
                "underlying": item.underlying,
                "spot": item.spot,
                "risk_free_rate": item.risk_free_rate,
                "return_20d": item.return_20d,
                "realized_vol_20d": item.realized_vol_20d,
                "realized_vol_60d": item.realized_vol_60d,
                "drawdown_6m": item.drawdown_6m,
                "atm_iv_1m": item.atm_iv_1m,
                "atm_iv_3m": item.atm_iv_3m,
                "atm_iv_6m": item.atm_iv_6m,
                "carry_rate": item.carry_rate,
                "regime": item.regime,
                "source": item.source,
            })
