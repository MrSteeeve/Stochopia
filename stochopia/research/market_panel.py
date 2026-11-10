"""Daily public P/Q market panel for the walk-forward world-model protocol.

The panel is built offline from the local Tushare/CFFEX caches and contains
only same-day observable quantities:

* ``close``: CSI500 / CSI1000 index closes (P side, realised paths);
* ``rate_3m``: 91-day continuous SHIBOR zero rate;
* ``carry``: front-curve futures-implied carry for IC / IM (Q side quote);
* ``iv_1m``: 30-day ATM implied volatility from the CSI1000 MO option chain
  (Q side quote).

Every value on day ``t`` is computed from inputs dated ``t`` (SHIBOR may lag
up to five calendar days, as in the formal builder).  Missing or
quality-rejected quotes are NaN rather than filled.  The panel is a research
input under ``data/derived/`` and is not Benchmark v3 evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..common import atomic_write_json, parse_yyyymmdd, read_csv_rows
from ..formal_market_builder import UNDERLYINGS, _curve_for
from ..formal_option_iv import build_chain_iv, parse_mo_master
from ..market_data_math import SHIBOR_COLUMNS, ShiborCurve, futures_implied_q

PANEL_SCHEMA = "stochopia.market-forecast-panel.v1"
UNDERLYING_ORDER = ("CSI500", "CSI1000")
IV_UNDERLYING = "CSI1000"
DEFAULT_OUTPUT_DIR = Path("data/derived/world_model")


@dataclass(frozen=True)
class MarketPanel:
    """Aligned daily arrays; index ``t`` is one exchange session."""

    dates: tuple[str, ...]
    close: np.ndarray  # (T, 2) CSI500, CSI1000
    rate_3m: np.ndarray  # (T,)
    carry: np.ndarray  # (T, 2) NaN when no same-day futures settlement
    iv_1m: np.ndarray  # (T,) CSI1000 ATM 30d IV, NaN when unavailable
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.dates)
        if self.close.shape != (count, 2) or self.carry.shape != (count, 2):
            raise ValueError("close and carry must be (T, 2)")
        if self.rate_3m.shape != (count,) or self.iv_1m.shape != (count,):
            raise ValueError("rate_3m and iv_1m must be (T,)")
        if list(self.dates) != sorted(self.dates) or len(set(self.dates)) != count:
            raise ValueError("panel dates must be strictly increasing")
        if not np.all(np.isfinite(self.close)) or np.any(self.close <= 0):
            raise ValueError("index closes must be positive and finite")

    @property
    def log_returns(self) -> np.ndarray:
        """``(T, 2)``; row 0 is NaN, row ``t`` is ``log(close[t] / close[t-1])``."""

        returns = np.full_like(self.close, np.nan)
        returns[1:] = np.diff(np.log(self.close), axis=0)
        return returns

    def save(self, output_dir: str | Path) -> Path:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "panel.npz"
        np.savez(
            target,
            dates=np.array(self.dates),
            close=self.close,
            rate_3m=self.rate_3m,
            carry=self.carry,
            iv_1m=self.iv_1m,
        )
        atomic_write_json(directory / "panel.json", dict(self.manifest))
        return target


def load_panel(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> MarketPanel:
    directory = Path(output_dir)
    with np.load(directory / "panel.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    manifest = json.loads((directory / "panel.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != PANEL_SCHEMA:
        raise ValueError(f"unsupported panel schema: {manifest.get('schema')!r}")
    return MarketPanel(
        dates=tuple(str(item) for item in arrays["dates"]),
        close=arrays["close"].astype(float),
        rate_3m=arrays["rate_3m"].astype(float),
        carry=arrays["carry"].astype(float),
        iv_1m=arrays["iv_1m"].astype(float),
        manifest=manifest,
    )


def _futures_carry(
    *,
    day: date,
    spot: float,
    product: str,
    curve: ShiborCurve,
    rows: list[Mapping[str, str]],
    master: Mapping[str, Mapping[str, str]],
) -> float:
    """91d futures-implied carry from same-day settlements.

    Unlike the monthly formal builder this does not reject |carry| > 0.2:
    the deep IC discounts of 2015-2016 are real quotes that a forecaster must
    face.  Only clearly corrupt values (|carry| >= 1) become NaN.
    """

    observations = []
    for row in rows:
        if row.get("product") != product:
            continue
        info = master.get(row.get("contract", ""))
        if not info:
            continue
        listed, delisted = parse_yyyymmdd(info["list_date"]), parse_yyyymmdd(info["delist_date"])
        settlement = float(row.get("settlement", "") or 0)
        if listed <= day < delisted and settlement > 0:
            observations.append(((delisted - day).days, settlement))
    if not observations:
        return math.nan
    try:
        value = futures_implied_q(observations, spot=spot, rcurve=curve, target_days=91)
    except ValueError:
        return math.nan
    return value if math.isfinite(value) and abs(value) < 1.0 else math.nan


def _coverage(values: np.ndarray, dates: tuple[str, ...]) -> dict[str, Any]:
    mask = np.isfinite(values)
    if not mask.any():
        return {"observed_days": 0, "first": None, "last": None}
    indices = np.flatnonzero(mask)
    return {
        "observed_days": int(mask.sum()),
        "first": dates[int(indices[0])],
        "last": dates[int(indices[-1])],
    }


def build_market_panel(
    *,
    tushare_root: str | Path = "data/raw/tushare",
    cffex_path: str | Path = "data/interim/cffex/ic_im_mo_daily.csv",
) -> MarketPanel:
    """Build the daily panel from local caches; no network access."""

    from ..tushare_batch import load_endpoint_rows

    tushare = Path(tushare_root)
    index_rows = load_endpoint_rows(tushare, "index_daily")
    closes: dict[str, dict[date, float]] = {}
    for underlying in UNDERLYING_ORDER:
        code = UNDERLYINGS[underlying]["ts_code"]
        series: dict[date, float] = {}
        for row in index_rows:
            if row.get("ts_code") != code:
                continue
            day, value = parse_yyyymmdd(row["trade_date"]), float(row["close"])
            if day in series and series[day] != value:
                raise ValueError(f"conflicting {code} closes on {day.isoformat()}")
            series[day] = value
        closes[underlying] = series
    common_dates = sorted(set(closes["CSI500"]) & set(closes["CSI1000"]))
    if not common_dates:
        raise ValueError("no common index trading dates")

    shibor: dict[date, dict[int, float]] = {}
    for row in load_endpoint_rows(tushare, "shibor"):
        shibor[parse_yyyymmdd(row["date"])] = {
            days: float(row[column]) for days, column in SHIBOR_COLUMNS.items()
        }

    futures_by_date: dict[date, list[Mapping[str, str]]] = defaultdict(list)
    mo_by_date: dict[date, list[Mapping[str, str]]] = defaultdict(list)
    for row in read_csv_rows(cffex_path):
        day = parse_yyyymmdd(row["trade_date"])
        if row.get("product") in {"IC", "IM"}:
            futures_by_date[day].append(row)
        elif row.get("product") == "MO":
            mo_by_date[day].append(row)
    futures_master = {
        row["symbol"]: row
        for row in load_endpoint_rows(tushare, "fut_basic")
        if row.get("fut_code") in {"IC", "IM"}
        and row.get("list_date")
        and row.get("delist_date")
    }
    mo_master = parse_mo_master(load_endpoint_rows(tushare, "opt_basic"))

    count = len(common_dates)
    close = np.empty((count, 2))
    rate = np.full(count, np.nan)
    carry = np.full((count, 2), np.nan)
    iv = np.full(count, np.nan)
    rejects: dict[str, int] = defaultdict(int)
    for t, day in enumerate(common_dates):
        for j, underlying in enumerate(UNDERLYING_ORDER):
            close[t, j] = closes[underlying][day]
        try:
            curve, _ = _curve_for(day, shibor)
        except ValueError:
            rejects["rate_unavailable"] += 1
            continue
        rate[t] = curve.continuous_rate(91)
        for j, underlying in enumerate(UNDERLYING_ORDER):
            carry[t, j] = _futures_carry(
                day=day,
                spot=close[t, j],
                product=UNDERLYINGS[underlying]["future"],
                curve=curve,
                rows=futures_by_date.get(day, []),
                master=futures_master,
            )
        chain = mo_by_date.get(day)
        if chain and day in shibor:
            spot = close[t, UNDERLYING_ORDER.index(IV_UNDERLYING)]
            try:
                result = build_chain_iv(
                    chain,
                    chain_date=day,
                    spot=spot,
                    rcurve=ShiborCurve(day, day, shibor[day]),
                    master=mo_master,
                    market="MO",
                )
            except ValueError:
                rejects["iv_chain_rejected"] += 1
                continue
            value = result.tenors.get("atm_iv_1m")
            if value is not None and math.isfinite(value) and 0.0 < value < 2.0:
                iv[t] = float(value)
            else:
                rejects["iv_gate_rejected"] += 1

    dates = tuple(day.isoformat() for day in common_dates)
    manifest = {
        "schema": PANEL_SCHEMA,
        "sources": {
            "index_daily": "tushare index_daily 000905.SH / 000852.SH close",
            "shibor": "tushare shibor eight-node curve, lag <= 5 calendar days",
            "carry": "CFFEX IC/IM same-day settlement, 91d futures-implied carry (|carry| < 1, no 0.2 gate)",
            "iv_1m": "CFFEX MO same-day settlement chain, 30d ATM total-variance IV",
        },
        "underlyings": list(UNDERLYING_ORDER),
        "iv_underlying": IV_UNDERLYING,
        "units": {
            "close": "index points",
            "rate_3m": "decimal continuous annualised",
            "carry": "decimal continuous annualised",
            "iv_1m": "decimal annualised implied volatility",
        },
        "coverage": {
            "dates": {"first": dates[0], "last": dates[-1], "sessions": count},
            "rate_3m": _coverage(rate, dates),
            "carry_CSI500": _coverage(carry[:, 0], dates),
            "carry_CSI1000": _coverage(carry[:, 1], dates),
            "iv_1m_CSI1000": _coverage(iv, dates),
        },
        "rejects": dict(sorted(rejects.items())),
        "inputs": {"tushare_root": str(tushare_root), "cffex_path": str(cffex_path)},
        "point_in_time": "every field on day t uses inputs dated <= t",
    }
    return MarketPanel(dates, close, rate, carry, iv, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the daily world-model market panel")
    parser.add_argument("--tushare-root", default="data/raw/tushare")
    parser.add_argument("--cffex-path", default="data/interim/cffex/ic_im_mo_daily.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    panel = build_market_panel(tushare_root=args.tushare_root, cffex_path=args.cffex_path)
    target = panel.save(args.output_dir)
    print(json.dumps({"panel": str(target), **panel.manifest["coverage"],
                      "rejects": panel.manifest["rejects"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
