"""Synthetic offline tests for the Phase-A formal market builder."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from mirage.benchmark import MarketSnapshot, load_market_snapshots
from mirage.formal_market_builder import _atomic_write, _csv_bytes, run_base_build
from mirage.market_data_math import ShiborCurve


def _write_endpoint(root: Path, endpoint: str, rows: list[dict[str, str]], params: dict) -> None:
    directory = root / endpoint
    directory.mkdir(parents=True)
    csv_path = directory / "fixture.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "endpoint": endpoint,
        "status": "complete",
        "params": params,
        "row_count": len(rows),
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    (directory / "fixture.json").write_text(json.dumps(metadata), encoding="utf-8")


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = tmp_path / "data"
    tushare = data / "raw" / "tushare"
    tushare.mkdir(parents=True)
    (tushare / "manifest.sha256").write_text("fixture\n", encoding="utf-8")
    cffex_raw = data / "raw" / "cffex"
    cffex_raw.mkdir(parents=True)
    (cffex_raw / "manifest.sha256").write_text("fixture\n", encoding="utf-8")

    start, end = date(2022, 6, 1), date(2025, 12, 31)
    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    index_rows = []
    values = {"000905.SH": 5000.0, "000852.SH": 6000.0}
    for current in dates:
        for code in values:
            values[code] *= 1.0002 if current.toordinal() % 3 else 0.9997
            index_rows.append({"ts_code": code, "trade_date": current.strftime("%Y%m%d"), "close": str(values[code])})
    _write_endpoint(tushare, "index_daily", index_rows, {})

    shibor_rows = [
        {"date": current.strftime("%Y%m%d"), "on": "2", "1w": "2", "2w": "2", "1m": "2", "3m": "2", "6m": "2", "9m": "2", "1y": "2"}
        for current in dates
    ]
    _write_endpoint(tushare, "shibor", shibor_rows, {})

    month_ends = {}
    spot_by_date = {}
    for row in index_rows:
        current = date.fromisoformat(f"{row['trade_date'][:4]}-{row['trade_date'][4:6]}-{row['trade_date'][6:]}")
        if current.year >= 2023:
            month_ends[(current.year, current.month)] = current
            spot_by_date[(row["ts_code"], current)] = float(row["close"])
    futures_rows, masters = [], []
    curve_rates = {days: 2.0 for days in (1, 7, 14, 30, 91, 182, 274, 365)}
    for current in month_ends.values():
        curve = ShiborCurve(current, current, curve_rates)
        rate, maturity, expiry = curve.continuous_rate(91), 91 / 365, current + timedelta(days=91)
        for product, code in (("IC", "000905.SH"), ("IM", "000852.SH")):
            contract = f"{product}{current:%y%m}X"
            settlement = spot_by_date[(code, current)] * math.exp((rate - 0.01) * maturity)
            futures_rows.append({"trade_date": current.strftime("%Y%m%d"), "product": product, "contract": contract, "settlement": str(settlement)})
            masters.append({
                "symbol": contract, "fut_code": product,
                "list_date": (current - timedelta(days=30)).strftime("%Y%m%d"),
                "delist_date": expiry.strftime("%Y%m%d"),
            })
    _write_endpoint(tushare, "fut_basic", masters, {"exchange": "CFFEX", "fut_type": "1"})
    cffex = data / "interim" / "cffex" / "ic_im_mo_daily.csv"
    cffex.parent.mkdir(parents=True)
    with cffex.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(futures_rows[0]))
        writer.writeheader()
        writer.writerows(futures_rows)
    return tushare, cffex, data / "derived"


def test_phase_a_build_is_complete_index_only_and_leakage_safe(tmp_path):
    tushare, cffex, derived = _fixtures(tmp_path)

    gate = run_base_build(tushare_root=tushare, cffex_path=cffex, output_root=derived)
    snapshots = load_market_snapshots(derived / "market_snapshots.csv")
    provenance = [json.loads(line) for line in (derived / "market_snapshot_provenance.jsonl").read_text().splitlines()]

    assert gate["ready"] is True
    assert gate["rows_by_underlying"] == {"CSI1000": 36, "CSI500": 36}
    assert gate["carry"] == {"observed": 72, "fallback_zero": 0}
    assert len(snapshots) == 72
    assert all(item.atm_iv_1m is item.atm_iv_3m is item.atm_iv_6m is None for item in snapshots)
    assert all(item.pricing_volatility()[1] == "realized_vol_20d" for item in snapshots)
    assert all(row["spot"]["ts_code"] in {"000905.SH", "000852.SH"} for row in provenance)
    assert all(row["max_input_date"] <= row["as_of"] for row in provenance)
    assert "510500" not in (derived / "market_snapshots.csv").read_text()
    assert (derived / "manifest.sha256").is_file()


def test_full_phase_csv_preserves_iv_overlay_values(tmp_path):
    target = tmp_path / "snapshots.csv"
    snapshot = MarketSnapshot(
        "CSI500_2025H1", 1, date(2025, 1, 31), "CSI500", 5000, 0.02,
        realized_vol_20d=0.2, realized_vol_60d=0.21,
        atm_iv_1m=0.22, atm_iv_3m=0.23, atm_iv_6m=0.24,
        source="base; sse_510500_etf_iv_proxy",
    )
    _atomic_write(target, _csv_bytes([snapshot]))
    loaded = load_market_snapshots(target)[0]
    assert (loaded.atm_iv_1m, loaded.atm_iv_3m, loaded.atm_iv_6m) == (0.22, 0.23, 0.24)


def test_synthetic_full_builder_preserves_72_base_rows(tmp_path, monkeypatch):
    tushare, cffex, derived = _fixtures(tmp_path)
    _write_endpoint(tushare, "opt_daily", [{"trade_date": "20250131"}], {})
    _write_endpoint(tushare, "opt_basic", [{"symbol": "fixture"}], {})
    _write_endpoint(tushare, "fund_daily", [{"ts_code": "510500.SH", "trade_date": "20250131", "close": "5"}], {})

    def fake_overlay(snapshots, provenance, **_):
        updated = []
        updated_provenance = []
        for snapshot, row in zip(snapshots, provenance):
            label = "sse_510500_etf_iv_proxy" if snapshot.underlying == "CSI500" else "cffex_mo_index_iv"
            updated.append(replace(snapshot, atm_iv_3m=0.23, source=f"{snapshot.source}; {label}"))
            item = dict(row)
            item["option_iv"] = {
                "source_label": label, "selected_date": snapshot.as_of.isoformat(),
                "staleness_trading_days": 0, "expiry_contracts": {}, "selected_versions": ["M"],
            }
            updated_provenance.append(item)
        stats = {
            "any_iv": 72, "all3_iv": 0, "total": 72, "any_rate": 1.0, "all3_rate": 0.0,
            "by_underlying": {}, "by_tenor": {}, "lag_counts": {"0": 72},
            "reject_counts": {}, "warnings": ["all-three-IV coverage below 60%"],
        }
        return updated, updated_provenance, [], stats

    monkeypatch.setattr("mirage.formal_market_builder.apply_full_phase", fake_overlay)
    gate = run_base_build(tushare_root=tushare, cffex_path=cffex, output_root=derived, phase="full")
    assert gate["ready"] is True
    assert len(load_market_snapshots(derived / "market_snapshots.csv")) == 72
