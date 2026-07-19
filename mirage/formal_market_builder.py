"""Build the Phase-A formal monthly market panel from local audited data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .benchmark import MarketSnapshot, load_market_snapshots
from .market_data_math import (
    ShiborCurve,
    futures_implied_q,
    realized_volatility,
    trailing_drawdown,
    trailing_return,
)
from .formal_option_iv import apply_full_phase


RAW_TUSHARE = Path("data/raw/tushare")
CFFEX_INTERIM = Path("data/interim/cffex/ic_im_mo_daily.csv")
OUTPUT_ROOT = Path("data/derived")
FIELDS = [
    "episode_id", "round", "date", "underlying", "spot", "risk_free_rate",
    "return_20d", "realized_vol_20d", "realized_vol_60d", "drawdown_6m",
    "atm_iv_1m", "atm_iv_3m", "atm_iv_6m", "carry_rate", "regime", "source",
]
UNDERLYINGS = {
    "CSI500": {"ts_code": "000905.SH", "future": "IC"},
    "CSI1000": {"ts_code": "000852.SH", "future": "IM"},
}
SHIBOR_COLUMNS = {1: "on", 7: "1w", 14: "2w", 30: "1m", 91: "3m", 182: "6m", 274: "9m", 365: "1y"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_endpoint(root: Path, endpoint: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metadata_path in sorted((root / endpoint).glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        csv_path = metadata_path.with_suffix(".csv")
        if metadata.get("endpoint") != endpoint or metadata.get("status") != "complete":
            raise ValueError(f"invalid {endpoint} cache metadata")
        if not csv_path.is_file() or metadata.get("sha256") != _sha256(csv_path):
            raise ValueError(f"invalid {endpoint} cache hash")
        cached = _read_csv(csv_path)
        if len(cached) != metadata.get("row_count"):
            raise ValueError(f"invalid {endpoint} cache row count")
        rows.extend(cached)
    if not rows:
        raise ValueError(f"no cached rows for {endpoint}")
    return rows


def _parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


def _regime(return_20d: float, vol_20d: float) -> str:
    volatility = "high_vol" if vol_20d >= 0.25 else "low_vol"
    trend = "uptrend" if return_20d >= 0.03 else "downtrend" if return_20d <= -0.03 else "sideways"
    return f"{volatility}_{trend}"


def _episode(year: int, month: int, underlying: str) -> tuple[str, int]:
    half = "H1" if month <= 6 else "H2"
    return f"{underlying}_{year}{half}", month if month <= 6 else month - 6


def _month_ends(rows: Sequence[tuple[date, float]]) -> list[int]:
    result: dict[tuple[int, int], int] = {}
    for index, (asof, _) in enumerate(rows):
        if date(2023, 1, 1) <= asof <= date(2025, 12, 31):
            result[(asof.year, asof.month)] = index
    return [result[key] for key in sorted(result)]


def _curve_for(asof: date, shibor: Mapping[date, Mapping[int, float]]) -> tuple[ShiborCurve, date]:
    eligible = [source for source in shibor if source <= asof and (asof - source).days <= 5]
    if not eligible:
        raise ValueError(f"no lag-compliant SHIBOR for {asof}")
    source = max(eligible)
    return ShiborCurve(asof, source, shibor[source]), source


def _carry_for(
    *,
    asof: date,
    spot: float,
    product: str,
    curve: ShiborCurve,
    futures_by_date: Mapping[date, Sequence[Mapping[str, str]]],
    master_by_symbol: Mapping[str, Mapping[str, str]],
) -> tuple[float, list[str], list[str]]:
    observations = []
    contracts = []
    for row in futures_by_date.get(asof, ()):
        if row.get("product") != product:
            continue
        master = master_by_symbol.get(row.get("contract", ""))
        if not master:
            continue
        list_date, delist_date = _parse_yyyymmdd(master["list_date"]), _parse_yyyymmdd(master["delist_date"])
        settlement = float(row.get("settlement", "") or 0)
        if list_date <= asof <= delist_date and delist_date > asof and settlement > 0:
            observations.append(((delist_date - asof).days, settlement))
            contracts.append(row["contract"])
    if not observations:
        return 0.0, [], ["missing same-day active futures settlement; carry set explicitly to 0"]
    carry = futures_implied_q(observations, spot=spot, rcurve=curve, target_days=91)
    if not -0.2 <= carry <= 0.2:
        raise ValueError(f"futures-implied carry outside [-0.2, 0.2] for {product} {asof}")
    return carry, sorted(contracts), []


def build_base_panel(
    *,
    index_rows: Iterable[Mapping[str, str]],
    shibor_rows: Iterable[Mapping[str, str]],
    futures_rows: Iterable[Mapping[str, str]],
    futures_master_rows: Iterable[Mapping[str, str]],
    raw_manifest_hashes: Mapping[str, str],
) -> tuple[list[MarketSnapshot], list[dict[str, Any]]]:
    indices: dict[str, list[tuple[date, float]]] = {}
    index_input = list(index_rows)
    for underlying, spec in UNDERLYINGS.items():
        rows = sorted(
            (_parse_yyyymmdd(row["trade_date"]), float(row["close"]))
            for row in index_input if row.get("ts_code") == spec["ts_code"]
        )
        if len({item[0] for item in rows}) != len(rows) or any(value <= 0 for _, value in rows):
            raise ValueError(f"invalid index history for {underlying}")
        indices[underlying] = rows
    shibor: dict[date, dict[int, float]] = {}
    for row in shibor_rows:
        source = _parse_yyyymmdd(row["date"])
        shibor[source] = {days: float(row[column]) for days, column in SHIBOR_COLUMNS.items()}
    futures_by_date: dict[date, list[Mapping[str, str]]] = defaultdict(list)
    for row in futures_rows:
        futures_by_date[_parse_yyyymmdd(row["trade_date"])].append(row)
    master_by_symbol = {
        row["symbol"]: row for row in futures_master_rows
        if row.get("fut_code") in {"IC", "IM"}
        and row.get("list_date") and row.get("delist_date")
    }

    snapshots: list[MarketSnapshot] = []
    provenance: list[dict[str, Any]] = []
    for underlying, rows in indices.items():
        spec = UNDERLYINGS[underlying]
        for index in _month_ends(rows):
            asof, spot = rows[index]
            history = [value for _, value in rows[: index + 1]]
            ret20 = trailing_return(history, 20)
            rv20 = realized_volatility(history, 20)
            rv60 = realized_volatility(history, 60)
            drawdown = trailing_drawdown(history, 126)
            if None in {ret20, rv20, rv60, drawdown}:
                raise ValueError(f"insufficient strict history for {underlying} {asof}")
            curve, rate_source = _curve_for(asof, shibor)
            rate = curve.continuous_rate(91)
            carry, contracts, warnings = _carry_for(
                asof=asof, spot=spot, product=spec["future"], curve=curve,
                futures_by_date=futures_by_date, master_by_symbol=master_by_symbol,
            )
            episode_id, round_num = _episode(asof.year, asof.month, underlying)
            snapshot = MarketSnapshot(
                episode_id, round_num, asof, underlying, spot, rate,
                return_20d=ret20, realized_vol_20d=rv20, realized_vol_60d=rv60,
                drawdown_6m=drawdown, atm_iv_1m=None, atm_iv_3m=None, atm_iv_6m=None,
                carry_rate=carry, regime=_regime(ret20, rv20),
                source="Tushare index_daily/SHIBOR; CFFEX futures settlement",
            )
            snapshot.pricing_volatility()
            max_input_date = max(asof, rate_source, asof if contracts else rate_source)
            if max_input_date > asof:
                raise AssertionError("future-dated input detected")
            snapshots.append(snapshot)
            provenance.append({
                "episode_id": episode_id,
                "round": round_num,
                "as_of": asof.isoformat(),
                "underlying": underlying,
                "raw_manifest_hashes": dict(raw_manifest_hashes),
                "spot": {"ts_code": spec["ts_code"], "observation_date": asof.isoformat()},
                "rate": {"observation_date": rate_source.isoformat(), "target_days": 91},
                "carry": {
                    "product": spec["future"],
                    "observation_date": asof.isoformat() if contracts else None,
                    "contracts": contracts,
                    "fallback_zero": not contracts,
                },
                "rv_counts": {"return_20d": 20, "rv20": 20, "rv60": 60, "drawdown_levels": 126},
                "atm_iv_phase": "base_null",
                "warnings": warnings,
                "max_input_date": max_input_date.isoformat(),
            })
    return sorted(snapshots, key=lambda row: (row.episode_id, row.round_num)), sorted(
        provenance, key=lambda row: (row["episode_id"], row["round"])
    )


def _csv_bytes(snapshots: Sequence[MarketSnapshot]) -> bytes:
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for item in snapshots:
        writer.writerow({
            "episode_id": item.episode_id, "round": item.round_num, "date": item.as_of.isoformat(),
            "underlying": item.underlying, "spot": item.spot, "risk_free_rate": item.risk_free_rate,
            "return_20d": item.return_20d, "realized_vol_20d": item.realized_vol_20d,
            "realized_vol_60d": item.realized_vol_60d, "drawdown_6m": item.drawdown_6m,
            "atm_iv_1m": item.atm_iv_1m if item.atm_iv_1m is not None else "",
            "atm_iv_3m": item.atm_iv_3m if item.atm_iv_3m is not None else "",
            "atm_iv_6m": item.atm_iv_6m if item.atm_iv_6m is not None else "",
            "carry_rate": item.carry_rate,
            "regime": item.regime, "source": item.source,
        })
    return buffer.getvalue().encode("utf-8")


def _gate(snapshots: Sequence[MarketSnapshot], provenance: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(item.underlying for item in snapshots)
    episodes = Counter(item.episode_id for item in snapshots)
    carry_observed = sum(not row["carry"]["fallback_zero"] for row in provenance)
    checks = {
        "rows_72": len(snapshots) == 72,
        "rows_36_per_underlying": counts == {"CSI500": 36, "CSI1000": 36},
        "episodes_12_x_6": len(episodes) == 12 and set(episodes.values()) == {6},
        "spot_rv_rate_complete": all(
            item.spot > 0 and item.risk_free_rate >= 0 and item.realized_vol_20d is not None
            and item.realized_vol_60d is not None for item in snapshots
        ),
        "carry_complete": len(provenance) == len(snapshots) and all(row.get("carry") for row in provenance),
        "max_input_not_after_asof": all(row["max_input_date"] <= row["as_of"] for row in provenance),
        "index_spot_only": all(row["spot"]["ts_code"] in {"000905.SH", "000852.SH"} for row in provenance),
        "phase_a_iv_null": all(item.atm_iv_1m is None and item.atm_iv_3m is None and item.atm_iv_6m is None for item in snapshots),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "base",
        "checks": checks,
        "ready": all(checks.values()),
        "rows": len(snapshots),
        "rows_by_underlying": dict(sorted(counts.items())),
        "episodes": len(episodes),
        "carry": {"observed": carry_observed, "fallback_zero": len(provenance) - carry_observed},
        "warnings": [warning for row in provenance for warning in row["warnings"]],
    }


def run_base_build(
    *, tushare_root: Path = RAW_TUSHARE, cffex_path: Path = CFFEX_INTERIM,
    output_root: Path = OUTPUT_ROOT, phase: str = "base",
) -> dict[str, Any]:
    if phase not in {"base", "full"}:
        raise ValueError("phase must be base or full")
    manifest_hashes = {
        "tushare": _sha256(tushare_root / "manifest.sha256"),
        "cffex": _sha256(cffex_path.parents[2] / "raw" / "cffex" / "manifest.sha256"),
    }
    index_rows = _load_endpoint(tushare_root, "index_daily")
    shibor_rows = _load_endpoint(tushare_root, "shibor")
    futures_rows = _read_csv(cffex_path)
    snapshots, provenance = build_base_panel(
        index_rows=index_rows,
        shibor_rows=shibor_rows,
        futures_rows=futures_rows,
        futures_master_rows=_load_endpoint(tushare_root, "fut_basic"),
        raw_manifest_hashes=manifest_hashes,
    )
    expiry_rows: list[dict[str, Any]] = []
    iv_stats: dict[str, Any] | None = None
    if phase == "full":
        snapshots, provenance, expiry_rows, iv_stats = apply_full_phase(
            snapshots, provenance,
            sse_rows=_load_endpoint(tushare_root, "opt_daily"),
            mo_rows=futures_rows,
            option_master_rows=_load_endpoint(tushare_root, "opt_basic"),
            fund_rows=_load_endpoint(tushare_root, "fund_daily"),
            index_rows=index_rows,
            shibor_rows=shibor_rows,
        )
    snapshot_path = output_root / "market_snapshots.csv"
    provenance_path = output_root / "market_snapshot_provenance.jsonl"
    gate_path = output_root / "gate_report.json"
    _atomic_write(snapshot_path, _csv_bytes(snapshots))
    loaded = load_market_snapshots(snapshot_path)
    provenance_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in provenance).encode("utf-8")
    _atomic_write(provenance_path, provenance_bytes)
    gate = _gate(loaded, provenance)
    gate["phase"] = phase
    gate["checks"]["load_market_snapshots"] = len(loaded) == len(snapshots)
    if phase == "full":
        gate["checks"].pop("phase_a_iv_null", None)
        expected_labels = {"CSI500": "sse_510500_etf_iv_proxy", "CSI1000": "cffex_mo_index_iv"}
        gate["checks"].update({
            "selected_adjusted_contracts_zero": all(
                set(row["option_iv"]["selected_versions"]) <= {"M"}
                for row in provenance
            ),
            "iv_lag_at_most_2": all(
                row["option_iv"]["staleness_trading_days"] is None
                or row["option_iv"]["staleness_trading_days"] <= 2 for row in provenance
            ),
            "iv_source_labels": all(
                row["option_iv"]["source_label"] == expected_labels[row["underlying"]]
                for row in provenance
            ),
            "public_iv_or_fallback_source": all(
                (
                    any(value is not None for value in (item.atm_iv_1m, item.atm_iv_3m, item.atm_iv_6m))
                    and expected_labels[item.underlying] in item.source
                )
                or (
                    all(value is None for value in (item.atm_iv_1m, item.atm_iv_3m, item.atm_iv_6m))
                    and "rv_fallback" in item.source
                )
                for item in loaded
            ),
            "pricing_volatility_or_rv_fallback": all(item.pricing_volatility()[0] > 0 for item in loaded),
            "csv_iv_matches_overlay": all(
                (csv_row.atm_iv_1m, csv_row.atm_iv_3m, csv_row.atm_iv_6m)
                == (memory_row.atm_iv_1m, memory_row.atm_iv_3m, memory_row.atm_iv_6m)
                for csv_row, memory_row in zip(loaded, snapshots)
            ),
        })
        gate["iv_coverage"] = iv_stats
        gate["warnings"].extend(iv_stats["warnings"])
        expiry_path = output_root / "option_expiry_points.csv"
        import io
        expiry_buffer = io.StringIO(newline="")
        expiry_fields = ["episode_id","round","underlying","source_label","version","selected_version","price_field","chain_date","expiry","dte","spot","forward","forward_ratio","mad_ratio","range_ratio","implied_vol","strikes","iv_observations","contracts"]
        expiry_writer = csv.DictWriter(expiry_buffer, fieldnames=expiry_fields, lineterminator="\n")
        expiry_writer.writeheader()
        expiry_writer.writerows(expiry_rows)
        _atomic_write(expiry_path, expiry_buffer.getvalue().encode("utf-8"))
    gate["ready"] = all(gate["checks"].values())
    _atomic_write(gate_path, (json.dumps(gate, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    manifest_paths = [snapshot_path, provenance_path, gate_path]
    if phase == "full":
        manifest_paths.append(output_root / "option_expiry_points.csv")
    manifest_lines = "".join(f"{_sha256(path)}  {path.name}\n" for path in manifest_paths)
    _atomic_write(output_root / "manifest.sha256", manifest_lines.encode("utf-8"))
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build formal MIRAGE market snapshots offline")
    parser.add_argument("--phase", choices=("base", "full"), required=True)
    args = parser.parse_args(argv)
    gate = run_base_build(phase=args.phase)
    payload = {"phase": args.phase, "ready": gate["ready"], "rows": gate["rows"], "carry": gate["carry"]}
    if args.phase == "full":
        payload["iv_coverage"] = gate["iv_coverage"]
    print(json.dumps(payload))
    return 0 if gate["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
