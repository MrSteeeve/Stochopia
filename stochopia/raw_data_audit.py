"""Offline integrity and coverage audit for Stochopia raw market data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv

from .cffex_data import iter_archive_rows, sha256_file, validate_archive
from .tushare_batch import DATE_COLUMNS, core_plan, month_end_trading_days


DEFAULT_TUSHARE_RAW = Path("data/raw/tushare")
DEFAULT_CFFEX_RAW = Path("data/raw/cffex")
DEFAULT_CFFEX_OUTPUT = Path("data/interim/cffex/ic_im_mo_daily.csv")
DEFAULT_REPORT = Path("data/audit/raw_coverage_report.json")
SSE_510500_OPT_CODE = "OP510500.SH"
SSE_510500_SYMBOL = re.compile(r"^510500([CP])\d{4}([MA])\d{5}$")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _date_column(columns: Iterable[str]) -> str | None:
    column_set = set(columns)
    return next((column for column in DATE_COLUMNS if column in column_set), None)


def _bounds(rows: Iterable[Mapping[str, str]], date_column: str | None) -> tuple[str | None, str | None]:
    if date_column is None:
        return None, None
    values = [str(row.get(date_column, "")) for row in rows if row.get(date_column)]
    return (min(values), max(values)) if values else (None, None)


def verify_manifest(root: Path) -> dict[str, Any]:
    manifest = root / "manifest.sha256"
    if not manifest.exists():
        return {"entries": 0, "matched": 0, "missing": [], "mismatched": [], "unlisted": [], "valid": False}
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            entries[relative] = digest
    missing, mismatched = [], []
    for relative, expected in entries.items():
        path = root / relative
        if not path.exists():
            missing.append(relative)
        elif sha256_file(path) != expected:
            mismatched.append(relative)
    data_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "manifest.sha256"
        and path.suffix.lower() in {".csv", ".json", ".zip"}
    }
    unlisted = sorted(data_files - set(entries))
    matched = len(entries) - len(missing) - len(mismatched)
    return {
        "entries": len(entries),
        "matched": matched,
        "missing": sorted(missing),
        "mismatched": sorted(mismatched),
        "unlisted": unlisted,
        "valid": matched == len(entries) and not unlisted,
    }


def verify_tushare_metadata(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for metadata_path in sorted(root.glob("*/*.json")):
        relative = str(metadata_path.relative_to(root))
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            csv_path = metadata_path.with_suffix(".csv")
            columns, rows = _read_csv(csv_path)
            date_column = _date_column(columns)
            min_date, max_date = _bounds(rows, date_column)
            checks = {
                "status": metadata.get("status") == "complete",
                "sha256": metadata.get("sha256") == sha256_file(csv_path),
                "row_count": metadata.get("row_count") == len(rows),
                "columns": metadata.get("columns") == columns,
                "min_date": metadata.get("min_date") == min_date,
                "max_date": metadata.get("max_date") == max_date,
            }
            params = metadata.get("params", {})
            boundary_violations = 0
            if date_column:
                start = params.get("start_date", params.get("trade_date", "00000000"))
                end = params.get("end_date", params.get("trade_date", "99999999"))
                boundary_violations = sum(
                    not (str(start) <= str(row[date_column]) <= str(end))
                    for row in rows
                    if row.get(date_column)
                )
            if not all(checks.values()) or boundary_violations:
                failures.append(
                    {
                        "metadata": relative,
                        "failed_checks": ",".join(key for key, value in checks.items() if not value),
                        "boundary_violations": str(boundary_violations),
                    }
                )
            records.append(
                {
                    "metadata": metadata,
                    "metadata_path": metadata_path,
                    "csv_path": csv_path,
                    "columns": columns,
                    "rows": rows,
                    "date_column": date_column,
                    "boundary_violations": boundary_violations,
                }
            )
        except Exception as exc:
            failures.append({"metadata": relative, "failed_checks": type(exc).__name__, "boundary_violations": "unknown"})
    return {
        "metadata_files": len(records) + len([failure for failure in failures if failure["boundary_violations"] == "unknown"]),
        "validated": len(records) - sum(bool(record["boundary_violations"]) for record in records) - sum(1 for failure in failures if failure["boundary_violations"] != "unknown"),
        "failures": failures,
        "valid": not failures,
    }, records


def _identity(endpoint: str, params: Mapping[str, Any], fields: str) -> str:
    return json.dumps(
        {"endpoint": endpoint, "params": dict(params), "fields": fields},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def core_and_option_completeness(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    identities = {
        _identity(record["metadata"]["endpoint"], record["metadata"].get("params", {}), record["metadata"].get("fields", ""))
        for record in records
    }
    expected_core = {
        _identity(spec.endpoint, spec.params, spec.fields) for spec in core_plan()
    }
    missing_core = sorted(expected_core - identities)
    trade_cal_record = next(
        (record for record in records if record["metadata"]["endpoint"] == "trade_cal"),
        None,
    )
    expected_dates = month_end_trading_days(trade_cal_record["rows"]) if trade_cal_record else []
    option_records = [record for record in records if record["metadata"]["endpoint"] == "opt_daily"]
    actual_dates = sorted(
        str(record["metadata"].get("params", {}).get("trade_date", ""))
        for record in option_records
    )
    return {
        "expected_core": len(expected_core),
        "present_core": len(expected_core) - len(missing_core),
        "missing_core_count": len(missing_core),
        "expected_option_dates": len(expected_dates),
        "present_option_dates": len(actual_dates),
        "missing_option_dates": sorted(set(expected_dates) - set(actual_dates)),
        "extra_option_dates": sorted(set(actual_dates) - set(expected_dates)),
        "duplicate_option_dates": sorted(date for date, count in Counter(actual_dates).items() if count > 1),
        "valid": not missing_core and actual_dates == expected_dates,
    }, expected_dates


def _open_dates(records: list[dict[str, Any]]) -> set[str]:
    trade_cal = next(record for record in records if record["metadata"]["endpoint"] == "trade_cal")
    return {
        row["cal_date"] for row in trade_cal["rows"] if row.get("is_open") in {"1", 1, True}
    }


def timeseries_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    open_dates = _open_dates(records)
    report: dict[str, Any] = {}
    for endpoint in ("index_daily", "fund_daily", "fund_adj", "shibor"):
        endpoint_records = [record for record in records if record["metadata"]["endpoint"] == endpoint]
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for record in endpoint_records:
            symbol = str(record["metadata"].get("params", {}).get("ts_code", endpoint))
            groups[symbol].extend(record["rows"])
        report[endpoint] = {}
        for symbol, rows in sorted(groups.items()):
            date_column = "date" if endpoint == "shibor" else "trade_date"
            key_columns = ("ts_code", date_column) if "ts_code" in (rows[0] if rows else {}) else (date_column,)
            keys = [tuple(row.get(column, "") for column in key_columns) for row in rows]
            dates = [row[date_column] for row in rows if row.get(date_column)]
            relevant_open = {date for date in open_dates if (not dates or min(dates) <= date <= max(dates))}
            report[endpoint][symbol] = {
                "rows": len(rows),
                "min_date": min(dates) if dates else None,
                "max_date": max(dates) if dates else None,
                "duplicate_primary_keys": len(keys) - len(set(keys)),
                "missing_open_dates": sorted(relevant_open - set(dates)),
                "missing_open_date_count": len(relevant_open - set(dates)),
                "rows_by_year": dict(sorted(Counter(date[:4] for date in dates).items())),
                "request_boundary_violations": sum(record["boundary_violations"] for record in endpoint_records),
            }
    return report


def option_coverage(records: list[dict[str, Any]], expected_dates: list[str]) -> dict[str, Any]:
    masters = [record for record in records if record["metadata"]["endpoint"] == "opt_basic"]
    master_by_exchange = {
        str(record["metadata"].get("params", {}).get("exchange")): record for record in masters
    }
    sse_rows = master_by_exchange.get("SSE", {}).get("rows", [])
    cffex_rows = master_by_exchange.get("CFFEX", {}).get("rows", [])
    master_codes = {row.get("ts_code", "") for row in sse_rows}
    exact_rows = [
        row for row in sse_rows
        if row.get("exchange") == "SSE" and row.get("opt_code") == SSE_510500_OPT_CODE
    ]

    def valid_date(value: str) -> bool:
        try:
            return datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
        except (TypeError, ValueError):
            return False

    def positive(value: str) -> bool:
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            return False

    invalid_codes: set[str] = set()
    m_a_counts: Counter[str] = Counter()
    for row in exact_rows:
        symbol_match = SSE_510500_SYMBOL.fullmatch(row.get("symbol", ""))
        if symbol_match:
            m_a_counts[symbol_match.group(2)] += 1
        dates_valid = all(
            valid_date(row.get(column, ""))
            for column in ("list_date", "maturity_date", "delist_date")
        )
        dates_ordered = dates_valid and (
            row["list_date"] <= row["maturity_date"] <= row["delist_date"]
        )
        if not (
            symbol_match
            and row.get("call_put") == symbol_match.group(1)
            and positive(row.get("exercise_price", ""))
            and positive(row.get("per_unit", ""))
            and dates_ordered
        ):
            invalid_codes.add(row.get("ts_code", ""))

    whitelist_rows = [row for row in exact_rows if row.get("ts_code", "") not in invalid_codes]
    whitelist_by_code = {row.get("ts_code", ""): row for row in whitelist_rows}
    candidate_codes = set(whitelist_by_code)
    option_records = [record for record in records if record["metadata"]["endpoint"] == "opt_daily"]
    all_chain_rows = [row for record in option_records for row in record["rows"]]
    exact_codes = {row.get("ts_code", "") for row in exact_rows}
    observed_exact = [row for row in all_chain_rows if row.get("ts_code") in exact_codes]
    joined_candidates = []
    active_join_failures = 0
    for row in observed_exact:
        master = whitelist_by_code.get(row.get("ts_code", ""))
        trade_date = row.get("trade_date", "")
        if master is None:
            active_join_failures += 1
            continue
        active_end = min(
            date for date in (master.get("delist_date", ""), master.get("last_edate", ""))
            if valid_date(date)
        )
        if master["list_date"] <= trade_date <= active_end:
            joined_candidates.append(row)
        else:
            active_join_failures += 1
    candidate_dates = {row.get("trade_date", "") for row in joined_candidates}
    unmatched_codes = {row.get("ts_code", "") for row in all_chain_rows} - master_codes
    raw_prerequisite_ready = not invalid_codes and active_join_failures == 0
    return {
        "master_rows": {"SSE": len(sse_rows), "CFFEX": len(cffex_rows)},
        "sse_510500_audit_rule": "exchange=SSE; opt_code=OP510500.SH; symbol=^510500[CP]\\d{4}[MA]\\d{5}$; call_put/positive/date/active-date checks",
        "sse_510500_candidate_contracts": len(candidate_codes),
        "exact_whitelist_contracts": len(exact_rows),
        "invalid_master_contracts": len(invalid_codes),
        "observed_active_join_failures": active_join_failures,
        "m_a_counts": {key: m_a_counts.get(key, 0) for key in ("M", "A")},
        "sse_chain_rows_total": len(all_chain_rows),
        "sse_candidate_chain_rows": len(joined_candidates),
        "sse_candidate_contracts_observed": len({row.get("ts_code", "") for row in joined_candidates}),
        "candidate_dates_covered": len(candidate_dates),
        "expected_candidate_dates": len(expected_dates),
        "candidate_dates_missing": sorted(set(expected_dates) - candidate_dates),
        "daily_codes_missing_from_master": len(unmatched_codes),
        "formal_whitelist_status": "ready" if raw_prerequisite_ready else "invalid",
        "raw_prerequisite_ready": raw_prerequisite_ready,
    }


def futures_master_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    master = next(
        (
            record for record in records
            if record["metadata"]["endpoint"] == "fut_basic"
            and record["metadata"].get("params", {}) == {"exchange": "CFFEX", "fut_type": "1"}
        ),
        None,
    )
    rows = master["rows"] if master else []
    target_rows = [
        row for row in rows
        if row.get("fut_code") in {"IC", "IM"}
        and re.fullmatch(r"\d{6}", row.get("d_month", ""))
    ]
    invalid = 0
    counts: Counter[str] = Counter()
    for row in target_rows:
        product = row.get("fut_code", "")
        counts[product] += 1
        try:
            positive_terms = float(row.get("multiplier", "")) > 0 and float(row.get("per_unit", "")) > 0
            list_date = datetime.strptime(row.get("list_date", ""), "%Y%m%d")
            delist_date = datetime.strptime(row.get("delist_date", ""), "%Y%m%d")
            dates_valid = list_date <= delist_date
        except (TypeError, ValueError):
            positive_terms = dates_valid = False
        if not (
            row.get("exchange") == "CFFEX"
            and row.get("ts_code")
            and row.get("symbol")
            and positive_terms
            and dates_valid
        ):
            invalid += 1
    covered = sorted(product for product in ("IC", "IM") if counts[product])
    missing = sorted({"IC", "IM"} - set(covered))
    return {
        "master_rows": len(rows),
        "target_contracts": sum(counts.values()),
        "contracts_by_product": {product: counts[product] for product in ("IC", "IM")},
        "covered_products": covered,
        "missing_products": missing,
        "invalid_master_contracts": invalid,
        "ready": not missing and invalid == 0,
    }


def _write_cffex_interim(zips: list[Path], output: Path) -> tuple[dict[str, Any], dict[str, Counter]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    fieldnames = [
        "trade_date", "product", "contract", "open", "high", "low", "volume",
        "turnover", "open_interest", "open_interest_change", "close", "settlement",
        "previous_settlement", "change1", "change2", "delta",
    ]
    counts: dict[str, Counter] = defaultdict(Counter)
    first_dates: dict[str, str] = {}
    total = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for archive in zips:
                for row in iter_archive_rows(archive):
                    payload = row.__dict__
                    writer.writerow(payload)
                    counts[row.product][row.trade_date[:4]] += 1
                    first_dates[row.product] = min(first_dates.get(row.product, row.trade_date), row.trade_date)
                    total += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return {
        "path": str(output),
        "rows": total,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "rows_by_product_year": {
            product: dict(sorted(years.items())) for product, years in sorted(counts.items())
        },
        "first_date_by_product": dict(sorted(first_dates.items())),
        "mo_first_date": first_dates.get("MO"),
    }, counts


def audit_cffex(root: Path, output: Path) -> dict[str, Any]:
    manifest = verify_manifest(root)
    zips = sorted(root.glob("*.zip"))
    invalid, wrong_member_month, member_dates = [], [], []
    member_total = 0
    for archive_path in zips:
        try:
            member_total += validate_archive(archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                for name in archive.namelist():
                    base = Path(name).name
                    if base.endswith("_1.csv") and len(base) >= 14:
                        date = base[:8]
                        member_dates.append(date)
                        if date[:6] != archive_path.stem:
                            wrong_member_month.append(f"{archive_path.name}:{base}")
        except Exception as exc:
            invalid.append(f"{archive_path.name}:{type(exc).__name__}")
    interim, _ = _write_cffex_interim(zips, output)
    return {
        "manifest": manifest,
        "archives": len(zips),
        "crc_valid_archives": len(zips) - len(invalid),
        "invalid_archives": invalid,
        "csv_members": member_total,
        "member_min_date": min(member_dates) if member_dates else None,
        "member_max_date": max(member_dates) if member_dates else None,
        "wrong_member_month": wrong_member_month,
        "duplicate_member_dates": len(member_dates) - len(set(member_dates)),
        "interim": interim,
    }


def run_audit(
    *,
    tushare_raw: Path = DEFAULT_TUSHARE_RAW,
    cffex_raw: Path = DEFAULT_CFFEX_RAW,
    cffex_output: Path = DEFAULT_CFFEX_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    tushare_manifest = verify_manifest(tushare_raw)
    metadata_report, records = verify_tushare_metadata(tushare_raw)
    completeness, option_dates = core_and_option_completeness(records)
    series = timeseries_coverage(records)
    options = option_coverage(records, option_dates)
    futures_master = futures_master_coverage(records)
    cffex = audit_cffex(cffex_raw, cffex_output)
    gaps = []
    if not tushare_manifest["valid"]:
        gaps.append("Tushare manifest integrity failure")
    if not metadata_report["valid"]:
        gaps.append("Tushare metadata or request-boundary validation failure")
    if not completeness["valid"]:
        gaps.append("Core cache or monthly option-date coverage incomplete")
    for endpoint, symbols in series.items():
        for symbol, stats in symbols.items():
            if stats["duplicate_primary_keys"] or stats["request_boundary_violations"]:
                gaps.append(f"{endpoint}:{symbol} duplicate keys or request-boundary violations")
    if options["candidate_dates_missing"] or options["daily_codes_missing_from_master"]:
        gaps.append("SSE option chain/master exact-code join coverage incomplete")
    if not options["raw_prerequisite_ready"]:
        gaps.append("Formal SSE 510500 whitelist has invalid masters or inactive chain joins")
    if not futures_master["ready"]:
        gaps.append("CFFEX IC/IM futures master coverage incomplete or invalid")
    if (
        cffex["archives"] != 48
        or cffex["crc_valid_archives"] != 48
        or not cffex["manifest"]["valid"]
        or cffex["wrong_member_month"]
    ):
        gaps.append("CFFEX archive/hash/CRC/member-date validation failure")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contains_raw_market_values": False,
        "tushare": {
            "manifest": tushare_manifest,
            "metadata": metadata_report,
            "completeness": completeness,
            "timeseries": series,
            "options": options,
            "futures_master": futures_master,
        },
        "cffex": cffex,
        "gate_a": {"ready": not gaps, "gaps": gaps},
    }
    _atomic_write(
        report_path,
        (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Stochopia raw-data coverage audit")
    parser.add_argument("--tushare-raw", type=Path, default=DEFAULT_TUSHARE_RAW)
    parser.add_argument("--cffex-raw", type=Path, default=DEFAULT_CFFEX_RAW)
    parser.add_argument("--cffex-output", type=Path, default=DEFAULT_CFFEX_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    report = run_audit(
        tushare_raw=args.tushare_raw,
        cffex_raw=args.cffex_raw,
        cffex_output=args.cffex_output,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "gate_a_ready": report["gate_a"]["ready"],
                "gap_count": len(report["gate_a"]["gaps"]),
                "cffex_interim": str(args.cffex_output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["gate_a"]["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
