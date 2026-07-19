"""Offline tests for raw-data integrity and coverage auditing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mirage import raw_data_audit as audit


def _write_manifest(root: Path, paths: list[Path]) -> None:
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
        for path in paths
    ]
    (root / "manifest.sha256").write_text("".join(lines), encoding="utf-8")


def test_verify_manifest_detects_hash_mismatch_and_unlisted_file(tmp_path):
    listed = tmp_path / "endpoint" / "listed.csv"
    listed.parent.mkdir()
    listed.write_text("date,value\n20250101,1\n", encoding="utf-8")
    _write_manifest(tmp_path, [listed])
    (tmp_path / ".DS_Store").write_bytes(b"local metadata")

    assert audit.verify_manifest(tmp_path)["valid"] is True

    listed.write_text("tampered", encoding="utf-8")
    extra = listed.parent / "extra.json"
    extra.write_text("{}", encoding="utf-8")
    result = audit.verify_manifest(tmp_path)
    assert result["valid"] is False
    assert result["mismatched"] == ["endpoint/listed.csv"]
    assert result["unlisted"] == ["endpoint/extra.json"]


def test_verify_metadata_checks_csv_hash_shape_and_date_bounds(tmp_path):
    directory = tmp_path / "index_daily"
    directory.mkdir()
    csv_path = directory / "key.csv"
    csv_path.write_text(
        "ts_code,trade_date,close\n000905.SH,20250103,1\n000905.SH,20250102,2\n",
        encoding="utf-8",
    )
    metadata = {
        "status": "complete",
        "endpoint": "index_daily",
        "params": {"ts_code": "000905.SH", "start_date": "20250101", "end_date": "20250131"},
        "fields": "ts_code,trade_date,close",
        "row_count": 2,
        "columns": ["ts_code", "trade_date", "close"],
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "min_date": "20250102",
        "max_date": "20250103",
    }
    (directory / "key.json").write_text(json.dumps(metadata), encoding="utf-8")

    report, records = audit.verify_tushare_metadata(tmp_path)

    assert report["valid"] is True
    assert report["validated"] == 1
    assert records[0]["boundary_violations"] == 0


def test_timeseries_coverage_reports_duplicates_missing_open_dates_and_year_counts():
    records = [
        {
            "metadata": {"endpoint": "trade_cal", "params": {}},
            "rows": [
                {"cal_date": "20250102", "is_open": "1"},
                {"cal_date": "20250103", "is_open": "1"},
                {"cal_date": "20250106", "is_open": "1"},
            ],
            "boundary_violations": 0,
        },
        {
            "metadata": {"endpoint": "index_daily", "params": {"ts_code": "000905.SH"}},
            "rows": [
                {"ts_code": "000905.SH", "trade_date": "20250102"},
                {"ts_code": "000905.SH", "trade_date": "20250102"},
                {"ts_code": "000905.SH", "trade_date": "20250106"},
            ],
            "boundary_violations": 0,
        },
    ]

    result = audit.timeseries_coverage(records)["index_daily"]["000905.SH"]

    assert result["duplicate_primary_keys"] == 1
    assert result["missing_open_dates"] == ["20250103"]
    assert result["rows_by_year"] == {"2025": 3}


def test_option_coverage_uses_exact_510500_whitelist_and_active_dates():
    master = {
        "ts_code": "10000001.SH", "symbol": "510500C2506M05000",
        "exchange": "SSE", "opt_code": "OP510500.SH", "call_put": "C",
        "exercise_price": "5", "per_unit": "10000", "list_date": "20250101",
        "maturity_date": "20250625", "delist_date": "20250625", "last_edate": "20250625",
    }
    adjusted = {**master, "ts_code": "10000002.SH", "symbol": "510500P2506A05000", "call_put": "P"}
    records = [
        {"metadata": {"endpoint": "opt_basic", "params": {"exchange": "SSE"}}, "rows": [master, adjusted]},
        {"metadata": {"endpoint": "opt_basic", "params": {"exchange": "CFFEX"}}, "rows": []},
        {"metadata": {"endpoint": "opt_daily"}, "rows": [
            {"ts_code": "10000001.SH", "trade_date": "20250131"},
            {"ts_code": "10000002.SH", "trade_date": "20250131"},
        ]},
    ]

    result = audit.option_coverage(records, ["20250131"])

    assert result["exact_whitelist_contracts"] == 2
    assert result["invalid_master_contracts"] == 0
    assert result["observed_active_join_failures"] == 0
    assert result["m_a_counts"] == {"M": 1, "A": 1}
    assert result["raw_prerequisite_ready"] is True


def test_option_coverage_rejects_bad_master_and_inactive_observation():
    invalid = {
        "ts_code": "10000001.SH", "symbol": "510500C2506M05000",
        "exchange": "SSE", "opt_code": "OP510500.SH", "call_put": "P",
        "exercise_price": "0", "per_unit": "10000", "list_date": "20250101",
        "maturity_date": "20250625", "delist_date": "20250625", "last_edate": "20250625",
    }
    records = [
        {"metadata": {"endpoint": "opt_basic", "params": {"exchange": "SSE"}}, "rows": [invalid]},
        {"metadata": {"endpoint": "opt_basic", "params": {"exchange": "CFFEX"}}, "rows": []},
        {"metadata": {"endpoint": "opt_daily"}, "rows": [{"ts_code": "10000001.SH", "trade_date": "20250131"}]},
    ]

    result = audit.option_coverage(records, ["20250131"])

    assert result["invalid_master_contracts"] == 1
    assert result["observed_active_join_failures"] == 1
    assert result["raw_prerequisite_ready"] is False


def test_futures_master_coverage_requires_valid_ic_and_im_contracts():
    def contract(product):
        return {
            "ts_code": f"{product}2506.CFX", "symbol": f"{product}2506",
            "exchange": "CFFEX", "fut_code": product, "multiplier": "200",
            "per_unit": "1", "list_date": "20250101", "delist_date": "20250620",
            "d_month": "202506",
        }

    result = audit.futures_master_coverage([
        {
            "metadata": {"endpoint": "fut_basic", "params": {"exchange": "CFFEX", "fut_type": "1"}},
            "rows": [contract("IC"), contract("IM")],
        }
    ])

    assert result["contracts_by_product"] == {"IC": 1, "IM": 1}
    assert result["invalid_master_contracts"] == 0
    assert result["ready"] is True


def test_atomic_write_leaves_no_temp_file(tmp_path):
    target = tmp_path / "audit" / "report.json"
    audit._atomic_write(target, b'{"ok": true}\n')
    assert target.read_bytes() == b'{"ok": true}\n'
    assert not list(target.parent.glob("*.tmp"))
