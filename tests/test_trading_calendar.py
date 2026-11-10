"""Tests for frozen exchange calendar artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from stochopia.trading_calendar import (
    build_calendar_from_tushare_raw,
    load_calendar_artifact,
    write_calendar_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_segment(
    root: Path,
    *,
    start: date,
    end: date,
    open_dates: set[date],
    pretrade_by_date: dict[date, date],
) -> tuple[Path, Path]:
    params = {
        "exchange": "SSE",
        "start_date": start.strftime("%Y%m%d"),
        "end_date": end.strftime("%Y%m%d"),
    }
    metadata = {
        "endpoint": "trade_cal",
        "params": params,
        "fields": "exchange,cal_date,is_open,pretrade_date",
        "sdk_version": "test-sdk",
        "bridge_host": "offline-test",
    }
    cache_key = f"trade_cal-{params['start_date']}-{params['end_date']}"
    directory = root / "trade_cal"
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{cache_key}.csv"
    rows = []
    current = end
    while current >= start:
        rows.append(
            {
                "exchange": "SSE",
                "cal_date": current.strftime("%Y%m%d"),
                "is_open": "1" if current in open_dates else "0",
                "pretrade_date": pretrade_by_date[current].strftime("%Y%m%d"),
            }
        )
        current -= timedelta(days=1)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("exchange", "cal_date", "is_open", "pretrade_date"),
        )
        writer.writeheader()
        writer.writerows(rows)
    metadata.update(
        {
            "status": "complete",
            "cache_key": cache_key,
            "columns": ["exchange", "cal_date", "is_open", "pretrade_date"],
            "row_count": len(rows),
            "min_date": start.strftime("%Y%m%d"),
            "max_date": end.strftime("%Y%m%d"),
            "retrieved_at": "2026-08-31T00:00:00+00:00",
            "sha256": _sha256(csv_path),
        }
    )
    sidecar_path = directory / f"{cache_key}.json"
    sidecar_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return csv_path, sidecar_path


def _raw_fixture(root: Path) -> tuple[Path, Path]:
    start, end = date(2024, 2, 7), date(2024, 2, 20)
    open_dates = {date(2024, 2, 7), date(2024, 2, 8), date(2024, 2, 19), date(2024, 2, 20)}
    previous = date(2024, 2, 6)
    pretrade: dict[date, date] = {}
    current = start
    while current <= end:
        pretrade[current] = previous
        if current in open_dates:
            previous = current
        current += timedelta(days=1)
    first = _write_segment(
        root,
        start=start,
        end=date(2024, 2, 13),
        open_dates=open_dates,
        pretrade_by_date=pretrade,
    )
    second = _write_segment(
        root,
        start=date(2024, 2, 14),
        end=end,
        open_dates=open_dates,
        pretrade_by_date=pretrade,
    )
    return first[0], second[0]


def test_builder_merges_adjacent_segments_and_preserves_real_closures(tmp_path):
    _raw_fixture(tmp_path)
    calendar = build_calendar_from_tushare_raw(tmp_path)

    assert calendar.coverage_start == date(2024, 2, 7)
    assert calendar.coverage_end == date(2024, 2, 20)
    assert calendar.open_dates == (
        date(2024, 2, 7),
        date(2024, 2, 8),
        date(2024, 2, 19),
        date(2024, 2, 20),
    )
    assert len(calendar.source_files) == 2
    assert calendar.transition_interval == "each-exchange-open-day-v1"


def test_calendar_artifact_round_trips_by_calendar_id(tmp_path):
    raw = tmp_path / "raw"
    _raw_fixture(raw)
    calendar = build_calendar_from_tushare_raw(raw)
    output = tmp_path / "derived"

    target = write_calendar_artifact(calendar, output)
    assert target.name == f"{calendar.calendar_id}.json"
    assert load_calendar_artifact(target) == calendar
    assert write_calendar_artifact(calendar, output) == target


def test_builder_fails_closed_on_broken_pretrade_chain(tmp_path):
    first_csv, _ = _raw_fixture(tmp_path)
    text = first_csv.read_text(encoding="utf-8").replace("20240208\n", "20240201\n", 1)
    first_csv.write_text(text, encoding="utf-8")
    sidecar = first_csv.with_suffix(".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["sha256"] = _sha256(first_csv)
    sidecar.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pretrade_date chain"):
        build_calendar_from_tushare_raw(tmp_path)
