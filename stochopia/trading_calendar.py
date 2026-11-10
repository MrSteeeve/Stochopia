"""Frozen, offline exchange-calendar artifacts for generated market paths.

The builder consumes verified local Tushare ``trade_cal`` cache entries.  The
generator consumes only the resulting ``<calendar_id>.json`` artifact and
never calls a network service at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping
from .common import as_date, atomic_write_text, canonical_json


CALENDAR_SCHEMA = "stochopia.exchange-calendar.v1"
TRANSITION_INTERVAL = "each-exchange-open-day-v1"
DEFAULT_EXCHANGE = "SSE"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_SOURCE = "Tushare Pro trade_cal"
DEFAULT_SOURCE_VERSION = "trade_cal-snapshot-v1"


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class CalendarSourceFile:
    """One raw CSV/sidecar pair contributing to the frozen calendar."""

    csv_path: str
    sidecar_path: str
    request_start: date | str
    request_end: date | str
    retrieved_at: str
    sdk_version: str
    bridge_host: str

    def __post_init__(self) -> None:
        for name in ("csv_path", "sidecar_path", "retrieved_at", "sdk_version", "bridge_host"):
            _non_empty(getattr(self, name), name)
        object.__setattr__(self, "request_start", as_date(self.request_start, "request_start"))
        object.__setattr__(self, "request_end", as_date(self.request_end, "request_end"))
        if self.request_end < self.request_start:
            raise ValueError("calendar source request_end precedes request_start")

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "sidecar_path": self.sidecar_path,
            "request_start": self.request_start.isoformat(),
            "request_end": self.request_end.isoformat(),
            "retrieved_at": self.retrieved_at,
            "sdk_version": self.sdk_version,
            "bridge_host": self.bridge_host,
        }


@dataclass(frozen=True)
class FrozenTradingCalendar:
    """Self-contained ordered open dates with source metadata."""

    calendar_id: str
    exchange: str
    timezone: str
    source: str
    source_version: str
    coverage_start: date | str
    coverage_end: date | str
    open_dates: tuple[date | str, ...]
    source_files: tuple[CalendarSourceFile, ...]
    schema: str = CALENDAR_SCHEMA
    transition_interval: str = TRANSITION_INTERVAL

    def __post_init__(self) -> None:
        if self.schema != CALENDAR_SCHEMA:
            raise ValueError(f"calendar schema must be {CALENDAR_SCHEMA!r}")
        if self.transition_interval != TRANSITION_INTERVAL:
            raise ValueError(f"transition_interval must be {TRANSITION_INTERVAL!r}")
        for name in ("calendar_id", "exchange", "timezone", "source", "source_version"):
            _non_empty(getattr(self, name), name)
        if self.calendar_id == "weekday-only-v0":
            raise ValueError("weekday-only-v0 cannot be migrated as an exchange calendar")
        coverage_start = as_date(self.coverage_start, "coverage_start")
        coverage_end = as_date(self.coverage_end, "coverage_end")
        if coverage_end < coverage_start:
            raise ValueError("coverage_end precedes coverage_start")
        object.__setattr__(self, "coverage_start", coverage_start)
        object.__setattr__(self, "coverage_end", coverage_end)
        try:
            open_dates = tuple(as_date(item, "open_dates item") for item in self.open_dates)
        except TypeError as exc:
            raise TypeError("open_dates must be iterable") from exc
        if not open_dates:
            raise ValueError("open_dates must not be empty")
        if any(right <= left for left, right in zip(open_dates, open_dates[1:])):
            raise ValueError("open_dates must be unique and strictly increasing")
        if any(item.weekday() >= 5 for item in open_dates):
            raise ValueError("SSE open_dates may not contain weekends")
        if open_dates[0] < coverage_start or open_dates[-1] > coverage_end:
            raise ValueError("open_dates fall outside calendar coverage")
        object.__setattr__(self, "open_dates", open_dates)
        try:
            source_files = tuple(self.source_files)
        except TypeError as exc:
            raise TypeError("source_files must be iterable") from exc
        if not source_files or not all(isinstance(item, CalendarSourceFile) for item in source_files):
            raise ValueError("source_files must contain CalendarSourceFile values")
        ordered = tuple(sorted(source_files, key=lambda item: item.request_start))
        if source_files != ordered:
            raise ValueError("source_files must be ordered by request_start")
        if ordered[0].request_start != coverage_start or ordered[-1].request_end != coverage_end:
            raise ValueError("source file request bounds must equal calendar coverage")
        for left, right in zip(ordered, ordered[1:]):
            if right.request_start != left.request_end + timedelta(days=1):
                raise ValueError("calendar source request ranges must be adjacent")
        object.__setattr__(self, "source_files", source_files)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "calendar_id": self.calendar_id,
            "exchange": self.exchange,
            "timezone": self.timezone,
            "source": self.source,
            "source_version": self.source_version,
            "transition_interval": self.transition_interval,
            "coverage_start": self.coverage_start.isoformat(),
            "coverage_end": self.coverage_end.isoformat(),
            "first_open_date": self.open_dates[0].isoformat(),
            "last_open_date": self.open_dates[-1].isoformat(),
            "open_date_count": len(self.open_dates),
            "open_dates": [item.isoformat() for item in self.open_dates],
            "source_files": [item.manifest for item in self.source_files],
        }

    @property
    def manifest(self) -> dict[str, Any]:
        return self.payload

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.manifest)

    def contains(self, value: date) -> bool:
        return value in set(self.open_dates)

    def open_dates_between(self, start: date, end: date) -> tuple[date, ...]:
        if start < self.coverage_start or end > self.coverage_end or end < start:
            raise ValueError("requested range falls outside calendar coverage")
        return tuple(item for item in self.open_dates if start <= item <= end)

    def month_end(self, year: int, month: int) -> date:
        candidates = [
            item for item in self.open_dates if item.year == year and item.month == month
        ]
        if not candidates:
            raise ValueError(f"calendar has no open date for {year:04d}-{month:02d}")
        if month == 12:
            natural_month_end = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            natural_month_end = date(year, month + 1, 1) - timedelta(days=1)
        if natural_month_end > self.coverage_end:
            raise ValueError("calendar coverage does not include the complete terminal month")
        return candidates[-1]

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "FrozenTradingCalendar":
        try:
            calendar = cls(
                calendar_id=payload["calendar_id"],
                exchange=payload["exchange"],
                timezone=payload["timezone"],
                source=payload["source"],
                source_version=payload["source_version"],
                coverage_start=payload["coverage_start"],
                coverage_end=payload["coverage_end"],
                open_dates=tuple(payload["open_dates"]),
                source_files=tuple(CalendarSourceFile(**item) for item in payload["source_files"]),
                schema=payload["schema"],
                transition_interval=payload["transition_interval"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("calendar artifact does not match the frozen schema") from exc
        return calendar


def load_calendar_artifact(path: str | Path) -> FrozenTradingCalendar:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calendar artifact must be a JSON object")
    return FrozenTradingCalendar.from_manifest(payload)


def build_calendar_from_tushare_raw(
    raw_root: str | Path,
    *,
    exchange: str = DEFAULT_EXCHANGE,
    calendar_id: str | None = None,
    source_version: str = DEFAULT_SOURCE_VERSION,
) -> FrozenTradingCalendar:
    """Merge all verified, adjacent local ``trade_cal`` cache segments."""

    root = Path(raw_root)
    merged: dict[date, tuple[str, str]] = {}
    sources: list[CalendarSourceFile] = []
    for sidecar_path in sorted((root / "trade_cal").glob("*.json")):
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        csv_path = sidecar_path.with_suffix(".csv")
        params = metadata.get("params", {})
        if (
            metadata.get("status") != "complete"
            or metadata.get("endpoint") != "trade_cal"
            or params.get("exchange") != exchange
            or metadata.get("fields") != "exchange,cal_date,is_open,pretrade_date"
            or metadata.get("columns") != ["exchange", "cal_date", "is_open", "pretrade_date"]
            or metadata.get("cache_key") != sidecar_path.stem
        ):
            raise ValueError(f"invalid trade_cal sidecar: {sidecar_path.name}")
        if not csv_path.is_file():
            raise ValueError(f"missing trade_cal CSV: {csv_path.name}")
        request_start = datetime.strptime(params["start_date"], "%Y%m%d").date()
        request_end = datetime.strptime(params["end_date"], "%Y%m%d").date()
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != metadata.get("row_count"):
            raise ValueError(f"trade_cal row count mismatch: {csv_path.name}")
        observed_dates: list[date] = []
        for row in rows:
            if row.get("exchange") != exchange or row.get("is_open") not in {"0", "1"}:
                raise ValueError(f"invalid trade_cal row in {csv_path.name}")
            current = datetime.strptime(row["cal_date"], "%Y%m%d").date()
            if not request_start <= current <= request_end:
                raise ValueError(f"trade_cal row outside request bounds: {csv_path.name}")
            if current in merged:
                raise ValueError(f"duplicate trade_cal date: {current.isoformat()}")
            merged[current] = (row["is_open"], row.get("pretrade_date", ""))
            observed_dates.append(current)
        if not observed_dates or min(observed_dates) != request_start or max(observed_dates) != request_end:
            raise ValueError(f"trade_cal segment does not cover its request bounds: {csv_path.name}")
        if metadata.get("min_date") != request_start.strftime("%Y%m%d") or metadata.get(
            "max_date"
        ) != request_end.strftime("%Y%m%d"):
            raise ValueError(f"trade_cal sidecar date bounds mismatch: {sidecar_path.name}")
        sources.append(
            CalendarSourceFile(
                csv_path=str(csv_path.relative_to(root)),
                sidecar_path=str(sidecar_path.relative_to(root)),
                request_start=request_start,
                request_end=request_end,
                retrieved_at=metadata.get("retrieved_at", metadata.get("timestamp", "unknown")),
                sdk_version=metadata.get("sdk_version", "unknown"),
                bridge_host=metadata.get("bridge_host", "unknown"),
            )
        )
    if not merged:
        raise ValueError(f"no verified {exchange} trade_cal cache rows")
    ordered_sources = tuple(sorted(sources, key=lambda item: item.request_start))
    coverage_start, coverage_end = ordered_sources[0].request_start, ordered_sources[-1].request_end
    expected_days = (coverage_end - coverage_start).days + 1
    if len(merged) != expected_days or set(merged) != {
        coverage_start + timedelta(days=offset) for offset in range(expected_days)
    }:
        raise ValueError("merged trade_cal rows must cover every natural date exactly once")
    previous_open: date | None = None
    for current in sorted(merged):
        is_open, pretrade_text = merged[current]
        if previous_open is not None:
            try:
                pretrade = datetime.strptime(pretrade_text, "%Y%m%d").date()
            except ValueError as exc:
                raise ValueError(
                    f"invalid pretrade_date for {current.isoformat()}"
                ) from exc
            if pretrade != previous_open:
                raise ValueError(
                    f"pretrade_date chain breaks at {current.isoformat()}"
                )
        if is_open == "1":
            previous_open = current
    open_dates = tuple(current for current in sorted(merged) if merged[current][0] == "1")
    identifier = calendar_id or (
        f"{exchange.lower()}-trade-cal-{coverage_start:%Y%m%d}-{coverage_end:%Y%m%d}-v1"
    )
    return FrozenTradingCalendar(
        calendar_id=identifier,
        exchange=exchange,
        timezone=DEFAULT_TIMEZONE,
        source=DEFAULT_SOURCE,
        source_version=source_version,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        open_dates=open_dates,
        source_files=ordered_sources,
    )


def write_calendar_artifact(
    calendar: FrozenTradingCalendar,
    output_dir: str | Path,
) -> Path:
    """Write ``<calendar_id>.json`` into ``output_dir`` and return its path."""

    if not isinstance(calendar, FrozenTradingCalendar):
        raise TypeError("calendar must be FrozenTradingCalendar")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{calendar.calendar_id}.json"
    atomic_write_text(target, calendar.canonical_json + "\n")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze a verified SSE trading calendar")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/tushare"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/derived/calendars")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    calendar = build_calendar_from_tushare_raw(args.raw_root)
    artifact = write_calendar_artifact(calendar, args.output_dir)
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "open_dates": len(calendar.open_dates),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
