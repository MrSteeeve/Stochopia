"""Resumable, rate-limited Tushare collection for Stochopia benchmark inputs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
import importlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv

from .common import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    read_csv_rows,
    read_csv_table,
    sha256_text,
)


DEFAULT_TUSHARE_HTTP_URL = "https://ts.gyzcloud.top/api"


class TushareConfigurationError(RuntimeError):
    """Raised when required Tushare configuration is missing."""


def create_pro_client(
    environ: Mapping[str, str] | None = None,
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Any:
    """Build a Tushare Pro client using environment-only credentials.

    The import is intentionally delayed until after the token check so a
    missing credential fails before SDK initialization or network activity.
    """

    env = os.environ if environ is None else environ
    token = env.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise TushareConfigurationError("TUSHARE_TOKEN is required")
    http_url = env.get("TUSHARE_HTTP_URL", DEFAULT_TUSHARE_HTTP_URL).strip()
    if not http_url:
        http_url = DEFAULT_TUSHARE_HTTP_URL

    ts = import_module("tushare")
    ts.set_token(token)
    pro = ts.pro_api()
    pro._DataApi__http_url = http_url
    return pro


START_DATE = "20210901"
END_DATE = "20251231"
OPTION_START_DATE = "20220901"
HISTORICAL_START_DATE = "20150101"
HISTORICAL_GAP_END_DATE = "20210831"
HISTORICAL_CALIBRATION_END_DATE = "20221231"
Q_HISTORY_START_DATE = "20150101"
Q_HISTORY_GAP_END_DATE = "20210831"
DEFAULT_RAW_DIR = Path("data/raw/tushare")
DEFAULT_MIN_INTERVAL = 2.1

INDEX_DAILY_FIELDS = (
    "ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount"
)
HISTORICAL_SOURCE_RECORD = {
    "provider": "Tushare Pro",
    "acquisition_route": "user-authorized Tushare bridge",
    "index_daily_documentation": "https://tushare.pro/document/1?doc_id=95",
    "data_service_agreement": "https://tushare.pro/document/1?doc_id=405",
    "license_decision_path": "data/reference/license_decision.json",
    "permitted_use": "local_research_only",
    "raw_redistribution": False,
    "history_type": "post_launch_provider_snapshot",
}
Q_HISTORY_SOURCE_RECORD = {
    "provider": "Tushare Pro",
    "acquisition_route": "user-authorized Tushare bridge",
    "shibor_documentation": "https://tushare.pro/document/1?doc_id=149",
    "data_service_agreement": "https://tushare.pro/document/1?doc_id=405",
    "license_decision_path": "data/reference/license_decision.json",
    "permitted_use": "local_research_only",
    "raw_redistribution": False,
    "history_type": "post_observation_date_provider_snapshot",
}

RATE_LIMIT_MARKERS = (
    "rate limit",
    "too many requests",
    "http 429",
    "限流",
    "冷却",
    "访问频率",
    "每分钟",
    "最多访问",
    "请求过于频繁",
    "超速",
)
DATE_COLUMNS = (
    "trade_date",
    "cal_date",
    "nav_date",
    "ex_date",
    "ann_date",
    "date",
    "list_date",
)


class BatchError(RuntimeError):
    """Base error for resumable Tushare collection."""


class RateLimitError(BatchError):
    """Raised immediately when the bridge asks the caller to cool down."""


class RequestFailed(BatchError):
    """Raised for a non-rate-limit API or serialization failure."""


@dataclass(frozen=True)
class RequestSpec:
    endpoint: str
    params: Mapping[str, Any]
    fields: str = ""
    optional: bool = False


@dataclass(frozen=True)
class CacheResult:
    endpoint: str
    cache_key: str
    csv_path: Path
    metadata_path: Path
    row_count: int
    columns: tuple[str, ...]
    cached: bool


class SerialRateLimiter:
    """Single-process request-start limiter; intentionally has no concurrency API."""

    def __init__(
        self,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        self.min_interval = min_interval
        self._clock = clock
        self._sleeper = sleeper
        self._last_started: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last_started is not None:
            delay = self.min_interval - (now - self._last_started)
            if delay > 0:
                self._sleeper(delay)
                now = self._clock()
        self._last_started = now


def annual_chunks(start_date: str = START_DATE, end_date: str = END_DATE) -> list[tuple[str, str]]:
    """Split an inclusive YYYYMMDD range at calendar-year boundaries."""

    if len(start_date) != 8 or len(end_date) != 8 or start_date > end_date:
        raise ValueError("invalid inclusive date range")
    chunks = []
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        chunks.append((max(start_date, f"{year}0101"), min(end_date, f"{year}1231")))
    return chunks


def month_end_trading_days(
    rows: Iterable[Mapping[str, str]],
    *,
    start_date: str = OPTION_START_DATE,
    end_date: str = END_DATE,
    count: int = 3,
) -> list[str]:
    """Select the last ``count`` open SSE days per month without future dates."""

    if count < 1:
        raise ValueError("count must be positive")
    grouped: dict[str, set[str]] = {}
    for row in rows:
        date = str(row.get("cal_date", ""))
        if row.get("is_open") in {1, "1", True} and start_date <= date <= end_date:
            grouped.setdefault(date[:6], set()).add(date)
    return [date for month in sorted(grouped) for date in sorted(grouped[month])[-count:]]


def core_plan() -> list[RequestSpec]:
    """Return the frozen 2021-09 through 2025-12 core request plan."""

    specs = [
        RequestSpec(
            "trade_cal",
            {"exchange": "SSE", "start_date": START_DATE, "end_date": END_DATE},
            "exchange,cal_date,is_open,pretrade_date",
        )
    ]
    for start, end in annual_chunks():
        for code in ("000905.SH", "000852.SH"):
            specs.append(
                RequestSpec(
                    "index_daily",
                    {"ts_code": code, "start_date": start, "end_date": end},
                    INDEX_DAILY_FIELDS,
                )
            )
        specs.append(
            RequestSpec(
                "fund_daily",
                {"ts_code": "510500.SH", "start_date": start, "end_date": end},
                "ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount",
            )
        )
        specs.append(
            RequestSpec(
                "fund_adj",
                {"ts_code": "510500.SH", "start_date": start, "end_date": end},
                "ts_code,trade_date,adj_factor",
            )
        )
        specs.append(
            RequestSpec(
                "fund_nav",
                {"ts_code": "510500.SH", "start_date": start, "end_date": end},
                (
                    "ts_code,ann_date,nav_date,unit_nav,accum_nav,accum_div,net_asset,"
                    "total_netasset,adj_nav"
                ),
                optional=True,
            )
        )
        specs.append(
            RequestSpec(
                "shibor",
                {"start_date": start, "end_date": end},
                "date,on,1w,2w,1m,3m,6m,9m,1y",
            )
        )
    specs.extend(
        [
            RequestSpec(
                "fund_div",
                {"ts_code": "510500.SH"},
                (
                    "ts_code,ann_date,imp_anndate,base_date,div_proc,record_date,"
                    "ex_date,pay_date,div_cash,base_unit"
                ),
            ),
            RequestSpec(
                "opt_basic",
                {"exchange": "SSE"},
                (
                    "ts_code,symbol,exchange,name,per_unit,opt_code,opt_type,call_put,"
                    "exercise_type,exercise_price,opt_multiplier,s_month,maturity_date,"
                    "list_price,list_date,delist_date,last_edate,last_ddate,quote_unit,min_price_chg"
                ),
            ),
            RequestSpec(
                "opt_basic",
                {"exchange": "CFFEX"},
                (
                    "ts_code,symbol,exchange,name,per_unit,opt_code,opt_type,call_put,"
                    "exercise_type,exercise_price,opt_multiplier,s_month,maturity_date,"
                    "list_price,list_date,delist_date,last_edate,last_ddate,quote_unit,min_price_chg"
                ),
            ),
            RequestSpec(
                "fut_basic",
                {"exchange": "CFFEX", "fut_type": "1"},
                (
                    "ts_code,symbol,exchange,name,fut_code,multiplier,trade_unit,per_unit,"
                    "quote_unit,quote_unit_desc,d_mode_desc,list_date,delist_date,d_month,last_ddate"
                ),
            ),
        ]
    )
    return specs


def historical_plan() -> list[RequestSpec]:
    """Return the frozen append-only plan for the pre-2021-09 history gap.

    The existing core cache already contains 2021-09 through 2025-12.  This
    plan deliberately stops on 2021-08-31 so collection cannot overlap or
    replace those raw files.  The 2015 start keeps both indices after their
    official launch dates and excludes pre-launch backcasts.
    """

    specs = [
        RequestSpec(
            "trade_cal",
            {
                "exchange": "SSE",
                "start_date": HISTORICAL_START_DATE,
                "end_date": HISTORICAL_GAP_END_DATE,
            },
            "exchange,cal_date,is_open,pretrade_date",
        )
    ]
    for start, end in annual_chunks(HISTORICAL_START_DATE, HISTORICAL_GAP_END_DATE):
        for code in ("000905.SH", "000852.SH"):
            specs.append(
                RequestSpec(
                    "index_daily",
                    {"ts_code": code, "start_date": start, "end_date": end},
                    INDEX_DAILY_FIELDS,
                )
            )
    return specs


def q_history_plan() -> list[RequestSpec]:
    """Backfill the non-overlapping 2015 through 2021-08 SHIBOR gap."""

    return [
        RequestSpec(
            "shibor",
            {"start_date": start, "end_date": end},
            "date,on,1w,2w,1m,3m,6m,9m,1y",
        )
        for start, end in annual_chunks(Q_HISTORY_START_DATE, Q_HISTORY_GAP_END_DATE)
    ]


def options_plan(trade_dates: Iterable[str]) -> list[RequestSpec]:
    """Build one whole-market SSE option request per selected trade date."""

    fields = (
        "ts_code,trade_date,exchange,pre_settle,pre_close,open,high,low,close,"
        "settle,vol,amount,oi"
    )
    return [
        RequestSpec("opt_daily", {"exchange": "SSE", "trade_date": date}, fields)
        for date in sorted(set(trade_dates))
        if OPTION_START_DATE <= date <= END_DATE
    ]


def pick_date_column(columns: Iterable[str], candidates: Sequence[str] = DATE_COLUMNS) -> str | None:
    """First known date column present in ``columns`` (precedence = ``candidates``)."""

    present = set(columns)
    return next((column for column in candidates if column in present), None)


def date_bounds(
    rows: Iterable[Mapping[str, Any]],
    column: str | None,
    *,
    compact: bool = False,
) -> tuple[str | None, str | None]:
    """``(min, max)`` of a date column as strings; ``compact`` strips ``-``."""

    if column is None:
        return None, None
    values = [str(row.get(column, "")) for row in rows if row.get(column)]
    if compact:
        values = [value.replace("-", "") for value in values]
    return (min(values), max(values)) if values else (None, None)


def _date_bounds(csv_text: str, columns: Sequence[str]) -> tuple[str | None, str | None]:
    return date_bounds(list(csv.DictReader(io.StringIO(csv_text))), pick_date_column(columns))


def tushare_cache_key(
    endpoint: str | None,
    params: Mapping[str, Any],
    fields: str,
    sdk_version: str,
    bridge_host: str,
) -> str:
    """Content-addressed cache key shared by the fetcher and every verifier."""

    payload = {
        "endpoint": endpoint,
        "params": dict(params),
        "fields": fields,
        "sdk_version": sdk_version,
        "bridge_host": bridge_host,
    }
    return sha256_text(canonical_json(payload))


def iter_endpoint_sidecars(root: Path, endpoint: str) -> Iterator[tuple[Path, Path]]:
    """``(metadata_path, csv_path)`` for every sidecar under ``root/endpoint``, sorted."""

    for metadata_path in sorted((root / endpoint).glob("*.json")):
        yield metadata_path, metadata_path.with_suffix(".csv")


def sidecar_checks(
    metadata: Mapping[str, Any],
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    min_date: str | None,
    max_date: str | None,
) -> dict[str, bool]:
    """Consistency checks of one cache sidecar against its parsed CSV, in protocol order."""

    return {
        "status": metadata.get("status") == "complete",
        "row_count": metadata.get("row_count") == len(rows),
        "columns": metadata.get("columns") == list(columns),
        "min_date": metadata.get("min_date") == min_date,
        "max_date": metadata.get("max_date") == max_date,
    }


def load_endpoint_rows(
    root: Path,
    endpoint: str,
    *,
    error: type[Exception] = ValueError,
    require_rows: bool = True,
) -> list[dict[str, str]]:
    """Concatenate every complete cached CSV of ``endpoint``."""

    rows: list[dict[str, str]] = []
    for metadata_path, csv_path in iter_endpoint_sidecars(root, endpoint):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("endpoint") != endpoint or metadata.get("status") != "complete":
            raise error(f"invalid {endpoint} cache metadata: {metadata_path.name}")
        if not csv_path.is_file():
            raise error(f"missing {endpoint} cache CSV: {metadata_path.name}")
        columns, cached = read_csv_table(csv_path)
        if len(cached) != metadata.get("row_count") or columns != metadata.get("columns"):
            raise error(f"invalid {endpoint} cache shape: {metadata_path.name}")
        rows.extend(cached)
    if require_rows and not rows:
        raise error(f"no cached rows for {endpoint}")
    return rows


def is_rate_limit_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in RATE_LIMIT_MARKERS)


def bridge_host(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    url = env.get("TUSHARE_HTTP_URL", DEFAULT_TUSHARE_HTTP_URL).strip()
    return urlsplit(url or DEFAULT_TUSHARE_HTTP_URL).netloc


def installed_sdk_version() -> str:
    try:
        return version("tushare")
    except PackageNotFoundError:
        return "unknown"


class BatchFetcher:
    """Fetch DataFrames serially into verified, resumable raw cache entries."""

    def __init__(
        self,
        client: Any,
        *,
        raw_dir: Path = DEFAULT_RAW_DIR,
        sdk_version: str,
        bridge_host: str,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        retries: int = 0,
        token: str = "",
        source_record: Mapping[str, Any] | None = None,
        limiter: SerialRateLimiter | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.client = client
        self.raw_dir = raw_dir
        self.sdk_version = sdk_version
        self.bridge_host = bridge_host
        self.retries = retries
        self.token = token
        self.source_record = dict(source_record or {})
        self.limiter = limiter or SerialRateLimiter(min_interval, sleeper=sleeper)
        self.sleeper = sleeper
        self.now = now
        self.stopped = False

    def cache_key(self, spec: RequestSpec) -> str:
        return tushare_cache_key(
            spec.endpoint, spec.params, spec.fields, self.sdk_version, self.bridge_host
        )

    def paths(self, spec: RequestSpec) -> tuple[Path, Path]:
        key = self.cache_key(spec)
        directory = self.raw_dir / spec.endpoint
        return directory / f"{key}.csv", directory / f"{key}.json"

    def _cached_result(self, spec: RequestSpec) -> CacheResult | None:
        key = self.cache_key(spec)
        csv_path, metadata_path = self.paths(spec)
        if not csv_path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("status") != "complete"
                or metadata.get("cache_key") != key
            ):
                return None
            return CacheResult(
                endpoint=spec.endpoint,
                cache_key=key,
                csv_path=csv_path,
                metadata_path=metadata_path,
                row_count=int(metadata["row_count"]),
                columns=tuple(metadata["columns"]),
                cached=True,
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def fetch(self, spec: RequestSpec) -> CacheResult:
        cached = self._cached_result(spec)
        if cached is not None:
            return cached
        csv_path, metadata_path = self.paths(spec)
        if self.stopped:
            raise RateLimitError("collector is stopped after a rate-limit response")

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            try:
                method = getattr(self.client, spec.endpoint)
                params = dict(spec.params)
                if spec.fields:
                    params["fields"] = spec.fields
                frame = method(**params)
                csv_text = frame.to_csv(index=False, lineterminator="\n")
                if not isinstance(csv_text, str):
                    raise TypeError("Tushare response did not serialize to CSV text")
                csv_bytes = csv_text.encode("utf-8")
                columns = tuple(str(column) for column in frame.columns)
                min_date, max_date = _date_bounds(csv_text, columns)
                csv_path, metadata_path = self.paths(spec)
                metadata = {
                    "status": "complete",
                    "cache_key": self.cache_key(spec),
                    "endpoint": spec.endpoint,
                    "params": dict(spec.params),
                    "fields": spec.fields,
                    "sdk_version": self.sdk_version,
                    "bridge_host": self.bridge_host,
                    "timestamp": self.now().isoformat(),
                    "row_count": len(frame),
                    "columns": list(columns),
                    "min_date": min_date,
                    "max_date": max_date,
                }
                if self.source_record:
                    metadata["source"] = self.source_record
                atomic_write_bytes(csv_path, csv_bytes)
                atomic_write_json(metadata_path, metadata)
                return CacheResult(
                    endpoint=spec.endpoint,
                    cache_key=self.cache_key(spec),
                    csv_path=csv_path,
                    metadata_path=metadata_path,
                    row_count=len(frame),
                    columns=columns,
                    cached=False,
                )
            except Exception as exc:
                message = str(exc)
                if self.token:
                    message = message.replace(self.token, "[REDACTED]")
                if is_rate_limit_error(message):
                    self.stopped = True
                    raise RateLimitError(
                        f"{spec.endpoint} stopped by rate limit; rerun later to resume cached work"
                    ) from None
                last_error = exc
                if attempt < self.retries:
                    self.sleeper(max(self.limiter.min_interval, 2 ** attempt))
        raise RequestFailed(
            f"{spec.endpoint} failed ({type(last_error).__name__}); completed cache was preserved"
        ) from None

    def load_cached_rows(self, spec: RequestSpec) -> list[dict[str, str]]:
        cached = self._cached_result(spec)
        if cached is None:
            raise BatchError(f"required cache missing for {spec.endpoint}")
        return read_csv_rows(cached.csv_path)

def _trade_cal_spec() -> RequestSpec:
    return core_plan()[0]


def cached_option_dates(fetcher: BatchFetcher) -> list[str]:
    rows = fetcher.load_cached_rows(_trade_cal_spec())
    return month_end_trading_days(rows)


def run_specs(fetcher: BatchFetcher, specs: Iterable[RequestSpec]) -> dict[str, Any]:
    summary: dict[str, Any] = {"planned": 0, "fetched": 0, "cached": 0, "rows": 0, "optional_failures": []}
    for spec in specs:
        summary["planned"] += 1
        try:
            result = fetcher.fetch(spec)
        except RequestFailed as exc:
            if not spec.optional:
                raise
            summary["optional_failures"].append(
                {"endpoint": spec.endpoint, "error": str(exc), "non_core_optional": True}
            )
            continue
        summary["cached" if result.cached else "fetched"] += 1
        summary["rows"] += result.row_count
    return summary


def audit_sse_master_candidates(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    """Audit-only name match; opt_code is deliberately not treated as underlying."""

    candidates = [
        str(row.get("ts_code", ""))
        for row in rows
        if "500ETF" in str(row.get("name", "")) or "中证500" in str(row.get("name", ""))
    ]
    return {"rule": "name_contains_500ETF_or_中证500", "candidate_count": len(candidates)}


def _build_fetcher(
    args: argparse.Namespace,
    *,
    source_record: Mapping[str, Any] | None = None,
) -> BatchFetcher:
    client = create_pro_client()
    return BatchFetcher(
        client,
        raw_dir=args.raw_dir,
        sdk_version=installed_sdk_version(),
        bridge_host=bridge_host(),
        min_interval=args.min_interval,
        retries=args.retries,
        token=os.environ.get("TUSHARE_TOKEN", ""),
        source_record=source_record,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stochopia resumable Tushare collector")
    parser.add_argument(
        "command",
        choices=("plan", "core", "options", "all", "historical-index", "q-history"),
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    core_specs = core_plan()
    if args.command == "q-history" and args.dry_run:
        specs = q_history_plan()
        print(
            json.dumps(
                {
                    "q_history_requests": len(specs),
                    "shibor_requests": len(specs),
                    "start_date": Q_HISTORY_START_DATE,
                    "end_date": Q_HISTORY_GAP_END_DATE,
                    "network_requests": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "historical-index" and args.dry_run:
        specs = historical_plan()
        print(
            json.dumps(
                {
                    "historical_requests": len(specs),
                    "index_daily_requests": sum(spec.endpoint == "index_daily" for spec in specs),
                    "trade_calendar_requests": sum(spec.endpoint == "trade_cal" for spec in specs),
                    "start_date": HISTORICAL_START_DATE,
                    "end_date": HISTORICAL_GAP_END_DATE,
                    "calibration_end_date": HISTORICAL_CALIBRATION_END_DATE,
                    "network_requests": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "plan" or args.dry_run:
        payload: dict[str, Any] = {"core_requests": len(core_specs), "network_requests": 0}
        if args.raw_dir.exists():
            placeholder = BatchFetcher(
                object(),
                raw_dir=args.raw_dir,
                sdk_version=installed_sdk_version(),
                bridge_host=bridge_host(),
                min_interval=args.min_interval,
            )
            try:
                dates = cached_option_dates(placeholder)
                payload.update({"option_dates": len(dates), "option_requests": len(options_plan(dates))})
            except BatchError:
                payload["options_pending_trade_cal_cache"] = True
        else:
            payload["options_pending_trade_cal_cache"] = True
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0

    historical = args.command == "historical-index"
    q_history = args.command == "q-history"
    fetcher = _build_fetcher(
        args,
        source_record=(
            HISTORICAL_SOURCE_RECORD
            if historical
            else Q_HISTORY_SOURCE_RECORD if q_history else None
        ),
    )
    result: dict[str, Any] = {}
    if historical:
        result["historical_index"] = run_specs(fetcher, historical_plan())
    if q_history:
        result["q_history"] = run_specs(fetcher, q_history_plan())
    if args.command in {"core", "all"}:
        result["core"] = run_specs(fetcher, core_specs)
    if args.command in {"options", "all"}:
        dates = cached_option_dates(fetcher)
        result["options"] = run_specs(fetcher, options_plan(dates))
        sse_master = next(
            spec for spec in core_specs if spec.endpoint == "opt_basic" and spec.params.get("exchange") == "SSE"
        )
        try:
            result["sse_contract_audit"] = audit_sse_master_candidates(fetcher.load_cached_rows(sse_master))
        except BatchError:
            result["sse_contract_audit"] = {"status": "pending_full_sse_master"}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    optional_failures = result.get("core", {}).get("optional_failures", [])
    return 2 if optional_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
