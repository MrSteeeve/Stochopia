"""Download and parse CFFEX monthly daily-market archives.

The public archive contains one GB18030-encoded CSV per trading day.  This
module keeps the downloaded ZIP files immutable, records their SHA256 hashes,
and emits a normalized CSV containing only IC, IM, and MO contracts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, ContextManager


DEFAULT_BASE_URL = (
    "http://www.cffex.com.cn/sj/historysj/{yyyymm}/zip/{yyyymm}.zip"
)
DEFAULT_RAW_DIR = Path("data/raw/cffex")
DEFAULT_PRODUCTS = ("IC", "IM", "MO")
DEFAULT_START_MONTH = "202201"
DEFAULT_END_MONTH = "202512"
DEFAULT_MIN_INTERVAL = 2.1
USER_AGENT = "Stochopia-CFFEX-downloader/0.1"

_MONTH_RE = re.compile(r"^(?P<year>\d{4})(?P<month>\d{2})$")
_DAILY_MEMBER_RE = re.compile(r"(?P<date>\d{8})_1\.csv$", re.IGNORECASE)
_PRODUCT_RE = re.compile(r"^(?P<product>[A-Za-z]+)")

SOURCE_COLUMNS = {
    "合约代码": "contract",
    "今开盘": "open",
    "最高价": "high",
    "最低价": "low",
    "成交量": "volume",
    "成交金额": "turnover",
    "持仓量": "open_interest",
    "持仓变化": "open_interest_change",
    "今收盘": "close",
    "今结算": "settlement",
    "前结算": "previous_settlement",
    "涨跌1": "change1",
    "涨跌2": "change2",
    "Delta": "delta",
}

NORMALIZED_COLUMNS = (
    "trade_date",
    "product",
    "contract",
    "open",
    "high",
    "low",
    "volume",
    "turnover",
    "open_interest",
    "open_interest_change",
    "close",
    "settlement",
    "previous_settlement",
    "change1",
    "change2",
    "delta",
)


@dataclass(frozen=True)
class CffexDailyRow:
    """One normalized contract-level daily observation."""

    trade_date: str
    product: str
    contract: str
    open: float | None
    high: float | None
    low: float | None
    volume: float | None
    turnover: float | None
    open_interest: float | None
    open_interest_change: float | None
    close: float | None
    settlement: float | None
    previous_settlement: float | None
    change1: float | None
    change2: float | None
    delta: float | None


@dataclass(frozen=True)
class ArchiveRecord:
    """Metadata recorded for one verified monthly archive."""

    month: str
    path: str
    url: str
    sha256: str
    size_bytes: int
    csv_members: int
    cached: bool


def validate_month(value: str) -> str:
    """Return a validated YYYYMM string."""

    match = _MONTH_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"month must use YYYYMM format: {value!r}")
    month = int(match.group("month"))
    if not 1 <= month <= 12:
        raise ValueError(f"month must be between 01 and 12: {value!r}")
    return value


def iter_months(start: str, end: str) -> Iterator[str]:
    """Yield inclusive YYYYMM values in chronological order."""

    start = validate_month(start)
    end = validate_month(end)
    if start > end:
        raise ValueError(f"start month {start} is after end month {end}")
    year, month = int(start[:4]), int(start[4:])
    end_year, end_month = int(end[:4]), int(end[4:])
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}{month:02d}"
        month += 1
        if month == 13:
            year += 1
            month = 1


def sha256_file(path: Path) -> str:
    """Compute a file SHA256 without loading the whole archive into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path) -> int:
    """Validate ZIP structure and contents; return the number of daily CSVs."""

    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"CRC check failed for {bad_member} in {path}")
            csv_members = [
                info
                for info in archive.infolist()
                if not info.is_dir() and _DAILY_MEMBER_RE.search(info.filename)
            ]
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid ZIP archive: {path}") from exc
    if not csv_members:
        raise ValueError(f"archive contains no YYYYMMDD_1.csv members: {path}")
    return len(csv_members)


def _default_opener(url: str, timeout: float) -> ContextManager[BinaryIO]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)


def _archive_record(month: str, path: Path, url: str, cached: bool) -> ArchiveRecord:
    return ArchiveRecord(
        month=month,
        path=str(path),
        url=url,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        csv_members=validate_archive(path),
        cached=cached,
    )


def download_month(
    month: str,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    base_url: str = DEFAULT_BASE_URL,
    retries: int = 3,
    timeout: float = 30.0,
    opener: Callable[[str, float], ContextManager[BinaryIO]] = _default_opener,
    sleeper: Callable[[float], None] = time.sleep,
) -> ArchiveRecord:
    """Download one month with retries, ZIP validation, and atomic replacement."""

    month = validate_month(month)
    if retries < 1:
        raise ValueError("retries must be at least 1")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{month}.zip"
    url = base_url.format(yyyymm=month)

    if target.exists():
        try:
            return _archive_record(month, target, url, cached=True)
        except ValueError:
            # Preserve the suspect file until a verified replacement is ready.
            pass

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=raw_dir,
                prefix=f".{month}.",
                suffix=".tmp",
                delete=False,
            ) as temp:
                temp_path = Path(temp.name)
                with opener(url, timeout) as response:
                    while chunk := response.read(1024 * 1024):
                        temp.write(chunk)
                temp.flush()
                os.fsync(temp.fileno())
            validate_archive(temp_path)
            os.replace(temp_path, target)
            temp_path = None
            return _archive_record(month, target, url, cached=False)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            last_error = exc
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            if attempt < retries:
                sleeper(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"failed to download verified CFFEX archive {month} after {retries} attempts"
    ) from last_error


def _has_valid_cached_archive(month: str, raw_dir: Path) -> bool:
    target = raw_dir / f"{validate_month(month)}.zip"
    if not target.exists():
        return False
    try:
        validate_archive(target)
    except ValueError:
        return False
    return True


def download_range(
    start: str,
    end: str,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    base_url: str = DEFAULT_BASE_URL,
    retries: int = 1,
    timeout: float = 30.0,
    min_interval: float = DEFAULT_MIN_INTERVAL,
    opener: Callable[[str, float], ContextManager[BinaryIO]] = _default_opener,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[ArchiveRecord]:
    """Download an inclusive month range with cache-aware network throttling."""

    if min_interval < 0:
        raise ValueError("min_interval must be non-negative")
    records: list[ArchiveRecord] = []
    last_new_download_completed: float | None = None
    for month in iter_months(start, end):
        needs_network = not _has_valid_cached_archive(month, raw_dir)
        if needs_network and last_new_download_completed is not None:
            delay = min_interval - (clock() - last_new_download_completed)
            if delay > 0:
                sleeper(delay)
        record = download_month(
            month,
            raw_dir=raw_dir,
            base_url=base_url,
            retries=retries,
            timeout=timeout,
            opener=opener,
            sleeper=sleeper,
        )
        records.append(record)
        if not record.cached:
            last_new_download_completed = clock()
    return records


def write_sha256_manifest(records: Iterable[ArchiveRecord], path: Path) -> None:
    """Atomically write a conventional SHA256 manifest sorted by month."""

    rows = sorted(records, key=lambda record: record.month)
    content = "".join(
        f"{record.sha256}  {Path(record.path).name}\n" for record in rows
    )
    _atomic_write_text(path, content)


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip().replace(",", "")
    if normalized in {"", "--", "-", "—", "N/A", "n/a", "null", "NULL"}:
        return None
    return float(normalized)


def _daily_members(archive: zipfile.ZipFile) -> list[tuple[str, zipfile.ZipInfo]]:
    members: list[tuple[str, zipfile.ZipInfo]] = []
    for info in archive.infolist():
        match = _DAILY_MEMBER_RE.search(info.filename)
        if not info.is_dir() and match is not None:
            members.append((match.group("date"), info))
    return sorted(members, key=lambda item: (item[0], item[1].filename))


def iter_archive_rows(
    path: Path, products: Iterable[str] = DEFAULT_PRODUCTS
) -> Iterator[CffexDailyRow]:
    """Yield normalized IC/IM/MO rows from one verified monthly archive."""

    allowed = {product.upper() for product in products}
    if not allowed:
        raise ValueError("at least one product is required")
    validate_archive(path)
    with zipfile.ZipFile(path) as archive:
        for trade_date, member in _daily_members(archive):
            with archive.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="gb18030", newline="")
                reader = csv.DictReader(text)
                if reader.fieldnames is None:
                    raise ValueError(f"missing CSV header in {member.filename}")
                fieldnames = {name.strip() for name in reader.fieldnames}
                missing = set(SOURCE_COLUMNS) - fieldnames
                if missing:
                    raise ValueError(
                        f"missing columns in {member.filename}: {sorted(missing)}"
                    )
                for source_row in reader:
                    cleaned = {
                        (key.strip() if key is not None else ""): (
                            value.strip() if value is not None else ""
                        )
                        for key, value in source_row.items()
                    }
                    contract = cleaned["合约代码"]
                    product_match = _PRODUCT_RE.match(contract)
                    if product_match is None:
                        continue
                    product = product_match.group("product").upper()
                    if product not in allowed:
                        continue
                    yield CffexDailyRow(
                        trade_date=trade_date,
                        product=product,
                        contract=contract,
                        open=_parse_number(cleaned["今开盘"]),
                        high=_parse_number(cleaned["最高价"]),
                        low=_parse_number(cleaned["最低价"]),
                        volume=_parse_number(cleaned["成交量"]),
                        turnover=_parse_number(cleaned["成交金额"]),
                        open_interest=_parse_number(cleaned["持仓量"]),
                        open_interest_change=_parse_number(cleaned["持仓变化"]),
                        close=_parse_number(cleaned["今收盘"]),
                        settlement=_parse_number(cleaned["今结算"]),
                        previous_settlement=_parse_number(cleaned["前结算"]),
                        change1=_parse_number(cleaned["涨跌1"]),
                        change2=_parse_number(cleaned["涨跌2"]),
                        delta=_parse_number(cleaned["Delta"]),
                    )


def write_normalized_csv(
    archives: Iterable[Path],
    output: Path,
    products: Iterable[str] = DEFAULT_PRODUCTS,
) -> int:
    """Atomically write normalized rows from sorted archives; return row count."""

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    row_count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=NORMALIZED_COLUMNS)
            writer.writeheader()
            for archive in sorted(archives):
                for row in iter_archive_rows(archive, products):
                    writer.writerow(asdict(row))
                    row_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return row_count


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CFFEX IC/IM/MO archive utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download", help="download verified monthly CFFEX archives"
    )
    download.add_argument("--start", default=DEFAULT_START_MONTH, type=validate_month)
    download.add_argument("--end", default=DEFAULT_END_MONTH, type=validate_month)
    download.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    download.add_argument("--base-url", default=DEFAULT_BASE_URL)
    download.add_argument("--retries", type=int, default=1)
    download.add_argument("--timeout", type=float, default=30.0)
    download.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL)
    download.add_argument("--manifest", type=Path)

    parse = subparsers.add_parser(
        "parse", help="parse downloaded archives into one normalized CSV"
    )
    parse.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parse.add_argument("--output", type=Path, required=True)
    parse.add_argument(
        "--products", nargs="+", choices=DEFAULT_PRODUCTS, default=list(DEFAULT_PRODUCTS)
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        records = download_range(
            args.start,
            args.end,
            raw_dir=args.raw_dir,
            base_url=args.base_url,
            retries=args.retries,
            timeout=args.timeout,
            min_interval=args.min_interval,
        )
        manifest = args.manifest or args.raw_dir / "manifest.sha256"
        write_sha256_manifest(records, manifest)
        print(
            json.dumps(
                {
                    "archives": len(records),
                    "cached": sum(record.cached for record in records),
                    "manifest": str(manifest),
                },
                ensure_ascii=False,
            )
        )
        return 0

    archives = sorted(args.raw_dir.glob("*.zip"))
    if not archives:
        raise SystemExit(f"no ZIP archives found in {args.raw_dir}")
    row_count = write_normalized_csv(archives, args.output, args.products)
    print(
        json.dumps(
            {"archives": len(archives), "rows": row_count, "output": str(args.output)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
