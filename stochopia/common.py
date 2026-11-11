"""Shared low-level helpers: canonical JSON, SHA-256 digests, atomic writes.

Hashes in Stochopia are identifiers, cache keys and seed sources, not tamper
evidence.  The
canonical encoding is ``sort_keys=True, separators=(",", ":"),
ensure_ascii=False, allow_nan=False``; changing it would change every recorded
identifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any


def canonical_json(payload: Any, *, default: Callable[[Any], Any] | None = None) -> str:
    """Return the one canonical JSON text used by every Stochopia hash."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=default,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any, *, default: Callable[[Any], Any] | None = None) -> str:
    """SHA-256 of the canonical JSON encoding of ``payload``."""

    return sha256_text(canonical_json(payload, default=default))


def as_date(value: object, name: str) -> date:
    """Coerce an ISO date string or ``date`` to ``date``; reject datetimes."""

    if isinstance(value, datetime):
        raise TypeError(f"{name} must be a date without a time")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO date") from exc
    raise TypeError(f"{name} must be a date or ISO date string")


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    """Durably replace ``path`` with ``content``; never leaves a partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination


def atomic_write_text(path: str | Path, content: str) -> Path:
    return atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    sort_keys: bool = True,
    indent: int | None = 2,
) -> Path:
    """Atomically write ``payload`` as human-readable JSON with a trailing newline."""

    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=sort_keys, indent=indent, allow_nan=False
    )
    return atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def normal_cdf(x: float) -> float:
    """Standard normal CDF via ``math.erf``."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def read_csv_table(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a UTF-8 CSV with a header row; returns ``(fieldnames, rows)``."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    return read_csv_table(path)[1]


def parse_yyyymmdd(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()
