import hashlib
import json
import math
from datetime import date, datetime

import pytest

from stochopia.common import (
    as_date,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    parse_yyyymmdd,
    sha256_json,
    sha256_text,
)


def test_canonical_json_is_sorted_compact_and_utf8():
    assert canonical_json({"b": 1, "a": "中"}) == '{"a":"中","b":1}'


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


def test_canonical_json_accepts_default_encoder():
    text = canonical_json({"d": date(2024, 1, 2)}, default=str)
    assert json.loads(text) == {"d": "2024-01-02"}


def test_sha256_helpers_agree():
    payload = {"k": [1, 2, 3]}
    text = canonical_json(payload)
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert sha256_json(payload) == expected
    assert sha256_text(text) == expected


def test_as_date():
    assert as_date("2024-03-05", "d") == date(2024, 3, 5)
    assert as_date(date(2024, 3, 5), "d") == date(2024, 3, 5)
    with pytest.raises(TypeError):
        as_date(datetime(2024, 3, 5), "d")
    with pytest.raises(ValueError):
        as_date("2024/03/05", "d")
    with pytest.raises(TypeError):
        as_date(20240305, "d")


def test_atomic_writes_create_parents_and_leave_no_temp(tmp_path):
    target = tmp_path / "nested" / "out.json"
    atomic_write_json(target, {"b": 1, "a": 2})
    assert target.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "b": 1\n}\n'
    atomic_write_json(target, {"b": 1, "a": 2}, sort_keys=False)
    assert target.read_text(encoding="utf-8") == '{\n  "b": 1,\n  "a": 2\n}\n'
    atomic_write_text(target, "hello")
    assert target.read_bytes() == b"hello"
    atomic_write_bytes(target, b"\x00\x01")
    assert target.read_bytes() == b"\x00\x01"
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]


def test_atomic_write_json_rejects_nan(tmp_path):
    target = tmp_path / "out.json"
    with pytest.raises(ValueError):
        atomic_write_json(target, {"x": math.nan})
    assert not target.exists()


def test_parse_yyyymmdd():
    assert parse_yyyymmdd("20240102") == date(2024, 1, 2)
