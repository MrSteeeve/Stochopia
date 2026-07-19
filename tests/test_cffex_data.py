"""Offline tests for CFFEX archive downloading and parsing."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from mirage import cffex_data


FIXTURE = Path(__file__).parent / "fixtures" / "cffex" / "20230131_1.csv"


def _fixture_zip(path: Path, *, payload: str | None = None) -> Path:
    text = payload if payload is not None else FIXTURE.read_text(encoding="utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("20230131_1.csv", text.encode("gb18030"))
    return path


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def test_iter_months_is_inclusive_across_year_boundary():
    assert list(cffex_data.iter_months("202211", "202302")) == [
        "202211",
        "202212",
        "202301",
        "202302",
    ]
    with pytest.raises(ValueError, match="after"):
        list(cffex_data.iter_months("202302", "202301"))


def test_parser_decodes_gb18030_and_filters_ic_im_mo(tmp_path):
    archive = _fixture_zip(tmp_path / "202301.zip")

    rows = list(cffex_data.iter_archive_rows(archive))

    assert [row.product for row in rows] == ["IC", "IM", "MO"]
    assert [row.contract for row in rows] == ["IC2302", "IM2302", "MO2302-C-7000"]
    assert all(row.trade_date == "20230131" for row in rows)
    assert rows[0].settlement == 6296.0
    assert rows[0].delta is None
    assert rows[1].delta is None
    assert rows[2].open is None
    assert rows[2].delta == pytest.approx(0.4321)


def test_validate_archive_rejects_corrupt_and_empty_zip(tmp_path):
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="invalid ZIP"):
        cffex_data.validate_archive(corrupt)

    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("README.txt", "no daily data")
    with pytest.raises(ValueError, match="no YYYYMMDD_1.csv"):
        cffex_data.validate_archive(empty)


def test_download_retries_then_atomically_writes_verified_archive(tmp_path):
    source = _fixture_zip(tmp_path / "source.zip").read_bytes()
    calls: list[tuple[str, float]] = []
    sleeps: list[float] = []

    def opener(url: str, timeout: float):
        calls.append((url, timeout))
        if len(calls) == 1:
            raise OSError("temporary failure")
        return _Response(source)

    raw_dir = tmp_path / "raw"
    record = cffex_data.download_month(
        "202301",
        raw_dir=raw_dir,
        base_url="https://example.test/{yyyymm}.zip",
        retries=2,
        timeout=4.5,
        opener=opener,
        sleeper=sleeps.append,
    )

    target = raw_dir / "202301.zip"
    assert target.read_bytes() == source
    assert record.cached is False
    assert record.sha256 == hashlib.sha256(source).hexdigest()
    assert record.csv_members == 1
    assert calls == [
        ("https://example.test/202301.zip", 4.5),
        ("https://example.test/202301.zip", 4.5),
    ]
    assert sleeps == [1]
    assert list(raw_dir.glob("*.tmp")) == []

    cached = cffex_data.download_month(
        "202301", raw_dir=raw_dir, opener=lambda *_: pytest.fail("network used")
    )
    assert cached.cached is True


def test_download_range_throttles_only_between_new_network_downloads(tmp_path):
    source = _fixture_zip(tmp_path / "source.zip").read_bytes()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "202302.zip").write_bytes(source)
    calls = []
    sleeps = []
    now = [100.0]

    def opener(url: str, timeout: float):
        calls.append((url, timeout))
        return _Response(source)

    def clock():
        return now[0]

    def sleeper(seconds: float):
        sleeps.append(seconds)
        now[0] += seconds

    records = cffex_data.download_range(
        "202301",
        "202303",
        raw_dir=raw_dir,
        base_url="https://example.test/{yyyymm}.zip",
        min_interval=2.1,
        opener=opener,
        sleeper=sleeper,
        clock=clock,
    )

    assert [record.cached for record in records] == [False, True, False]
    assert [url for url, _ in calls] == [
        "https://example.test/202301.zip",
        "https://example.test/202303.zip",
    ]
    assert sleeps == [pytest.approx(2.1)]


def test_download_cli_defaults_to_single_attempt_and_2_1_second_interval():
    args = cffex_data.build_parser().parse_args(["download"])
    assert args.retries == 1
    assert args.min_interval == pytest.approx(2.1)


def test_manifest_and_normalized_csv_are_deterministic(tmp_path):
    archive = _fixture_zip(tmp_path / "202301.zip")
    digest = cffex_data.sha256_file(archive)
    record = cffex_data.ArchiveRecord(
        month="202301",
        path=str(archive),
        url="http://example.test/202301.zip",
        sha256=digest,
        size_bytes=archive.stat().st_size,
        csv_members=1,
        cached=False,
    )
    manifest = tmp_path / "raw" / "manifest.sha256"
    cffex_data.write_sha256_manifest([record], manifest)
    assert manifest.read_text(encoding="utf-8") == f"{digest}  202301.zip\n"

    output = tmp_path / "processed" / "cffex_daily.csv"
    count = cffex_data.write_normalized_csv([archive], output)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert count == 3
    assert lines[0].split(",") == list(cffex_data.NORMALIZED_COLUMNS)
    assert lines[1].startswith("20230131,IC,IC2302,")
    assert lines[3].startswith("20230131,MO,MO2302-C-7000,")


def test_cli_parse_uses_only_local_archives(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _fixture_zip(raw_dir / "202301.zip")
    output = tmp_path / "cffex.csv"

    assert cffex_data.main(
        ["parse", "--raw-dir", str(raw_dir), "--output", str(output)]
    ) == 0

    assert output.exists()
    assert '"rows": 3' in capsys.readouterr().out
