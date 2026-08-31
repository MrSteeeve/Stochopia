"""Offline tests for the resumable Tushare batch collector."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone

import pytest

from stochopia import tushare_batch as batch


class FakeFrame:
    def __init__(self, rows):
        self.rows = list(rows)
        self.columns = list(self.rows[0]) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def to_csv(self, index=False, lineterminator="\n"):
        assert index is False
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=self.columns, lineterminator=lineterminator)
        writer.writeheader()
        writer.writerows(self.rows)
        return output.getvalue()


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response or FakeFrame([{"trade_date": "20250102", "close": "1.0"}])
        self.error = error
        self.calls = []

    def __getattr__(self, endpoint):
        def request(**params):
            self.calls.append((endpoint, params))
            if self.error is not None:
                raise self.error
            return self.response

        return request


def test_rate_limiter_enforces_2_1_seconds_between_request_starts():
    now = [10.0]
    sleeps = []

    def clock():
        return now[0]

    def sleeper(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = batch.SerialRateLimiter(2.1, clock=clock, sleeper=sleeper)
    limiter.wait()
    now[0] += 0.5
    limiter.wait()

    assert sleeps == [pytest.approx(1.6)]


def test_fetch_writes_atomic_cache_and_cache_hit_skips_request(tmp_path):
    client = FakeClient(
        FakeFrame(
            [
                {"trade_date": "20250103", "close": "1.2"},
                {"trade_date": "20250102", "close": "1.1"},
            ]
        )
    )
    fetcher = batch.BatchFetcher(
        client,
        raw_dir=tmp_path,
        sdk_version="1.2.3",
        bridge_host="bridge.test",
        token="secret-token",
        limiter=batch.SerialRateLimiter(0),
        now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    spec = batch.RequestSpec(
        "index_daily",
        {"ts_code": "000905.SH", "start_date": "20250101", "end_date": "20251231"},
        "trade_date,close",
    )

    first = fetcher.fetch(spec)
    second = fetcher.fetch(spec)

    assert first.cached is False
    assert second.cached is True
    assert len(client.calls) == 1
    assert client.calls[0][1]["fields"] == "trade_date,close"
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["row_count"] == 2
    assert metadata["min_date"] == "20250102"
    assert metadata["max_date"] == "20250103"
    assert metadata["sha256"] == hashlib.sha256(first.csv_path.read_bytes()).hexdigest()
    assert "secret-token" not in first.csv_path.read_text(encoding="utf-8")
    assert "secret-token" not in first.metadata_path.read_text(encoding="utf-8")
    assert not list(tmp_path.rglob("*.tmp"))


def test_rate_limit_stops_collector_immediately_without_retry(tmp_path):
    client = FakeClient(error=RuntimeError("每分钟访问次数已达上限 secret-token"))
    fetcher = batch.BatchFetcher(
        client,
        raw_dir=tmp_path,
        sdk_version="1",
        bridge_host="bridge.test",
        token="secret-token",
        limiter=batch.SerialRateLimiter(0),
        retries=3,
    )
    spec = batch.RequestSpec("shibor", {"start_date": "20250101", "end_date": "20251231"})

    with pytest.raises(batch.RateLimitError) as first:
        fetcher.fetch(spec)
    with pytest.raises(batch.RateLimitError):
        fetcher.fetch(spec)

    assert "secret-token" not in str(first.value)
    assert len(client.calls) == 1


def test_annual_chunks_cover_frozen_range_without_gaps():
    assert batch.annual_chunks() == [
        ("20210901", "20211231"),
        ("20220101", "20221231"),
        ("20230101", "20231231"),
        ("20240101", "20241231"),
        ("20250101", "20251231"),
    ]


def test_core_plan_has_expected_annual_blocks_and_full_option_masters():
    specs = batch.core_plan()

    assert specs[0] == batch.RequestSpec(
        "trade_cal",
        {"exchange": "SSE", "start_date": "20210901", "end_date": "20251231"},
        "exchange,cal_date,is_open,pretrade_date",
    )
    assert len([spec for spec in specs if spec.endpoint == "index_daily"]) == 10
    assert len([spec for spec in specs if spec.endpoint == "fund_daily"]) == 5
    assert len([spec for spec in specs if spec.endpoint == "fund_adj"]) == 5
    assert len([spec for spec in specs if spec.endpoint == "fund_nav"]) == 5
    assert len([spec for spec in specs if spec.endpoint == "shibor"]) == 5
    masters = [spec for spec in specs if spec.endpoint == "opt_basic"]
    assert [spec.params["exchange"] for spec in masters] == ["SSE", "CFFEX"]
    assert len(specs) == 35
    assert [spec for spec in specs if spec.endpoint == "fut_basic"] == [
        batch.RequestSpec(
            "fut_basic",
            {"exchange": "CFFEX", "fut_type": "1"},
            "ts_code,symbol,exchange,name,fut_code,multiplier,trade_unit,per_unit,quote_unit,quote_unit_desc,d_mode_desc,list_date,delist_date,d_month,last_ddate",
        )
    ]


def test_month_end_option_dates_select_last_three_and_never_future():
    rows = [
        {"cal_date": date, "is_open": "1"}
        for date in (
            "20220831",
            "20220926",
            "20220927",
            "20220928",
            "20220929",
            "20220930",
            "20221027",
            "20221028",
            "20221031",
            "20260102",
        )
    ]
    rows.append({"cal_date": "20220925", "is_open": "0"})

    dates = batch.month_end_trading_days(rows)

    assert dates == [
        "20220928",
        "20220929",
        "20220930",
        "20221027",
        "20221028",
        "20221031",
    ]
    assert all(batch.OPTION_START_DATE <= date <= batch.END_DATE for date in dates)
    option_specs = batch.options_plan(dates)
    assert all(spec.params["exchange"] == "SSE" for spec in option_specs)
    assert [spec.params["trade_date"] for spec in option_specs] == dates


def test_manifest_hashes_cached_csv_and_metadata(tmp_path):
    fetcher = batch.BatchFetcher(
        FakeClient(),
        raw_dir=tmp_path,
        sdk_version="1",
        bridge_host="bridge.test",
        limiter=batch.SerialRateLimiter(0),
    )
    result = fetcher.fetch(batch.RequestSpec("index_daily", {"ts_code": "000905.SH"}))

    manifest = fetcher.write_manifest()
    lines = manifest.read_text(encoding="utf-8").splitlines()

    assert f"{hashlib.sha256(result.csv_path.read_bytes()).hexdigest()}  " in lines[0]
    assert any(result.metadata_path.name in line for line in lines)


def test_audit_does_not_treat_opt_code_as_underlying():
    audit = batch.audit_sse_master_candidates(
        [
            {"ts_code": "wrong", "opt_code": "510500", "name": "unrelated"},
            {"ts_code": "candidate", "opt_code": "not-underlying", "name": "南方中证500ETF购"},
        ]
    )
    assert audit["candidate_count"] == 1


def test_dry_run_does_not_require_token_or_make_requests(tmp_path, capsys):
    assert batch.main(["core", "--dry-run", "--raw-dir", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["core_requests"] == len(batch.core_plan())
    assert payload["network_requests"] == 0
