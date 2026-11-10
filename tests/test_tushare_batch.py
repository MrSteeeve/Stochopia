"""Offline tests for the resumable Tushare batch collector."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

import pytest
from types import SimpleNamespace

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
    assert "sha256" not in metadata
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


def test_historical_plan_is_disjoint_from_core_and_uses_exact_index_schema():
    """The historical extension must never silently widen the frozen core span."""

    historical = batch.historical_plan()
    core = batch.core_plan()
    expected_fields = "ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount"

    trade_cal = [spec for spec in historical if spec.endpoint == "trade_cal"]
    index_daily = [spec for spec in historical if spec.endpoint == "index_daily"]
    assert trade_cal == [
        batch.RequestSpec(
            "trade_cal",
            {"exchange": "SSE", "start_date": "20150101", "end_date": "20210831"},
            "exchange,cal_date,is_open,pretrade_date",
        )
    ]
    assert len(index_daily) == 14
    assert {spec.params["ts_code"] for spec in index_daily} == {"000905.SH", "000852.SH"}
    assert {spec.fields for spec in index_daily} == {expected_fields}
    assert all(spec.params["start_date"] >= "20150101" for spec in index_daily)
    assert all(spec.params["end_date"] <= "20210831" for spec in index_daily)
    assert all(
        not (
            historical_spec.endpoint == core_spec.endpoint
            and historical_spec.params == core_spec.params
            and historical_spec.fields == core_spec.fields
        )
        for historical_spec in historical
        for core_spec in core
    )


def test_q_history_plan_backfills_only_the_non_overlapping_shibor_gap():
    plan = batch.q_history_plan()
    assert len(plan) == 7
    assert all(spec.endpoint == "shibor" for spec in plan)
    assert plan[0].params == {"start_date": "20150101", "end_date": "20151231"}
    assert plan[-1].params == {"start_date": "20210101", "end_date": "20210831"}
    assert all(spec.params["end_date"] <= "20210831" for spec in plan)
    assert all(spec.fields == "date,on,1w,2w,1m,3m,6m,9m,1y" for spec in plan)
    assert batch.Q_HISTORY_SOURCE_RECORD["raw_redistribution"] is False




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

    assert batch.main(["q-history", "--dry-run", "--raw-dir", str(tmp_path)]) == 0
    q_payload = json.loads(capsys.readouterr().out)
    assert q_payload["q_history_requests"] == 7
    assert q_payload["network_requests"] == 0



def test_pick_date_column_and_date_bounds():
    rows = [{"trade_date": "2024-01-03", "x": "1"}, {"trade_date": "", "x": "2"}, {"trade_date": "2024-01-02", "x": "3"}]
    assert batch.pick_date_column(["x", "trade_date"]) == "trade_date"
    assert batch.pick_date_column(["x"]) is None
    assert batch.date_bounds(rows, None) == (None, None)
    assert batch.date_bounds(rows, "trade_date") == ("2024-01-02", "2024-01-03")
    assert batch.date_bounds(rows, "trade_date", compact=True) == ("20240102", "20240103")
    assert batch.date_bounds([], "trade_date") == (None, None)


def test_tushare_cache_key_is_order_independent_and_matches_fetcher(tmp_path):
    key = batch.tushare_cache_key("index_daily", {"b": "2", "a": "1"}, "f", "1.0", "host")
    assert key == batch.tushare_cache_key("index_daily", {"a": "1", "b": "2"}, "f", "1.0", "host")
    assert key != batch.tushare_cache_key("index_daily", {"a": "1", "b": "2"}, "f", "1.0", "other")


def test_load_endpoint_rows_requires_consistent_sidecars(tmp_path):
    endpoint = tmp_path / "index_daily"
    endpoint.mkdir()
    csv_path = endpoint / "k.csv"
    csv_path.write_text("trade_date,close\n20240102,1.0\n", encoding="utf-8")
    metadata = {
        "endpoint": "index_daily",
        "status": "complete",
        "row_count": 1,
        "columns": ["trade_date", "close"],
        "min_date": "20240102",
        "max_date": "20240102",
    }
    (endpoint / "k.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert batch.load_endpoint_rows(tmp_path, "index_daily") == [{"trade_date": "20240102", "close": "1.0"}]
    checks = batch.sidecar_checks(metadata, ["trade_date", "close"], [{}], "20240102", "20240102")
    assert list(checks) == ["status", "row_count", "columns", "min_date", "max_date"]
    assert all(checks.values())
    (endpoint / "k.json").write_text(json.dumps({**metadata, "columns": ["trade_date"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="cache shape"):
        batch.load_endpoint_rows(tmp_path, "index_daily")
    with pytest.raises(batch.BatchError):
        batch.load_endpoint_rows(tmp_path, "index_daily", error=batch.BatchError)
    assert batch.load_endpoint_rows(tmp_path, "missing_endpoint", require_rows=False) == []
    with pytest.raises(ValueError, match="no cached rows"):
        batch.load_endpoint_rows(tmp_path, "missing_endpoint")


# --- Tushare Pro client initialization (formerly stochopia.tushare_data) ---


def test_initialization_order_and_default_bridge_url():
    calls: list[tuple[str, str | None]] = []
    pro = SimpleNamespace()

    class FakeTushare:
        @staticmethod
        def set_token(token: str) -> None:
            calls.append(("set_token", token))

        @staticmethod
        def pro_api():
            calls.append(("pro_api", None))
            return pro

    def fake_import(name: str):
        calls.append(("import", name))
        return FakeTushare

    result = batch.create_pro_client(
        {"TUSHARE_TOKEN": "test-token"}, import_module=fake_import
    )

    assert result is pro
    assert calls == [
        ("import", "tushare"),
        ("set_token", "test-token"),
        ("pro_api", None),
    ]
    assert pro._DataApi__http_url == batch.DEFAULT_TUSHARE_HTTP_URL


def test_http_url_can_be_overridden():
    pro = SimpleNamespace()
    fake_ts = SimpleNamespace(set_token=lambda _: None, pro_api=lambda: pro)

    batch.create_pro_client(
        {
            "TUSHARE_TOKEN": "test-token",
            "TUSHARE_HTTP_URL": "https://example.test/tushare",
        },
        import_module=lambda _: fake_ts,
    )

    assert pro._DataApi__http_url == "https://example.test/tushare"


def test_missing_token_fails_before_tushare_import():
    imports: list[str] = []

    with pytest.raises(batch.TushareConfigurationError, match="TUSHARE_TOKEN"):
        batch.create_pro_client({}, import_module=lambda name: imports.append(name))

    assert imports == []
