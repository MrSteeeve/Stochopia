"""Offline tests for the MIRAGE Tushare initialization bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mirage.tushare_data import (
    DEFAULT_TUSHARE_HTTP_URL,
    TushareConfigurationError,
    create_pro_client,
)


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

    result = create_pro_client(
        {"TUSHARE_TOKEN": "test-token"}, import_module=fake_import
    )

    assert result is pro
    assert calls == [
        ("import", "tushare"),
        ("set_token", "test-token"),
        ("pro_api", None),
    ]
    assert pro._DataApi__http_url == DEFAULT_TUSHARE_HTTP_URL


def test_http_url_can_be_overridden():
    pro = SimpleNamespace()
    fake_ts = SimpleNamespace(set_token=lambda _: None, pro_api=lambda: pro)

    create_pro_client(
        {
            "TUSHARE_TOKEN": "test-token",
            "TUSHARE_HTTP_URL": "https://example.test/tushare",
        },
        import_module=lambda _: fake_ts,
    )

    assert pro._DataApi__http_url == "https://example.test/tushare"


def test_missing_token_fails_before_tushare_import():
    imports: list[str] = []

    with pytest.raises(TushareConfigurationError, match="TUSHARE_TOKEN"):
        create_pro_client({}, import_module=lambda name: imports.append(name))

    assert imports == []
