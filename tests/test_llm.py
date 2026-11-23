"""OpenAICompatClient 截断自动升额重试的单元测试（伪造 HTTP 层）。"""

from __future__ import annotations

import json

import pytest

import stochopia.llm as llm_module
from stochopia.llm import LLMError, LLMTruncationError, ModelConfig, OpenAICompatClient


def _response(content: str, finish_reason: str) -> "_FakeResponse":
    return _FakeResponse(
        {
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )


class _FakeResponse:
    status_code = 200

    def __init__(self, data: dict) -> None:
        self._data = data
        self.text = json.dumps(data)

    def json(self) -> dict:
        return self._data


class _FakeAsyncClient:
    """伪造 httpx.AsyncClient：记录每次请求 payload，按序弹出脚本化响应。"""

    scripted: list[_FakeResponse] = []
    payloads: list[dict] = []

    def __init__(self, timeout: float | None = None) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).payloads.append(dict(json))
        return type(self).scripted.pop(0)


@pytest.fixture
def fake_http(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "sk-test")
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.scripted = []
    _FakeAsyncClient.payloads = []
    return _FakeAsyncClient


def _client(max_tokens: int = 2000) -> OpenAICompatClient:
    return OpenAICompatClient(
        ModelConfig(
            name="fake-model",
            provider="openai-compatible",
            base_url="https://example.com/v1",
            model="fake",
            api_key_env="TEST_LLM_KEY",
            max_tokens=max_tokens,
        )
    )


async def test_truncation_retries_with_doubled_max_tokens(fake_http):
    """首次截断后自动把 max_tokens 翻倍重试，第二次成功返回内容。"""
    fake_http.scripted = [_response("被截断的内", "length"), _response("完整回复", "stop")]
    client = _client()
    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == "完整回复"
    assert [p["max_tokens"] for p in fake_http.payloads] == [2000, 4000]
    # The effective (escalated) max_tokens is recorded with the usage/trace.
    assert client.total_usage["max_tokens_escalations"] == [
        {"requested_max_tokens": 2000, "effective_max_tokens": 4000}
    ]


async def test_truncation_exhausts_retries_then_raises(fake_http):
    """连续截断：2000→4000→8000 共 3 次调用后抛 LLMTruncationError。"""
    fake_http.scripted = [_response("x", "length")] * 3
    with pytest.raises(LLMTruncationError, match="截断"):
        await _client().chat([{"role": "user", "content": "hi"}])
    assert [p["max_tokens"] for p in fake_http.payloads] == [2000, 4000, 8000]


async def test_truncation_respects_ceiling(fake_http):
    """升额封顶 TRUNCATION_MAX_TOKENS_CEILING：达到上限后不再重试。"""
    fake_http.scripted = [_response("x", "length")] * 2
    with pytest.raises(LLMTruncationError):
        await _client().chat([{"role": "user", "content": "hi"}], max_tokens=12000)
    assert [p["max_tokens"] for p in fake_http.payloads] == [12000, 16000]


async def test_truncation_error_is_llm_error(fake_http):
    """LLMTruncationError 是 LLMError 子类，上层原有 except LLMError 不受影响。"""
    assert issubclass(LLMTruncationError, LLMError)
    fake_http.scripted = [_response("ok", "stop")]
    assert await _client().chat([{"role": "user", "content": "hi"}]) == "ok"


@pytest.mark.parametrize(
    ("model", "options"),
    [
        ("gpt-6.2-sol", {"max_completion_tokens": 128, "reasoning_effort": "low"}),
        ("deepseek-flash", {"max_tokens": 128, "thinking": {"type": "disabled"}}),
    ],
)
async def test_formal_json_omits_temperature_and_sends_native_model_options(fake_http, model, options):
    client = _client()
    client.config.model = model
    client.config.temperature = None
    client.config.formal_json = True
    fake_http.scripted = [_FakeResponse({**_response('{"ok":true}', "stop").json(), "model": model})]
    messages = [{"role": "user", "content": "Return JSON."}]

    assert await client.chat(messages, max_tokens=128) == '{"ok":true}'
    assert fake_http.payloads == [
        {"model": model, "messages": messages, "response_format": {"type": "json_object"}, **options}
    ]
    assert client.total_usage["api_attempts"] == 1
    assert client.total_usage["api_seconds"] >= 0
    assert client.total_usage["prompt_tokens"] == 10
    assert client.total_usage["completion_tokens"] == 5


@pytest.mark.parametrize("model_field", [{"model": "unexpected-model"}, {}])
async def test_formal_json_model_mismatch_latches_after_recording_usage(fake_http, model_field):
    client = _client()
    client.config.model = "gpt-6.2-sol"
    client.config.temperature = None
    client.config.formal_json = True
    fake_http.scripted = [_FakeResponse({**_response('{"ok":true}', "stop").json(), **model_field})]
    messages = [{"role": "user", "content": "Return JSON."}]

    with pytest.raises(LLMError, match="不匹配"):
        await client.chat(messages)
    with pytest.raises(LLMError, match="已停止"):
        await client.chat(messages)

    assert client.unexpected_model == model_field.get("model", "")
    assert len(fake_http.payloads) == 1
    assert client.total_usage["api_attempts"] == client.total_usage["calls"] == 1
    assert client.total_usage["prompt_tokens"] == 10
    assert client.total_usage["completion_tokens"] == 5


class _FlakyThenOkClient(_FakeAsyncClient):
    """前 N 次 post 抛出瞬时网络异常，之后按脚本返回。"""

    failures_left = 0
    exc_factory = None

    async def post(self, url, json=None, headers=None):
        if _FlakyThenOkClient.failures_left > 0:
            _FlakyThenOkClient.failures_left -= 1
            raise _FlakyThenOkClient.exc_factory()
        return await super().post(url, json=json, headers=headers)


@pytest.fixture
def flaky_http(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "sk-test")
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", _FlakyThenOkClient)
    monkeypatch.setattr(OpenAICompatClient, "RETRY_DELAYS", (0.0, 0.0, 0.0))
    _FlakyThenOkClient.scripted = []
    _FlakyThenOkClient.payloads = []
    return _FlakyThenOkClient


async def test_remote_protocol_error_is_retried(flaky_http):
    """连接被服务端/代理掐断（RemoteProtocolError）应重试而非直接失败。"""
    flaky_http.failures_left = 2
    flaky_http.exc_factory = lambda: llm_module.httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body"
    )
    flaky_http.scripted = [_response("恢复后的回复", "stop")]
    result = await _client().chat([{"role": "user", "content": "hi"}])
    assert result == "恢复后的回复"


async def test_persistent_transport_error_raises_after_retries(flaky_http):
    """传输层故障持续不恢复：耗尽重试后抛 LLMError，报错含原因。"""
    flaky_http.failures_left = 99
    flaky_http.exc_factory = lambda: llm_module.httpx.RemoteProtocolError("incomplete chunked read")
    with pytest.raises(LLMError, match="网络传输错误"):
        await _client().chat([{"role": "user", "content": "hi"}])
