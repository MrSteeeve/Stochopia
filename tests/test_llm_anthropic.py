"""AnthropicClient 的最小单测（伪造 HTTP 层）：验证 messages 格式转换、system
角色抽取、鉴权头，以及响应解析。"""

from __future__ import annotations

import json

import pytest

import stochopia.llm as llm_module
from stochopia.llm import AnthropicClient, LLMError, ModelConfig


def _response(text: str, stop_reason: str = "end_turn") -> "_FakeResponse":
    return _FakeResponse(
        {
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 12, "output_tokens": 6},
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
    """伪造 httpx.AsyncClient：记录每次请求的 url/payload/headers，按序弹出脚本化响应。"""

    scripted: list[_FakeResponse] = []
    payloads: list[dict] = []
    headers_seen: list[dict] = []
    urls: list[str] = []

    def __init__(self, timeout: float | None = None) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).urls.append(url)
        type(self).payloads.append(dict(json))
        type(self).headers_seen.append(dict(headers or {}))
        return type(self).scripted.pop(0)


@pytest.fixture
def fake_http(monkeypatch):
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-test")
    monkeypatch.setattr(llm_module.httpx, "AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.scripted = []
    _FakeAsyncClient.payloads = []
    _FakeAsyncClient.headers_seen = []
    _FakeAsyncClient.urls = []
    return _FakeAsyncClient


def _client(max_tokens: int = 2000) -> AnthropicClient:
    return AnthropicClient(
        ModelConfig(
            name="fake-claude",
            provider="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-5",
            api_key_env="TEST_ANTHROPIC_KEY",
            max_tokens=max_tokens,
        )
    )


async def test_posts_to_v1_messages_endpoint(fake_http):
    """请求打 {base_url}/v1/messages，而非 OpenAI 风格的 /chat/completions。"""
    fake_http.scripted = [_response("你好")]
    await _client().chat([{"role": "user", "content": "hi"}])
    assert fake_http.urls == ["https://api.anthropic.com/v1/messages"]


async def test_auth_headers_use_x_api_key(fake_http):
    """鉴权走 x-api-key + anthropic-version，而非 Authorization: Bearer。"""
    fake_http.scripted = [_response("你好")]
    await _client().chat([{"role": "user", "content": "hi"}])
    headers = fake_http.headers_seen[0]
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in headers


async def test_system_role_extracted_to_top_level_field(fake_http):
    """messages 里的 system 角色被抽出拼进顶层 system 字段，不出现在 messages 数组里。"""
    fake_http.scripted = [_response("好的")]
    await _client().chat(
        [
            {"role": "system", "content": "你是结构化产品设计师"},
            {"role": "user", "content": "设计一个产品"},
        ]
    )
    payload = fake_http.payloads[0]
    assert payload["system"] == "你是结构化产品设计师"
    assert payload["messages"] == [{"role": "user", "content": "设计一个产品"}]
    assert all(m.get("role") != "system" for m in payload["messages"])


async def test_multiple_system_messages_joined(fake_http):
    """多条 system 消息拼接为一个字符串（用空行分隔）。"""
    fake_http.scripted = [_response("好的")]
    await _client().chat(
        [
            {"role": "system", "content": "第一条系统指令"},
            {"role": "system", "content": "第二条系统指令"},
            {"role": "user", "content": "继续"},
        ]
    )
    payload = fake_http.payloads[0]
    assert payload["system"] == "第一条系统指令\n\n第二条系统指令"


async def test_no_system_field_when_no_system_message(fake_http):
    """没有 system 角色消息时，payload 里不应出现 system 字段。"""
    fake_http.scripted = [_response("好的")]
    await _client().chat([{"role": "user", "content": "继续"}])
    payload = fake_http.payloads[0]
    assert "system" not in payload


async def test_parses_content_first_text_block(fake_http):
    """响应从 content[0].text 取正文。"""
    fake_http.scripted = [_response("这是回复正文")]
    result = await _client().chat([{"role": "user", "content": "hi"}])
    assert result == "这是回复正文"


async def test_usage_accumulated_from_input_output_tokens(fake_http):
    """usage 累加读取 input_tokens / output_tokens（Anthropic 字段名，非 OpenAI 的 prompt/completion）。"""
    fake_http.scripted = [_response("ok")]
    client = _client()
    await client.chat([{"role": "user", "content": "hi"}])
    assert client.total_usage["prompt_tokens"] == 12
    assert client.total_usage["completion_tokens"] == 6
    assert client.total_usage["calls"] == 1


async def test_missing_api_key_raises_llm_error(fake_http, monkeypatch):
    """未设置对应环境变量时应报清晰的 LLMError，而不是走到网络请求。"""
    monkeypatch.delenv("TEST_ANTHROPIC_KEY", raising=False)
    with pytest.raises(LLMError, match="TEST_ANTHROPIC_KEY"):
        await _client().chat([{"role": "user", "content": "hi"}])
    assert fake_http.payloads == []
