"""LLM 客户端封装：模型注册表、OpenAI 兼容客户端、Mock 客户端。"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml


class LLMError(Exception):
    """LLM 调用相关的统一异常，报错信息应带足上下文。"""


class LLMTruncationError(LLMError):
    """输出因 max_tokens 上限被截断；chat 内部会自动升额重试，重试耗尽才抛出。"""


@dataclass
class ModelConfig:
    """模型注册表中的一条配置。"""

    name: str
    provider: str
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: float = 60.0


class ModelRegistry(dict):
    """模型注册表：dict[模型名 -> ModelConfig]。"""


def load_model_registry(path: str | Path) -> ModelRegistry:
    """解析 config/models.yaml；文件缺失或格式错误抛 LLMError。"""
    p = Path(path)
    if not p.is_file():
        raise LLMError(f"模型注册表文件不存在：{p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LLMError(f"模型注册表 YAML 解析失败（{p}）：{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise LLMError(f"模型注册表格式错误（{p}）：顶层需要 models 映射")
    registry: dict[str, ModelConfig] = {}
    for name, cfg in data["models"].items():
        if not isinstance(cfg, dict) or "provider" not in cfg:
            raise LLMError(f"模型 {name} 配置格式错误（{p}）：需要 provider 字段")
        allowed = {"provider", "base_url", "model", "api_key_env", "temperature", "max_tokens", "timeout"}
        unknown = set(cfg) - allowed
        if unknown:
            raise LLMError(f"模型 {name} 配置含未知字段（{p}）：{sorted(unknown)}")
        registry[name] = ModelConfig(name=name, **cfg)
    return ModelRegistry(registry)


class BaseLLMClient:
    """LLM 客户端基类，统计 token 用量。"""

    def __init__(self) -> None:
        self.total_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        raise NotImplementedError

    def _record_max_tokens_escalation(self, requested: int, effective: int) -> None:
        """截断升额重试后成功：把实际生效的 max_tokens 记入 usage（随 trace 序列化）。"""
        self.total_usage.setdefault("max_tokens_escalations", []).append(
            {"requested_max_tokens": requested, "effective_max_tokens": effective}
        )


class OpenAICompatClient(BaseLLMClient):
    """OpenAI 兼容 chat/completions 客户端，429/5xx/超时指数退避重试。

    输出被 max_tokens 截断（finish_reason=length）时不直接报错，而是把
    max_tokens 翻倍后立即重试（独立计数，最多 TRUNCATION_RETRIES 次，
    封顶 TRUNCATION_MAX_TOKENS_CEILING），仍截断才抛 LLMTruncationError；升额后成功时
    实际生效的 max_tokens 记入 total_usage["max_tokens_escalations"]。
    """

    RETRY_DELAYS = (1.0, 2.0, 4.0)
    TRUNCATION_RETRIES = 2
    TRUNCATION_MAX_TOKENS_CEILING = 16000

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        cfg = self.config
        api_key = os.environ.get(cfg.api_key_env, "").strip()
        if not api_key:
            raise LLMError(
                f"模型 {cfg.name} 缺少 API key：请设置环境变量 {cfg.api_key_env}"
                "（可复制 .env.example 为 .env 并填入）。"
            )
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
        }
        if seed is not None:
            # OpenAI 兼容端点的尽力复现参数；不支持该字段的供应商会静默忽略。
            payload["seed"] = seed
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        requested_max_tokens = payload["max_tokens"]
        last_error = ""
        net_attempt = 0
        trunc_attempt = 0
        while True:
            try:
                async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                # 超时：可重试
                last_error = f"请求超时：{exc!r}"
            except httpx.TransportError as exc:
                # 瞬时传输层故障（连接被掐断、协议错误、代理抖动等）：可重试
                last_error = f"网络传输错误：{exc!r}"
            except httpx.HTTPError as exc:
                # 其他请求错误（非法 URL、重定向过多等）：不重试，直接抛出
                raise LLMError(f"模型 {cfg.name} 请求失败（{url}）：{exc!r}") from exc
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    # 限流或服务端错误：可重试
                    last_error = f"HTTP {resp.status_code}：{resp.text[:200]}"
                elif resp.status_code >= 400:
                    # 其余 4xx：不重试，带响应体摘要直接抛出
                    raise LLMError(
                        f"模型 {cfg.name} 调用失败（HTTP {resp.status_code}，不可重试）："
                        f"{resp.text[:300]}"
                    )
                else:
                    try:
                        text = self._parse_response(resp)
                    except LLMTruncationError:
                        # 截断：升额重试（独立计数，不占用网络重试次数、不退避等待）
                        if (
                            trunc_attempt >= self.TRUNCATION_RETRIES
                            or payload["max_tokens"] >= self.TRUNCATION_MAX_TOKENS_CEILING
                        ):
                            raise
                        trunc_attempt += 1
                        payload["max_tokens"] = min(
                            payload["max_tokens"] * 2, self.TRUNCATION_MAX_TOKENS_CEILING
                        )
                        continue
                    if trunc_attempt:
                        self._record_max_tokens_escalation(requested_max_tokens, payload["max_tokens"])
                    return text
            if net_attempt < len(self.RETRY_DELAYS):
                await asyncio.sleep(self.RETRY_DELAYS[net_attempt])
                net_attempt += 1
                continue
            raise LLMError(
                f"模型 {cfg.name} 重试 {len(self.RETRY_DELAYS)} 次后仍失败：{last_error}"
            )

    def _parse_response(self, resp: httpx.Response) -> str:
        """解析响应体，累加 usage；输出被 max_tokens 截断时报错而非静默接受。"""
        cfg = self.config
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"模型 {cfg.name} 响应不是合法 JSON：{resp.text[:300]}") from exc
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"模型 {cfg.name} 响应格式异常：{str(data)[:300]}") from exc
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            self.total_usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.total_usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        self.total_usage["calls"] += 1
        if isinstance(choice, dict) and choice.get("finish_reason") == "length":
            raise LLMTruncationError(
                f"模型 {cfg.name} 输出因 max_tokens 上限被截断（finish_reason=length），"
                "自动升额重试后仍不完整；可在 config/models.yaml 调大该模型的 max_tokens 默认值。"
            )
        return content if isinstance(content, str) else str(content)


class AnthropicClient(BaseLLMClient):
    """Anthropic 原生 Messages API 客户端，429/5xx/超时指数退避重试。

    与 OpenAICompatClient 的差异只在请求/响应格式：POST {base_url}/v1/messages
    （而非 /chat/completions），鉴权走 x-api-key + anthropic-version 头（而非
    Authorization: Bearer），messages 里的 system 角色需要抽出放到顶层 system
    字段（Anthropic 不接受 messages 数组里出现 role=system），响应从
    content[0].text 取正文。截断处理同样是 stop_reason=max_tokens 时把
    max_tokens 翻倍重试，行为对齐 OpenAICompatClient。
    """

    RETRY_DELAYS = (1.0, 2.0, 4.0)
    TRUNCATION_RETRIES = 2
    TRUNCATION_MAX_TOKENS_CEILING = 16000
    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        cfg = self.config
        api_key = os.environ.get(cfg.api_key_env, "").strip()
        if not api_key:
            raise LLMError(
                f"模型 {cfg.name} 缺少 API key：请设置环境变量 {cfg.api_key_env}"
                "（可复制 .env.example 为 .env 并填入）。"
            )
        url = cfg.base_url.rstrip("/") + "/v1/messages"
        # Anthropic 不接受 messages 数组里的 role=system：抽出拼成顶层 system 字段，
        # 其余 user/assistant 原样传递。
        system_parts = [
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        ]
        chat_messages = [m for m in messages if m.get("role") != "system"]
        payload: dict = {
            "model": cfg.model,
            "messages": chat_messages,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
            "temperature": cfg.temperature if temperature is None else temperature,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        # seed 无对应参数，Anthropic Messages API 不支持复现种子；静默忽略。
        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

        requested_max_tokens = payload["max_tokens"]
        last_error = ""
        net_attempt = 0
        trunc_attempt = 0
        while True:
            try:
                async with httpx.AsyncClient(timeout=cfg.timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                # 超时：可重试
                last_error = f"请求超时：{exc!r}"
            except httpx.TransportError as exc:
                # 瞬时传输层故障（连接被掐断、协议错误、代理抖动等）：可重试
                last_error = f"网络传输错误：{exc!r}"
            except httpx.HTTPError as exc:
                # 其他请求错误（非法 URL、重定向过多等）：不重试，直接抛出
                raise LLMError(f"模型 {cfg.name} 请求失败（{url}）：{exc!r}") from exc
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    # 限流或服务端错误：可重试
                    last_error = f"HTTP {resp.status_code}：{resp.text[:200]}"
                elif resp.status_code >= 400:
                    # 其余 4xx：不重试，带响应体摘要直接抛出
                    raise LLMError(
                        f"模型 {cfg.name} 调用失败（HTTP {resp.status_code}，不可重试）："
                        f"{resp.text[:300]}"
                    )
                else:
                    try:
                        text = self._parse_response(resp)
                    except LLMTruncationError:
                        # 截断：升额重试（独立计数，不占用网络重试次数、不退避等待）
                        if (
                            trunc_attempt >= self.TRUNCATION_RETRIES
                            or payload["max_tokens"] >= self.TRUNCATION_MAX_TOKENS_CEILING
                        ):
                            raise
                        trunc_attempt += 1
                        payload["max_tokens"] = min(
                            payload["max_tokens"] * 2, self.TRUNCATION_MAX_TOKENS_CEILING
                        )
                        continue
                    if trunc_attempt:
                        self._record_max_tokens_escalation(requested_max_tokens, payload["max_tokens"])
                    return text
            if net_attempt < len(self.RETRY_DELAYS):
                await asyncio.sleep(self.RETRY_DELAYS[net_attempt])
                net_attempt += 1
                continue
            raise LLMError(
                f"模型 {cfg.name} 重试 {len(self.RETRY_DELAYS)} 次后仍失败：{last_error}"
            )

    def _parse_response(self, resp: httpx.Response) -> str:
        """解析响应体，累加 usage；输出被 max_tokens 截断时报错而非静默接受。"""
        cfg = self.config
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"模型 {cfg.name} 响应不是合法 JSON：{resp.text[:300]}") from exc
        try:
            content = data["content"]
            text = content[0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"模型 {cfg.name} 响应格式异常：{str(data)[:300]}") from exc
        usage = data.get("usage") or {}
        if isinstance(usage, dict):
            self.total_usage["prompt_tokens"] += int(usage.get("input_tokens") or 0)
            self.total_usage["completion_tokens"] += int(usage.get("output_tokens") or 0)
        self.total_usage["calls"] += 1
        if data.get("stop_reason") == "max_tokens":
            raise LLMTruncationError(
                f"模型 {cfg.name} 输出因 max_tokens 上限被截断（stop_reason=max_tokens），"
                "自动升额重试后仍不完整；可在 config/models.yaml 调大该模型的 max_tokens 默认值。"
            )
        return text if isinstance(text, str) else str(text)


MOCK_DEFAULT_REPLY = "（模拟回复）我了解的情况有限，但可以分享一些看法。"


class MockLLMClient(BaseLLMClient):
    """脚本化 Mock 客户端：按序弹出回复，弹完后重复最后一条。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__()
        self._responses = list(responses) if responses else None

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> str:
        self.total_usage["calls"] += 1
        if not self._responses:
            return MOCK_DEFAULT_REPLY
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def create_client(model_name: str, registry: dict[str, ModelConfig]) -> BaseLLMClient:
    """按模型名创建客户端；mock provider 返回 MockLLMClient。"""
    if model_name not in registry:
        available = "、".join(sorted(registry)) or "（无）"
        raise LLMError(f"模型 {model_name} 不在注册表中，可用模型：{available}")
    cfg = registry[model_name]
    if cfg.provider == "mock":
        return MockLLMClient()
    if cfg.provider == "openai-compatible":
        return OpenAICompatClient(cfg)
    if cfg.provider == "anthropic":
        return AnthropicClient(cfg)
    raise LLMError(f"模型 {model_name} 的 provider 不受支持：{cfg.provider}")
