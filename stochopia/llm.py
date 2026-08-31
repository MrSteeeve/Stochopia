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


# 长输出调用按用途覆盖 max_tokens，避免默认 2000 不够用导致输出被截断
LONG_OUTPUT_MAX_TOKENS = 4000


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


# defaults 节缺失时的内置默认模型名（兼容 v0.1 无 defaults 的旧配置）
BUILTIN_DEFAULTS: dict[str, str] = {
    "test_model": "deepseek-v4-flash",
    "env_model": "deepseek-v4-flash",
    "judge_model": "deepseek-v4-pro",
    "polish_model": "deepseek-v4-flash",
}


class ModelRegistry(dict):
    """模型注册表：本体是 dict[str, ModelConfig]，defaults 存各用途的默认模型名。"""

    def __init__(self, models: dict[str, ModelConfig], defaults: dict[str, str]) -> None:
        super().__init__(models)
        self.defaults = defaults


def load_model_registry(path: str | Path) -> ModelRegistry:
    """解析 config/models.yaml；文件缺失或格式错误抛 LLMError。

    兼容无 defaults 节的旧配置：缺失的默认模型名回退到内置值。
    """
    p = Path(path)
    if not p.is_file():
        raise LLMError(f"模型注册表文件不存在：{p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LLMError(f"模型注册表 YAML 解析失败（{p}）：{exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise LLMError(f"模型注册表格式错误（{p}）：顶层需要 models 映射")
    defaults_node = data.get("defaults") or {}
    if not isinstance(defaults_node, dict):
        raise LLMError(f"模型注册表格式错误（{p}）：defaults 需为映射")
    unknown_defaults = set(defaults_node) - set(BUILTIN_DEFAULTS)
    if unknown_defaults:
        raise LLMError(f"defaults 节含未知字段（{p}）：{sorted(unknown_defaults)}")
    defaults = {**BUILTIN_DEFAULTS, **{k: str(v) for k, v in defaults_node.items()}}
    registry: dict[str, ModelConfig] = {}
    for name, cfg in data["models"].items():
        if not isinstance(cfg, dict) or "provider" not in cfg:
            raise LLMError(f"模型 {name} 配置格式错误（{p}）：需要 provider 字段")
        allowed = {"provider", "base_url", "model", "api_key_env", "temperature", "max_tokens", "timeout"}
        unknown = set(cfg) - allowed
        if unknown:
            raise LLMError(f"模型 {name} 配置含未知字段（{p}）：{sorted(unknown)}")
        registry[name] = ModelConfig(name=name, **cfg)
    return ModelRegistry(registry, defaults)


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


class OpenAICompatClient(BaseLLMClient):
    """OpenAI 兼容 chat/completions 客户端，429/5xx/超时指数退避重试。

    输出被 max_tokens 截断（finish_reason=length）时不直接报错，而是把
    max_tokens 翻倍后立即重试（独立计数，最多 TRUNCATION_RETRIES 次，
    封顶 TRUNCATION_MAX_TOKENS_CEILING），仍截断才抛 LLMTruncationError。
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
                        return self._parse_response(resp)
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
                        return self._parse_response(resp)
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


def create_client(
    model_name: str, registry: dict[str, ModelConfig], dry_run: bool = False
) -> BaseLLMClient:
    """按模型名创建客户端；dry_run 或 mock provider 返回 MockLLMClient。"""
    if dry_run:
        return MockLLMClient()
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


# --dry-run 的脚本化参考产品：一份自动赎回结构（autocallable），目标客户 client_b。
# 经过手工验证：
#   R8-R11（波动偏高段）hurdle_hit_prob 超过 0.50 门槛，可成交并产生佣金；
#   R12+ 触发监管期限上限（6 个月）约束而终止，属预期行为；R1-7 胜率偏低未成交。
# notional=15_000_000：不超过 client_b 在 R10-11（capital 缩至 1500 万）的可投额度；
# 同时满足 basic_notional_limit 上限（≤ 5000 万）。
# 每轮固定 4 步：① 查询 client_b（满足接触门）② 查询 trading_desk ③ 查询 risk_control ④ 提交产品。
DRY_RUN_PRODUCT: dict = {
    "product_type": "autocallable",
    "notional": 15000000,
    "maturity_months": 12,
    "strike_pct": 1.0,
    "barrier_pct": 0.75,
    "barrier_type": "knock_in",
    "coupon_rate": 0.15,
    "participation_rate": 1.0,
    "principal_protected": False,
    "target_client": "client_b",
    "pitch": "自动赎回票据：高波动期获取票息，若提前触发赎回则锁定年化收益",
    "hedging_plan": "delta 对冲，通过期权组合管理敲入敲出风险敞口",
}


def make_dry_run_clients(scenario) -> tuple[BaseLLMClient, dict[str, BaseLLMClient]]:
    """构造脚本化 Mock 客户端，保证 --dry-run 能以零成本跑通整个 Playground。

    被测（结构化产品设计师）客户端：每轮固定 4 步动作——
    ① 查询 client_b（满足接触门，获取本轮客户简报）
    ② 向 trading_desk 提问并附产品草案 JSON（触发确定性报价，跳过 LLM）
    ③ 向 risk_control 提问并附产品草案 JSON（触发合规预检，跳过 LLM）
    ④ 提交同一份 autocallable 产品（见 DRY_RUN_PRODUCT）
    脚本按 scenario.total_rounds × 4 条准备好，逐轮弹出，不依赖 MockLLMClient
    的"弹完重复最后一条"兜底行为。

    R8-R11：胜率达标，成交并产生佣金。R12 起监管收紧期限上限，脚本产品
    同步切换为 6 个月期（避免触发违规修改流程、打乱固定步数的脚本节奏），
    此时客户胜率不足，成交失败属预期行为。

    环境智能体：各自返回一两句简短的固定回复。
    """
    agent_ids = scenario.agent_ids()
    desk_target = "trading_desk" if "trading_desk" in agent_ids else agent_ids[0]
    risk_target = "risk_control" if "risk_control" in agent_ids else agent_ids[-1]
    client_target = "client_b" if "client_b" in agent_ids else agent_ids[0]

    responses: list[str] = []
    for round_num in range(1, scenario.total_rounds + 1):
        # R12 起 maturity_cap 约束生效：切换 6 个月期，避免违规修改打乱脚本步数
        product = {**DRY_RUN_PRODUCT}
        if round_num >= 12:
            product["maturity_months"] = 6
        _draft = {k: v for k, v in product.items() if k != "target_client"}
        _draft_str = json.dumps(_draft, ensure_ascii=False)
        # 步骤 1：查询目标客户（client_b），满足接触门
        responses.append(
            "```json\n"
            + json.dumps(
                {
                    "action": "query",
                    "target": client_target,
                    "message": "李总您好，请问您本轮的配置需求和风险偏好？",
                },
                ensure_ascii=False,
            )
            + "\n```"
        )
        # 步骤 2：查询 trading_desk（附草案 → 触发确定性报价，跳过 LLM）
        responses.append(
            "```json\n"
            + json.dumps(
                {
                    "action": "query",
                    "target": desk_target,
                    "message": f"请对以下产品草案出具确定性报价：{_draft_str}",
                },
                ensure_ascii=False,
            )
            + "\n```"
        )
        # 步骤 3：查询 risk_control（附草案 → 触发合规预检，跳过 LLM）
        responses.append(
            "```json\n"
            + json.dumps(
                {
                    "action": "query",
                    "target": risk_target,
                    "message": f"请对以下产品草案进行合规预检：{_draft_str}",
                },
                ensure_ascii=False,
            )
            + "\n```"
        )
        # 步骤 4：提交产品
        responses.append(
            "```json\n"
            + json.dumps({"action": "submit_product", "product": product}, ensure_ascii=False)
            + "\n```"
        )
    test_client: BaseLLMClient = MockLLMClient(responses)

    env_clients: dict[str, BaseLLMClient] = {
        spec.id: MockLLMClient(
            [
                f"（模拟）我是{spec.display_name}，目前情况大致稳定，暂时没有特别要补充的。",
                f"（模拟）作为{spec.display_name}，这一点我们内部会持续关注。",
            ]
        )
        for spec in scenario.agents
    }
    return test_client, env_clients
