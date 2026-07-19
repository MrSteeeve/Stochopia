"""角色配置 schema（v2 全角色 LLM 化基础设施）。

按 draft_codex §2 实现 RoleSpecV2 / InferenceSpec / RetryPolicy 与
``load_role_specs``。连接信息仍放 ``config/models.yaml``，角色行为放
``config/benchmark_roles.yaml``；本模块只负责把后者严格加载为冻结的
RoleSpecV2 集合，并在加载时计算 system prompt 的 SHA256。

loader 采取 fail-fast：任何未知字段、重复 role、缺失 prompt、模型不在
registry、seed_offset 重复、或 client/risk/desk 的 numeric_authority 不是
``supplied_facts_only`` 都会抛 :class:`RoleConfigError`。structurer 的
``model_ref`` 支持 ``${job.model}`` 占位，加载时原样保留，由调用方在建 job
时解析为具体测试模型名。

``judges:`` 节（离线 judge-runs 批跑的默认模型/repeats）由 :func:`load_judges_config`
加载为 :class:`JudgesConfig`；``load_role_specs`` 在其存在时也会做同样的校验，
但只返回 ``roles`` 部分。judges 节不接受 ``exclude_same_model_family``：
judge-runs 的自评跳过按精确模型名比较实现，模型家族推断不可靠。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .llm import ModelRegistry

# structurer 的 model_ref 占位：加载时不解析，由调用方在建 job 时替换为具体测试模型
JOB_MODEL_PLACEHOLDER = "${job.model}"

# 环境角色（非 structurer、非 judge）的数值权威只允许这一个取值：一切数字都来自
# 系统注入的 supplied facts，角色不得自行编造数字。
_SUPPLIED_FACTS_ONLY = "supplied_facts_only"

_ROLE_TYPES = ("structurer", "client", "risk_control", "trading_desk", "judge")
# 这些角色类型必须 numeric_authority == supplied_facts_only
_NUMERIC_LOCKED_ROLES = frozenset({"client", "risk_control", "trading_desk"})

_ALLOWED_TOP_KEYS = frozenset({"protocol_version", "main_npc_lineup_id", "roles", "judges"})
_ALLOWED_JUDGES_KEYS = frozenset({
    "models", "repeats", "temperature", "max_tokens", "blind_model_identity",
})
_ALLOWED_ROLE_KEYS = frozenset({
    "role",
    "model_ref",
    "system_prompt_file",
    "private_state_file",
    "temperature",
    "max_tokens",
    "timeout_s",
    "seed_policy",
    "seed_offset",
    "retry",
    "output_schema",
    "tools",
    "max_calls_per_round",
    "history_scope",
    "numeric_authority",
    "failure_policy",
})
_ALLOWED_RETRY_KEYS = frozenset({"transport_retries", "format_retries", "backoff_s"})
_SEED_POLICIES = ("derived", "fixed", "none")
_HISTORY_SCOPES = ("round", "episode")


class RoleConfigError(Exception):
    """角色配置加载相关异常，报错信息应带足上下文（文件、角色、字段）。"""


@dataclass(frozen=True)
class InferenceSpec:
    """一个角色的推理参数（不含连接信息，连接信息在 models.yaml）。"""

    model_ref: str
    temperature: float
    max_tokens: int
    timeout_s: float
    seed_policy: Literal["derived", "fixed", "none"]
    seed_offset: int


@dataclass(frozen=True)
class RetryPolicy:
    """传输层与格式层的重试策略。"""

    transport_retries: int = 2
    format_retries: int = 1
    backoff_s: tuple[float, ...] = (1.0, 2.0)


@dataclass(frozen=True)
class JudgesConfig:
    """``judges:`` 节的校验结果（离线 judge-runs 批跑的默认模型/重复次数配置）。

    没有 ``exclude_same_model_family`` 字段：judge-runs 的自评跳过按
    ``judge_model == item["model"]`` 精确模型名比较实现（见
    ``benchmark_cli._run_judge_runs``），模型家族推断不可靠，故不收这个字段。
    """

    models: tuple[str, ...]
    repeats: int
    temperature: float
    max_tokens: int
    blind_model_identity: bool


@dataclass(frozen=True)
class RoleSpecV2:
    """一个角色的完整规格（行为契约 + 推理参数 + prompt 指纹）。"""

    id: str
    role: Literal["structurer", "client", "risk_control", "trading_desk", "judge"]
    system_prompt_file: Path
    system_prompt_sha256: str
    inference: InferenceSpec
    retry: RetryPolicy
    output_schema: str
    tools: tuple[str, ...]
    max_calls_per_round: int
    history_scope: Literal["round", "episode"]
    numeric_authority: Literal["none", "supplied_facts_only"]
    failure_policy: str
    private_state_file: Path | None = None


def _require(mapping: dict, key: str, ctx: str):
    if key not in mapping or mapping[key] is None:
        raise RoleConfigError(f"{ctx} 缺少必填字段 {key!r}")
    return mapping[key]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_path(base_dir: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p)


def _parse_retry(raw, ctx: str) -> RetryPolicy:
    if raw is None:
        return RetryPolicy()
    if not isinstance(raw, dict):
        raise RoleConfigError(f"{ctx} 的 retry 必须是映射，实际为 {type(raw).__name__}")
    unknown = set(raw) - _ALLOWED_RETRY_KEYS
    if unknown:
        raise RoleConfigError(f"{ctx} 的 retry 含未知字段：{sorted(unknown)}")
    transport = int(raw.get("transport_retries", 2))
    fmt = int(raw.get("format_retries", 1))
    backoff_raw = raw.get("backoff_s", (1.0, 2.0))
    if not isinstance(backoff_raw, (list, tuple)):
        raise RoleConfigError(f"{ctx} 的 retry.backoff_s 必须是列表")
    backoff = tuple(float(x) for x in backoff_raw)
    return RetryPolicy(transport_retries=transport, format_retries=fmt, backoff_s=backoff)


def _parse_role(role_id: str, raw: dict, registry: ModelRegistry, base_dir: Path) -> RoleSpecV2:
    ctx = f"角色 {role_id!r}"
    if not isinstance(raw, dict):
        raise RoleConfigError(f"{ctx} 配置必须是映射，实际为 {type(raw).__name__}")
    unknown = set(raw) - _ALLOWED_ROLE_KEYS
    if unknown:
        raise RoleConfigError(f"{ctx} 含未知字段：{sorted(unknown)}")

    role = str(_require(raw, "role", ctx))
    if role not in _ROLE_TYPES:
        raise RoleConfigError(f"{ctx} 的 role 非法：{role!r}，合法值 {_ROLE_TYPES}")

    model_ref = str(_require(raw, "model_ref", ctx))
    if model_ref != JOB_MODEL_PLACEHOLDER and model_ref not in registry:
        available = "、".join(sorted(registry)) or "（无）"
        raise RoleConfigError(
            f"{ctx} 的 model_ref {model_ref!r} 不在模型注册表中，可用模型：{available}"
        )

    prompt_rel = str(_require(raw, "system_prompt_file", ctx))
    prompt_path = _resolve_path(base_dir, prompt_rel)
    if not prompt_path.is_file():
        raise RoleConfigError(f"{ctx} 的 system_prompt_file 不存在：{prompt_path}")
    prompt_sha = _sha256_text(prompt_path.read_text(encoding="utf-8"))

    private_state_file = raw.get("private_state_file")
    private_path: Path | None = None
    if private_state_file is not None:
        private_path = _resolve_path(base_dir, str(private_state_file))

    numeric_authority = str(raw.get("numeric_authority", "none"))
    if numeric_authority not in ("none", _SUPPLIED_FACTS_ONLY):
        raise RoleConfigError(
            f"{ctx} 的 numeric_authority 非法：{numeric_authority!r}"
        )
    if role in _NUMERIC_LOCKED_ROLES and numeric_authority != _SUPPLIED_FACTS_ONLY:
        raise RoleConfigError(
            f"{ctx}（{role}）的 numeric_authority 必须是 {_SUPPLIED_FACTS_ONLY!r}，"
            f"实际为 {numeric_authority!r}：环境角色的一切数字都必须来自系统注入事实。"
        )

    seed_policy = str(raw.get("seed_policy", "derived"))
    if seed_policy not in _SEED_POLICIES:
        raise RoleConfigError(f"{ctx} 的 seed_policy 非法：{seed_policy!r}，合法值 {_SEED_POLICIES}")

    history_scope = str(raw.get("history_scope", "episode"))
    if history_scope not in _HISTORY_SCOPES:
        raise RoleConfigError(
            f"{ctx} 的 history_scope 非法：{history_scope!r}，合法值 {_HISTORY_SCOPES}"
        )

    tools_raw = raw.get("tools", [])
    if not isinstance(tools_raw, (list, tuple)):
        raise RoleConfigError(f"{ctx} 的 tools 必须是列表")
    tools = tuple(str(t) for t in tools_raw)

    inference = InferenceSpec(
        model_ref=model_ref,
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 600)),
        timeout_s=float(raw.get("timeout_s", 30.0)),
        seed_policy=seed_policy,  # type: ignore[arg-type]
        seed_offset=int(raw.get("seed_offset", 0)),
    )

    return RoleSpecV2(
        id=role_id,
        role=role,  # type: ignore[arg-type]
        system_prompt_file=prompt_path,
        system_prompt_sha256=prompt_sha,
        inference=inference,
        retry=_parse_retry(raw.get("retry"), ctx),
        output_schema=str(_require(raw, "output_schema", ctx)),
        tools=tools,
        max_calls_per_round=int(raw.get("max_calls_per_round", 3)),
        history_scope=history_scope,  # type: ignore[arg-type]
        numeric_authority=numeric_authority,  # type: ignore[arg-type]
        failure_policy=str(raw.get("failure_policy", "no_action")),
        private_state_file=private_path,
    )


def _load_yaml_dict(cfg_path: Path) -> dict:
    if not cfg_path.is_file():
        raise RoleConfigError(f"角色配置文件不存在：{cfg_path}")
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RoleConfigError(f"角色配置 YAML 解析失败（{cfg_path}）：{exc}") from exc
    if not isinstance(data, dict):
        raise RoleConfigError(f"角色配置格式错误（{cfg_path}）：顶层需为映射")
    return data


def _parse_judges(raw: dict, registry: ModelRegistry, ctx: str) -> JudgesConfig:
    """校验并解析 judges 节：未知字段、models 类型/数量/注册表成员资格。

    不校验/不接受 ``exclude_same_model_family``（实际实现按精确模型名跳过
    自评，家族推断不可靠，见 :class:`JudgesConfig`），未知字段一律拒绝。
    """
    if not isinstance(raw, dict):
        raise RoleConfigError(f"{ctx} 必须是映射")
    unknown = set(raw) - _ALLOWED_JUDGES_KEYS
    if unknown:
        raise RoleConfigError(f"{ctx} 含未知字段：{sorted(unknown)}")
    models = raw.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise RoleConfigError(f"{ctx} 的 models 必须是至少两个异构模型的列表")
    models_t = tuple(str(m) for m in models)
    for m in models_t:
        if m not in registry:
            raise RoleConfigError(f"{ctx} 的 models 中的模型 {m!r} 不在注册表中")
    return JudgesConfig(
        models=models_t,
        repeats=int(raw.get("repeats", 3)),
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 1200)),
        blind_model_identity=bool(raw.get("blind_model_identity", True)),
    )


def load_judges_config(path: str | Path, registry: ModelRegistry) -> JudgesConfig:
    """加载 ``benchmark_roles.yaml`` 的 ``judges:`` 节为 :class:`JudgesConfig`。

    独立于 :func:`load_role_specs`（后者仍会做同样的 judges 校验，但只返回
    ``roles`` 部分），供 ``judge-runs`` CLI 在未显式传 ``--judge-models`` /
    ``--repeats`` 时取默认值；文件不存在、格式错误或缺 judges 节都抛
    :class:`RoleConfigError`。
    """
    cfg_path = Path(path)
    data = _load_yaml_dict(cfg_path)
    judges_node = data.get("judges")
    if judges_node is None:
        raise RoleConfigError(f"角色配置缺少 judges 节（{cfg_path}）")
    return _parse_judges(judges_node, registry, f"角色配置 judges（{cfg_path}）")


def load_role_specs(
    path: str | Path,
    registry: ModelRegistry,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, RoleSpecV2]:
    """加载 ``benchmark_roles.yaml`` 为 ``dict[role_id -> RoleSpecV2]``。

    ``base_dir`` 用于解析 prompt / private_state 相对路径；缺省时取配置文件所在
    目录的父目录（即 ``config/benchmark_roles.yaml`` → 仓库根），这样 yaml 里的
    ``scenarios/mirage_csi/prompts/…`` 这类仓库相对路径可直接命中。

    校验（任一失败即抛 :class:`RoleConfigError`）：未知顶层/角色字段、重复
    role、缺 prompt 文件、model_ref 不在 registry、seed_offset 重复、
    client/risk/desk 的 numeric_authority != supplied_facts_only、judges 节
    （存在时）未知字段/models 类型数量/注册表成员资格。judges 节的默认值取用见
    :func:`load_judges_config`。
    """
    cfg_path = Path(path)
    data = _load_yaml_dict(cfg_path)

    unknown_top = set(data) - _ALLOWED_TOP_KEYS
    if unknown_top:
        raise RoleConfigError(f"角色配置含未知顶层字段（{cfg_path}）：{sorted(unknown_top)}")

    resolved_base = Path(base_dir) if base_dir is not None else cfg_path.resolve().parent.parent

    roles_node = _require(data, "roles", str(cfg_path))
    if not isinstance(roles_node, dict) or not roles_node:
        raise RoleConfigError(f"角色配置 roles 必须是非空映射（{cfg_path}）")

    specs: dict[str, RoleSpecV2] = {}
    seen_roles: dict[str, str] = {}
    seen_offsets: dict[int, str] = {}
    for role_id, raw in roles_node.items():
        spec = _parse_role(str(role_id), raw, registry, resolved_base)
        if spec.role in seen_roles:
            raise RoleConfigError(
                f"角色配置重复 role={spec.role!r}（{cfg_path}）："
                f"{seen_roles[spec.role]!r} 与 {spec.id!r} 冲突"
            )
        seen_roles[spec.role] = spec.id
        offset = spec.inference.seed_offset
        if offset in seen_offsets:
            raise RoleConfigError(
                f"角色配置 seed_offset={offset} 重复（{cfg_path}）："
                f"{seen_offsets[offset]!r} 与 {spec.id!r} 冲突"
            )
        seen_offsets[offset] = spec.id
        specs[spec.id] = spec

    # judges 节存在时做完整校验（未知字段、models 类型/数量/注册表成员资格）；
    # 结构化结果由 load_judges_config 单独暴露给 judge-runs CLI。
    judges_node = data.get("judges")
    if judges_node is not None:
        _parse_judges(judges_node, registry, f"角色配置 judges（{cfg_path}）")

    return specs
