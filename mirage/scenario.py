"""场景加载：scenario.yaml、背景、角色 prompt、市场状态、约束与客户画像。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .products import PRODUCT_TYPES, ClientProfile, Constraint, MarketState

CHECK_TYPES = ("max", "min", "max_abs", "forbidden_type")
RISK_APPETITES = ("conservative", "moderate", "aggressive")

# round_overrides 中允许覆盖的客户字段（不含 id、name、round_overrides 等元字段）
OVERRIDABLE_FIELDS = frozenset({
    "capital",
    "max_loss_pct",
    "min_return_pct",
    "risk_appetite",
    "preferences",
    "max_maturity_months",
    "principal_protection_required",
    "allowed_product_types",
    "accepting_new_products",
    "min_hit_prob",
    "current_focus",
})


class ScenarioError(Exception):
    """场景配置相关异常，报错信息带路径。"""


@dataclass
class AgentSpec:
    """一个环境智能体的规格。"""

    id: str
    display_name: str
    public_desc: str
    prompt: str
    model: str | None = None


@dataclass
class PlaygroundScenario:
    """一个完整的 Structurer Playground 场景。"""

    id: str
    name: str
    description: str
    total_rounds: int
    interactions_per_round: int
    background: str
    index: dict
    agents: list[AgentSpec]
    market_states: list[MarketState]
    constraints: list[Constraint]
    clients: list[ClientProfile]

    def agent_ids(self) -> list[str]:
        return [a.id for a in self.agents]

    def get_agent(self, agent_id: str) -> AgentSpec:
        for a in self.agents:
            if a.id == agent_id:
                return a
        raise ScenarioError(f"场景 {self.id} 中不存在智能体：{agent_id}")

    def get_market_state(self, round_num: int) -> MarketState:
        for m in self.market_states:
            if m.round_num == round_num:
                return m
        raise ScenarioError(f"场景 {self.id} 中不存在第 {round_num} 轮的市场状态")

    def get_client(self, client_id: str, round_num: int | None = None) -> ClientProfile:
        """返回客户画像。

        round_num 为 None 时返回 base profile（v1.1 行为）；
        指定 round_num 时返回一个新的 ClientProfile 副本，
        将所有 round_overrides 中 round_num 匹配的条目按列表顺序合并，
        current_focus 若无 override 则默认为空字符串。
        """
        for c in self.clients:
            if c.id == client_id:
                if round_num is None:
                    return c
                # 按列表顺序合并所有匹配当前轮次的 override
                combined: dict = {}
                for override_entry in c.round_overrides:
                    if round_num in override_entry["rounds"]:
                        for k, v in override_entry.items():
                            if k != "rounds":
                                combined[k] = v
                # 若没有任何 override 设置 current_focus，默认置空
                combined.setdefault("current_focus", "")
                return replace(c, **combined)
        raise ScenarioError(f"场景 {self.id} 中不存在客户：{client_id}")

    def get_clients(self, round_num: int) -> list[ClientProfile]:
        """返回所有客户在 round_num 轮次的 profile 列表。"""
        return [self.get_client(c.id, round_num) for c in self.clients]


def _read_text(path: Path, what: str) -> str:
    """读取文本文件，缺失时报带路径的错。"""
    if not path.is_file():
        raise ScenarioError(f"{what}文件不存在：{path}")
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path, what: str) -> dict:
    """读取 YAML 文件，缺失或解析失败时报带路径的错。"""
    text = _read_text(path, what)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{what}文件 YAML 解析失败（{path}）：{exc}") from exc
    if not isinstance(data, dict):
        raise ScenarioError(f"{what}文件格式错误（{path}）：顶层应为映射")
    return data


def _require(data: dict, key: str, path: Path):
    if key not in data or data[key] is None:
        raise ScenarioError(f"配置缺少字段 {key}（{path}）")
    return data[key]


def _load_market_states(path: Path, index: dict, total_rounds: int) -> list[MarketState]:
    """加载 market_states.yaml，并校验轮次集合恰好为 {1..total_rounds}。"""
    data = _load_yaml(path, "市场状态")
    rounds_cfg = data.get("rounds")
    if not isinstance(rounds_cfg, list) or not rounds_cfg:
        raise ScenarioError(f"市场状态需要非空的 rounds 列表（{path}）")

    states: list[MarketState] = []
    seen_rounds: list[int] = []
    for r in rounds_cfg:
        if not isinstance(r, dict):
            raise ScenarioError(f"市场状态条目格式错误（{path}）：{r!r}")
        for k in ("round", "spot", "volatility", "recent_trend", "vix_level"):
            if k not in r or r[k] is None:
                raise ScenarioError(f"市场状态条目缺少字段 {k}（{path}）：{r!r}")
        try:
            round_num = int(r["round"])
        except (TypeError, ValueError) as exc:
            raise ScenarioError(f"市场状态 round 必须是整数（{path}）：{r!r}") from exc
        seen_rounds.append(round_num)
        try:
            spot = float(r["spot"])
            volatility = float(r["volatility"])
        except (TypeError, ValueError) as exc:
            raise ScenarioError(
                f"市场状态 spot/volatility 必须是数字（{path}）：{r!r}"
            ) from exc
        try:
            trend_alpha = float(r.get("trend_alpha", 0.0))
        except (TypeError, ValueError) as exc:
            raise ScenarioError(f"市场状态 trend_alpha 必须是数字（{path}）：{r!r}") from exc

        risk_free_rate = r.get("risk_free_rate", index.get("risk_free_rate", 0.0))
        dividend_yield = r.get("dividend_yield", index.get("dividend_yield", 0.0))
        try:
            risk_free_rate = float(risk_free_rate)
            dividend_yield = float(dividend_yield)
        except (TypeError, ValueError) as exc:
            raise ScenarioError(
                f"市场状态 risk_free_rate/dividend_yield 必须是数字（{path}）：{r!r}"
            ) from exc

        states.append(
            MarketState(
                round_num=round_num,
                index_name=str(index.get("name", "")),
                spot=spot,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                recent_trend=str(r["recent_trend"]),
                vix_level=str(r["vix_level"]),
                trend_alpha=trend_alpha,
            )
        )

    expected = set(range(1, total_rounds + 1))
    actual = set(seen_rounds)
    if len(seen_rounds) != len(actual):
        raise ScenarioError(f"市场状态 round 出现重复（{path}）：{sorted(seen_rounds)}")
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ScenarioError(
            f"市场状态 round 集合与 total_rounds={total_rounds} 不匹配（{path}）："
            f"缺少 {missing}，多余 {extra}"
        )

    states.sort(key=lambda m: m.round_num)
    return states


def _load_constraints(path: Path) -> list[Constraint]:
    """加载 constraints.yaml，允许空列表。"""
    data = _load_yaml(path, "约束配置")
    constraints_cfg = data.get("constraints")
    if constraints_cfg is None:
        constraints_cfg = []
    if not isinstance(constraints_cfg, list):
        raise ScenarioError(f"约束配置 constraints 必须是列表（{path}）")

    constraints: list[Constraint] = []
    for c in constraints_cfg:
        if not isinstance(c, dict):
            raise ScenarioError(f"约束条目格式错误（{path}）：{c!r}")
        for k in ("id", "activate_round", "source", "description", "check_field", "check_type", "check_value"):
            if k not in c or c[k] is None:
                raise ScenarioError(f"约束条目缺少字段 {k}（{path}）：{c!r}")

        check_type = str(c["check_type"])
        if check_type not in CHECK_TYPES:
            raise ScenarioError(
                f"约束 {c['id']} 的 check_type 必须是 {CHECK_TYPES} 之一，实际为 {check_type!r}（{path}）"
            )

        try:
            activate_round = int(c["activate_round"])
        except (TypeError, ValueError) as exc:
            raise ScenarioError(
                f"约束 {c['id']} 的 activate_round 必须是整数（{path}）"
            ) from exc
        if activate_round < 1:
            raise ScenarioError(
                f"约束 {c['id']} 的 activate_round 必须 >= 1，实际为 {activate_round}（{path}）"
            )

        constraints.append(
            Constraint(
                id=str(c["id"]),
                activate_round=activate_round,
                source=str(c["source"]),
                description=str(c["description"]),
                check_field=str(c["check_field"]),
                check_type=check_type,
                check_value=c["check_value"],
            )
        )
    return constraints


def _load_clients(path: Path, total_rounds: int) -> list[ClientProfile]:
    """加载 clients.yaml，非空列表。v1.2 支持新字段与 round_overrides。"""
    data = _load_yaml(path, "客户配置")
    clients_cfg = data.get("clients")
    if not isinstance(clients_cfg, list) or not clients_cfg:
        raise ScenarioError(f"客户配置需要非空的 clients 列表（{path}）")

    clients: list[ClientProfile] = []
    valid_rounds = set(range(1, total_rounds + 1))

    for c in clients_cfg:
        if not isinstance(c, dict):
            raise ScenarioError(f"客户条目格式错误（{path}）：{c!r}")
        for k in ("id", "name", "capital", "max_loss_pct", "min_return_pct", "risk_appetite"):
            if k not in c or c[k] is None:
                raise ScenarioError(f"客户条目缺少字段 {k}（{path}）：{c!r}")

        risk_appetite = str(c["risk_appetite"])
        if risk_appetite not in RISK_APPETITES:
            raise ScenarioError(
                f"客户 {c['id']} 的 risk_appetite 必须是 {RISK_APPETITES} 之一，"
                f"实际为 {risk_appetite!r}（{path}）"
            )

        try:
            capital = float(c["capital"])
            max_loss_pct = float(c["max_loss_pct"])
            min_return_pct = float(c["min_return_pct"])
        except (TypeError, ValueError) as exc:
            raise ScenarioError(
                f"客户 {c['id']} 的 capital/max_loss_pct/min_return_pct 必须是数字（{path}）"
            ) from exc

        # v1.2 新字段：max_maturity_months
        max_maturity_months_raw = c.get("max_maturity_months", 12)
        try:
            max_maturity_months = int(max_maturity_months_raw)
        except (TypeError, ValueError) as exc:
            raise ScenarioError(
                f"客户 {c['id']} 的 max_maturity_months 必须是整数（{path}）"
            ) from exc

        # v1.2 新字段：principal_protection_required
        principal_protection_required = c.get("principal_protection_required", False)
        if not isinstance(principal_protection_required, bool):
            raise ScenarioError(
                f"客户 {c['id']} 的 principal_protection_required 必须是布尔值（{path}）"
            )

        # v1.2 新字段：accepting_new_products
        accepting_new_products = c.get("accepting_new_products", True)
        if not isinstance(accepting_new_products, bool):
            raise ScenarioError(
                f"客户 {c['id']} 的 accepting_new_products 必须是布尔值（{path}）"
            )

        # v1.2 新字段：min_hit_prob
        min_hit_prob_raw = c.get("min_hit_prob", 0.5)
        try:
            min_hit_prob = float(min_hit_prob_raw)
        except (TypeError, ValueError) as exc:
            raise ScenarioError(
                f"客户 {c['id']} 的 min_hit_prob 必须是数字（{path}）"
            ) from exc
        if not (0.0 <= min_hit_prob <= 1.0):
            raise ScenarioError(
                f"客户 {c['id']} 的 min_hit_prob 必须在 [0, 1] 范围内，"
                f"实际为 {min_hit_prob}（{path}）"
            )

        # v1.2 新字段：allowed_product_types
        allowed_product_types_raw = c.get("allowed_product_types")
        if allowed_product_types_raw is not None:
            if not isinstance(allowed_product_types_raw, list):
                raise ScenarioError(
                    f"客户 {c['id']} 的 allowed_product_types 必须是列表或 null（{path}）"
                )
            for pt in allowed_product_types_raw:
                if pt not in PRODUCT_TYPES:
                    raise ScenarioError(
                        f"客户 {c['id']} 的 allowed_product_types 包含非法类型 {pt!r}，"
                        f"合法值为 {PRODUCT_TYPES}（{path}）"
                    )
            allowed_product_types: list[str] | None = list(allowed_product_types_raw)
        else:
            allowed_product_types = None

        # v1.2 新字段：round_overrides
        round_overrides_raw = c.get("round_overrides", [])
        if not isinstance(round_overrides_raw, list):
            raise ScenarioError(
                f"客户 {c['id']} 的 round_overrides 必须是列表（{path}）"
            )
        round_overrides: list[dict] = []
        for i, ov in enumerate(round_overrides_raw):
            if not isinstance(ov, dict):
                raise ScenarioError(
                    f"客户 {c['id']} 的 round_overrides[{i}] 必须是映射（{path}）"
                )
            if "rounds" not in ov or not isinstance(ov["rounds"], list):
                raise ScenarioError(
                    f"客户 {c['id']} 的 round_overrides[{i}] 缺少 rounds 列表（{path}）"
                )
            for r in ov["rounds"]:
                if r not in valid_rounds:
                    raise ScenarioError(
                        f"客户 {c['id']} 的 round_overrides[{i}] 中的轮次 {r} 超出范围 "
                        f"1..{total_rounds}（{path}）"
                    )
            # 校验 override 中的字段名
            for k in ov:
                if k == "rounds":
                    continue
                if k not in OVERRIDABLE_FIELDS:
                    raise ScenarioError(
                        f"客户 {c['id']} 的 round_overrides[{i}] 包含非法键 {k!r}，"
                        f"可覆盖字段为 {sorted(OVERRIDABLE_FIELDS)}（{path}）"
                    )
            # 校验 override 中的 min_hit_prob
            if "min_hit_prob" in ov:
                try:
                    ohp = float(ov["min_hit_prob"])
                except (TypeError, ValueError) as exc:
                    raise ScenarioError(
                        f"客户 {c['id']} 的 round_overrides[{i}].min_hit_prob 必须是数字（{path}）"
                    ) from exc
                if not (0.0 <= ohp <= 1.0):
                    raise ScenarioError(
                        f"客户 {c['id']} 的 round_overrides[{i}].min_hit_prob 必须在 [0,1] 内，"
                        f"实际为 {ohp}（{path}）"
                    )
            # 校验 override 中的 allowed_product_types
            if "allowed_product_types" in ov and ov["allowed_product_types"] is not None:
                if not isinstance(ov["allowed_product_types"], list):
                    raise ScenarioError(
                        f"客户 {c['id']} 的 round_overrides[{i}].allowed_product_types 必须是列表（{path}）"
                    )
                for pt in ov["allowed_product_types"]:
                    if pt not in PRODUCT_TYPES:
                        raise ScenarioError(
                            f"客户 {c['id']} 的 round_overrides[{i}].allowed_product_types "
                            f"包含非法类型 {pt!r}（{path}）"
                        )
            round_overrides.append(dict(ov))

        clients.append(
            ClientProfile(
                id=str(c["id"]),
                name=str(c["name"]),
                capital=capital,
                max_loss_pct=max_loss_pct,
                min_return_pct=min_return_pct,
                risk_appetite=risk_appetite,
                preferences=str(c.get("preferences", "")),
                max_maturity_months=max_maturity_months,
                principal_protection_required=principal_protection_required,
                allowed_product_types=allowed_product_types,
                accepting_new_products=accepting_new_products,
                min_hit_prob=min_hit_prob,
                round_overrides=round_overrides,
            )
        )
    return clients


def load_scenario(scenario_dir: str | Path) -> PlaygroundScenario:
    """加载一个 Structurer Playground 场景目录。"""
    d = Path(scenario_dir)
    yaml_path = d / "scenario.yaml"
    data = _load_yaml(yaml_path, "场景配置")

    name = str(_require(data, "name", yaml_path))
    description = str(_require(data, "description", yaml_path))
    try:
        total_rounds = int(_require(data, "total_rounds", yaml_path))
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"total_rounds 必须是整数（{yaml_path}）") from exc

    interactions_per_round_raw = data.get("interactions_per_round", 5)
    try:
        interactions_per_round = int(interactions_per_round_raw)
    except (TypeError, ValueError) as exc:
        raise ScenarioError(f"interactions_per_round 必须是整数（{yaml_path}）") from exc

    index = data.get("index") or {}
    if not isinstance(index, dict):
        raise ScenarioError(f"index 必须是映射（{yaml_path}）")

    background = _read_text(d / str(_require(data, "background_file", yaml_path)), "场景背景")

    market_states = _load_market_states(
        d / str(_require(data, "market_states_file", yaml_path)), index, total_rounds
    )
    constraints = _load_constraints(d / str(_require(data, "constraints_file", yaml_path)))
    clients = _load_clients(
        d / str(_require(data, "clients_file", yaml_path)), total_rounds
    )

    agents_cfg = _require(data, "agents", yaml_path)
    if not isinstance(agents_cfg, list) or not agents_cfg:
        raise ScenarioError(f"agents 必须是非空列表（{yaml_path}）")
    agents: list[AgentSpec] = []
    for a in agents_cfg:
        if not isinstance(a, dict):
            raise ScenarioError(f"agent 配置格式错误（{yaml_path}）：{a!r}")
        for k in ("id", "display_name", "public_desc", "prompt_file"):
            if k not in a or a[k] is None:
                raise ScenarioError(f"agent 配置缺少字段 {k}（{yaml_path}）：{a!r}")
        prompt = _read_text(d / str(a["prompt_file"]), f"角色 {a['id']} 的 prompt")
        agents.append(
            AgentSpec(
                id=str(a["id"]),
                display_name=str(a["display_name"]),
                public_desc=str(a["public_desc"]),
                prompt=prompt,
                model=a.get("model"),
            )
        )

    # 为客户智能体注入定性披露条款（具体数字由引擎每轮注入；此处只注入行为准则）
    client_ids = {c.id for c in clients}
    for agent in agents:
        if agent.id in client_ids:
            agent.prompt += (
                "\n\n当被直接问到投资门槛时你会如实说出当前的具体数字；"
                "你的需求会随市场变化，以每轮系统注入的最新状态为准。"
            )

    return PlaygroundScenario(
        id=d.name,
        name=name,
        description=description,
        total_rounds=total_rounds,
        interactions_per_round=interactions_per_round,
        background=background,
        index=index,
        agents=agents,
        market_states=market_states,
        constraints=constraints,
        clients=clients,
    )


def list_scenarios(scenarios_root: str | Path) -> list[str]:
    """列出含 scenario.yaml 的子目录名。"""
    root = Path(scenarios_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "scenario.yaml").is_file())
