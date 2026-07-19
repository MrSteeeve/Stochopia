"""产品与约束：市场状态、产品规格、客户画像与约束校验的数据结构。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


class ProductError(Exception):
    """产品规格非法或无法解析。"""


PRODUCT_TYPES = (
    "vanilla_call",
    "vanilla_put",
    "barrier_call",
    "barrier_put",
    "autocallable",
    "snowball",
    "custom",
)


@dataclass
class MarketState:
    """当轮的市场环境。"""

    round_num: int
    index_name: str
    spot: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float
    recent_trend: str
    vix_level: str
    trend_alpha: float = 0.0


@dataclass
class ProductSpec:
    """被测智能体设计的结构化产品。"""

    product_type: str
    notional: float
    maturity_months: int
    strike_pct: float
    barrier_pct: float | None
    barrier_type: str | None
    coupon_rate: float | None
    participation_rate: float
    principal_protected: bool
    target_client: str
    pitch: str
    hedging_plan: str


def _prob_bucket(p: float) -> str:
    """把概率映射为客户口吻的定性表述（客户不会说小数点概率）。"""
    if p < 0.2:
        return "很小"
    if p < 0.4:
        return "较小"
    if p < 0.6:
        return "中等"
    if p < 0.8:
        return "较大"
    return "很大"


@dataclass
class ClientProfile:
    """客户画像：资金规模、风险偏好与决策阈值。

    带 round_ 前缀语义的字段（max_maturity_months、principal_protection_required、
    allowed_product_types、accepting_new_products、current_focus）描述"当轮状态"，
    由 scenario.get_client(id, round_num) 按轮次时间表解析后返回的副本携带。
    """

    id: str
    name: str
    capital: float
    max_loss_pct: float
    min_return_pct: float
    risk_appetite: str
    preferences: str = ""
    max_maturity_months: int = 12
    principal_protection_required: bool = False
    allowed_product_types: list[str] | None = None
    accepting_new_products: bool = True
    min_hit_prob: float = 0.5
    current_focus: str = ""
    round_overrides: list[dict] = field(default_factory=list)

    def would_buy(self, product: ProductSpec, pricing: dict) -> tuple[bool, str]:
        """依据定价结果，确定性地判断客户是否会购买该产品。

        客户报出的 min_return_pct 语义是"产品要有像样的机会做到这个年化"：
        决策检查 pricing["hurdle_hit_prob"]（真实漂移下实现年化 ≥ 门槛的概率）
        是否达到 min_hit_prob，而非概率加权的平均收益。
        pricing 另需包含 client_price（元）与 loss_frac（预期损失比例，期望口径）。
        本方法应在"当轮解析后"的客户状态上调用。
        """
        if not self.accepting_new_products:
            return False, "本轮暂停申购新产品"

        if self.allowed_product_types is not None and (
            product.product_type not in self.allowed_product_types
        ):
            return False, f"产品类型 {product.product_type} 不在本轮可接受范围内"

        if product.maturity_months > self.max_maturity_months:
            return (
                False,
                f"期限超出本轮可接受上限（{product.maturity_months} > {self.max_maturity_months} 个月）",
            )

        if self.principal_protection_required and not product.principal_protected:
            return False, "本轮要求保本，产品不保本"

        if self.risk_appetite == "conservative" and not product.principal_protected:
            return False, "不保本，不符合风险偏好"

        if product.notional > self.capital:
            return False, "名义本金超过可投资金额"

        loss_frac = pricing["loss_frac"]
        if loss_frac > self.max_loss_pct + 1e-9:
            return (
                False,
                f"潜在损失超出承受范围（{loss_frac:.2%} > {self.max_loss_pct:.2%}）",
            )

        hit_prob = pricing["hurdle_hit_prob"]
        if hit_prob < self.min_hit_prob - 1e-9:
            return (
                False,
                f"做到年化 {self.min_return_pct:.1%} 的把握{_prob_bucket(hit_prob)}，"
                f"达不到我的要求（我要有{_prob_bucket(self.min_hit_prob)}的把握才行）",
            )

        return True, "符合投资标准"


@dataclass
class Constraint:
    """一条在特定轮次生效的合规/风控约束。"""

    id: str
    activate_round: int
    source: str
    description: str
    check_field: str
    check_type: str
    check_value: float | str

    def validate(self, product: ProductSpec, greeks: dict) -> tuple[bool, str]:
        """校验产品是否满足本约束。check_field 优先取自 greeks，其次取产品字段。"""
        if self.check_field in greeks:
            value = greeks[self.check_field]
        elif hasattr(product, self.check_field):
            value = getattr(product, self.check_field)
        else:
            raise ProductError(
                f"约束 {self.id} 配置错误：字段 {self.check_field} 既不在 greeks 中，"
                f"也不是产品字段"
            )

        tol = 1e-9
        if self.check_type == "max":
            if value <= self.check_value + tol:
                return True, "通过"
            return (
                False,
                f"{self.description}：{self.check_field} 实际值 {value} 超过上限 {self.check_value}",
            )
        if self.check_type == "min":
            if value >= self.check_value - tol:
                return True, "通过"
            return (
                False,
                f"{self.description}：{self.check_field} 实际值 {value} 低于下限 {self.check_value}",
            )
        if self.check_type == "max_abs":
            if abs(value) <= self.check_value + tol:
                return True, "通过"
            return (
                False,
                f"{self.description}：{self.check_field} 实际值 {value} 的绝对值超过上限 {self.check_value}",
            )
        if self.check_type == "forbidden_type":
            if product.product_type != self.check_value:
                return True, "通过"
            return (
                False,
                f"{self.description}：产品类型 {product.product_type} 属于被禁止的类型 {self.check_value}",
            )
        raise ProductError(f"约束 {self.id} 配置错误：未知的 check_type {self.check_type!r}")


@dataclass
class RoundResult:
    """单轮博弈结果：记录该轮的市场状态、生效约束、设计出的产品（若有）及其
    定价（公允价值、希腊字母、对冲成本）、报给客户的价格、触发的约束违规、
    客户的成交/拒绝决策、本轮及累计佣金、声誉值，以及结构化设计师的推理文本。
    """

    round_num: int
    market_state: MarketState
    active_constraints: list[str]
    product_designed: ProductSpec | None
    fair_value: float | None
    greeks: dict | None
    hedging_cost: float | None
    client_price: float | None
    constraint_violations: list[str]
    client_decision: tuple[bool, str]
    commission: float
    cumulative_commission: float
    reputation: float
    structurer_reasoning: str


@dataclass
class PlaygroundResult:
    """一次完整对局的结果。"""

    rounds: list[RoundResult]
    total_commission: float
    usage: dict = field(default_factory=dict)


DRAFT_TARGET = "__draft__"

# 显式 JSON null 与字段缺失等价的可选字段及其默认值
_NULLABLE_DEFAULTS = {
    "participation_rate": 1.0,
    "principal_protected": False,
    "pitch": "",
    "hedging_plan": "",
}


def parse_product_spec(data: dict) -> ProductSpec:
    """解析并校验提交的产品 JSON，字段齐全且合法才返回 ProductSpec，否则聚合报错。

    可选字段（participation_rate、principal_protected、pitch、hedging_plan）
    的显式 null 视同缺失，取默认值。
    """
    if not isinstance(data, dict):
        raise ProductError("产品规格必须是字典（JSON 对象）")

    data = {
        k: (_NULLABLE_DEFAULTS[k] if k in _NULLABLE_DEFAULTS and v is None else v)
        for k, v in data.items()
    }

    errors: list[str] = []

    def _is_number(x) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)

    def _is_finite_number(x) -> bool:
        return _is_number(x) and math.isfinite(x)

    product_type = data.get("product_type")
    if product_type not in PRODUCT_TYPES:
        errors.append(
            f"product_type 必须是 {PRODUCT_TYPES} 之一，实际为 {product_type!r}"
        )

    notional = data.get("notional")
    if not _is_finite_number(notional) or notional <= 0 or notional > 1e12:
        errors.append(f"notional 必须是 (0, 1e12] 范围内的有限数字，实际为 {notional!r}")

    maturity_months_raw = data.get("maturity_months")
    maturity_months: int | None = None
    if not _is_finite_number(maturity_months_raw):
        errors.append(f"maturity_months 必须是整数，实际为 {maturity_months_raw!r}")
    else:
        try:
            is_integer = float(maturity_months_raw) == int(maturity_months_raw)
        except (OverflowError, ValueError):
            is_integer = False
        if not is_integer:
            errors.append(f"maturity_months 必须是整数，实际为 {maturity_months_raw!r}")
        else:
            maturity_months = int(maturity_months_raw)
            if not (1 <= maturity_months <= 60):
                errors.append(f"maturity_months 必须在 1 到 60 之间，实际为 {maturity_months}")

    strike_pct = data.get("strike_pct")
    if not _is_finite_number(strike_pct) or strike_pct <= 0 or strike_pct > 10:
        errors.append(f"strike_pct 必须是 (0, 10] 范围内的有限数字，实际为 {strike_pct!r}")

    barrier_pct = data.get("barrier_pct")
    if barrier_pct is not None and (
        not _is_finite_number(barrier_pct) or barrier_pct <= 0 or barrier_pct > 10
    ):
        errors.append(f"barrier_pct 必须为空或 (0, 10] 范围内的有限数字，实际为 {barrier_pct!r}")

    barrier_type = data.get("barrier_type")
    if barrier_type is not None and barrier_type not in ("knock_in", "knock_out"):
        errors.append(
            f"barrier_type 必须为空或 knock_in / knock_out 之一，实际为 {barrier_type!r}"
        )

    if (barrier_pct is None) != (barrier_type is None):
        errors.append("barrier_pct 与 barrier_type 必须同时设置或同时为空")

    coupon_rate = data.get("coupon_rate")
    if coupon_rate is not None and (
        not _is_finite_number(coupon_rate) or coupon_rate < 0 or coupon_rate > 5
    ):
        errors.append(f"coupon_rate 必须为空或 [0, 5] 范围内的有限数字，实际为 {coupon_rate!r}")

    if product_type in ("autocallable", "snowball") and coupon_rate is None:
        errors.append(f"{product_type} 类型必须设置 coupon_rate")

    participation_rate = data.get("participation_rate", 1.0)
    if not _is_finite_number(participation_rate) or participation_rate < 0 or participation_rate > 10:
        errors.append(
            f"participation_rate 必须是 [0, 10] 范围内的有限数字，实际为 {participation_rate!r}"
        )
    elif (
        product_type in ("autocallable", "snowball")
        and abs(participation_rate - 1.0) > 1e-9
    ):
        errors.append(
            "autocallable/snowball 产品的 participation_rate 必须为 1.0"
            "（票息结构不含参与率杠杆）"
        )

    principal_protected = data.get("principal_protected", False)
    if not isinstance(principal_protected, bool):
        errors.append(f"principal_protected 必须是布尔值，实际为 {principal_protected!r}")

    target_client = data.get("target_client")
    if not isinstance(target_client, str) or not target_client.strip():
        errors.append("target_client 必须是非空字符串")

    pitch = data.get("pitch", "")
    if not isinstance(pitch, str):
        errors.append(f"pitch 必须是字符串，实际为 {pitch!r}")

    hedging_plan = data.get("hedging_plan", "")
    if not isinstance(hedging_plan, str):
        errors.append(f"hedging_plan 必须是字符串，实际为 {hedging_plan!r}")

    if errors:
        raise ProductError("；".join(errors))

    return ProductSpec(
        product_type=product_type,
        notional=float(notional),
        maturity_months=maturity_months,
        strike_pct=float(strike_pct),
        barrier_pct=float(barrier_pct) if barrier_pct is not None else None,
        barrier_type=barrier_type,
        coupon_rate=float(coupon_rate) if coupon_rate is not None else None,
        participation_rate=float(participation_rate),
        principal_protected=bool(principal_protected),
        target_client=target_client,
        pitch=pitch,
        hedging_plan=hedging_plan,
    )


# 发行政策（全程生效的产品线规则，违反视同约束违规，有一次修改机会）
ISSUANCE_MAX_MATURITY_MONTHS = 12
ISSUANCE_MAX_PARTICIPATION_PROTECTED = 1.5
ISSUANCE_MAX_PARTICIPATION_UNPROTECTED = 3.0


def check_issuance_policy(product: ProductSpec) -> list[tuple[str, str]]:
    """检查发行政策，返回 (违规 id, 原因) 列表；合规返回空列表。"""
    violations: list[tuple[str, str]] = []
    if product.maturity_months > ISSUANCE_MAX_MATURITY_MONTHS:
        violations.append(
            (
                "issuance_maturity_cap",
                f"发行政策：产品期限不得超过 {ISSUANCE_MAX_MATURITY_MONTHS} 个月"
                f"（实际 {product.maturity_months} 个月）",
            )
        )
    cap = (
        ISSUANCE_MAX_PARTICIPATION_PROTECTED
        if product.principal_protected
        else ISSUANCE_MAX_PARTICIPATION_UNPROTECTED
    )
    if product.participation_rate > cap + 1e-9:
        kind = "保本" if product.principal_protected else "非保本"
        violations.append(
            (
                "issuance_participation_cap",
                f"发行政策：{kind}产品参与率不得超过 {cap:g}"
                f"（实际 {product.participation_rate:g}）",
            )
        )
    return violations


def parse_product_draft(data: dict) -> ProductSpec:
    """解析产品草案（用于交易台报价 / 风控预检）：与正式提交同规则，
    但 target_client 可缺省（草案阶段无需指定客户，缺省填 DRAFT_TARGET）。
    """
    if not isinstance(data, dict):
        raise ProductError("产品草案必须是字典（JSON 对象）")
    if not (isinstance(data.get("target_client"), str) and data.get("target_client", "").strip()):
        data = {**data, "target_client": DRAFT_TARGET}
    return parse_product_spec(data)


def check_protection_claim(product: ProductSpec) -> tuple[bool, str]:
    """校验 principal_protected 声明与产品结构是否一致（诚实性检查）。"""
    if product.product_type == "custom":
        return True, "无法校验"
    if product.product_type == "snowball" and product.principal_protected:
        return False, "雪球客户承担敲入下行，不可能保本"
    if (
        product.product_type == "autocallable"
        and product.barrier_pct is not None
        and product.principal_protected
    ):
        return False, "带敲入障碍即不保本"
    return True, "一致"
