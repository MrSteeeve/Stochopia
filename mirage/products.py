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

FUNDING_STYLES = ("premium_paid", "funded_note")


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
    # Funding semantics are deliberately separate from risk notional.  Old
    # callers may omit them; the effective_* properties below provide the
    # deterministic v3 migration defaults.
    funding_style: str | None = None
    face_value: float | None = None
    issue_price_pct: float | None = None
    protected_amount: float | None = None
    # 障碍方向是发行时固定的合约条款。旧调用方可继续省略；解析器会为
    # barrier_call / barrier_put 从发行时 barrier_pct 相对 1.0 的位置推断。
    barrier_direction: str | None = None
    # 以下字段是存续期重估状态，不接受提交 JSON 注入；由环境在发行/演进时维护。
    reference_spot: float | None = None
    barrier_touched: bool = False
    knock_in_active: bool = False
    elapsed_months: int = 0

    @property
    def effective_funding_style(self) -> str:
        """Return the explicit style or the frozen compatibility default."""

        if self.funding_style is not None:
            return self.funding_style
        if (
            self.principal_protected
            or self.product_type in {"autocallable", "snowball"}
        ):
            return "funded_note"
        return "premium_paid"

    @property
    def effective_face_value(self) -> float:
        """Contract redemption base, distinct from exposure notional."""

        return self.notional if self.face_value is None else self.face_value

    @property
    def effective_issue_price_pct(self) -> float | None:
        """Issue price as a face-value fraction for funded notes."""

        if self.effective_funding_style == "funded_note":
            return 1.0 if self.issue_price_pct is None else self.issue_price_pct
        return self.issue_price_pct

    @property
    def effective_protected_amount(self) -> float:
        """Contractually protected redemption amount in currency units."""

        if self.protected_amount is not None:
            return self.protected_amount
        return self.effective_face_value if self.principal_protected else 0.0


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

        cash_outlay = float(pricing.get("cash_outlay", pricing.get("client_price", 0.0)))
        if cash_outlay > self.capital:
            return False, "客户实际现金出资超过可投资金额"

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
    的显式 null 视同缺失，取默认值。reference_spot、barrier_touched、
    knock_in_active、elapsed_months 是环境维护的内部状态，解析提交时始终重置。
    """
    if not isinstance(data, dict):
        raise ProductError("产品规格必须是字典（JSON 对象）")

    data = {
        k: (_NULLABLE_DEFAULTS[k] if k in _NULLABLE_DEFAULTS and v is None else v)
        for k, v in data.items()
    }

    errors: list[str] = []
    allowed_fields = {
        "product_type",
        "notional",
        "maturity_months",
        "strike_pct",
        "barrier_pct",
        "barrier_type",
        "barrier_direction",
        "coupon_rate",
        "participation_rate",
        "principal_protected",
        "target_client",
        "pitch",
        "hedging_plan",
        "funding_style",
        "face_value",
        "issue_price_pct",
        "protected_amount",
        # Environment-maintained legacy fields are accepted only so this
        # parser can deterministically discard them at the external boundary.
        "reference_spot",
        "barrier_touched",
        "knock_in_active",
        "elapsed_months",
    }
    unknown_fields = set(data) - allowed_fields
    if unknown_fields:
        errors.append(f"产品规格包含未知字段：{sorted(unknown_fields)}")

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

    barrier_direction = data.get("barrier_direction")
    is_barrier_product = product_type in ("barrier_call", "barrier_put")
    if barrier_direction is not None and barrier_direction not in ("up", "down"):
        errors.append(
            "barrier_direction 必须为空或 up / down 之一，"
            f"实际为 {barrier_direction!r}"
        )
    elif not is_barrier_product and barrier_direction is not None:
        errors.append("barrier_direction 仅适用于 barrier_call / barrier_put 产品")
    elif is_barrier_product and barrier_direction is None and _is_finite_number(barrier_pct):
        # 发行时方向只推断一次；后续现货跨越障碍也不会翻转方向。
        barrier_direction = "up" if barrier_pct > 1.0 else "down"

    if barrier_direction is not None and barrier_pct is None:
        errors.append("barrier_direction 必须与 barrier_pct 同时设置")

    if not is_barrier_product:
        # 即使提交携带了非法值，也不让它进入非障碍产品的内部状态。
        barrier_direction = None

    coupon_rate = data.get("coupon_rate")
    if coupon_rate is not None and (
        not _is_finite_number(coupon_rate) or coupon_rate < 0 or coupon_rate > 5
    ):
        errors.append(f"coupon_rate 必须为空或 [0, 5] 范围内的有限数字，实际为 {coupon_rate!r}")

    if product_type in ("autocallable", "snowball") and coupon_rate is None:
        errors.append(f"{product_type} 类型必须设置 coupon_rate")

    participation_rate = data.get("participation_rate", 1.0)
    if not _is_finite_number(participation_rate) or participation_rate <= 0 or participation_rate > 10:
        errors.append(
            f"participation_rate 必须是 (0, 10] 范围内的有限数字，实际为 {participation_rate!r}"
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

    inferred_funding_style = (
        "funded_note"
        if principal_protected is True or product_type in {"autocallable", "snowball"}
        else "premium_paid"
    )
    funding_style = data.get("funding_style")
    if funding_style is None:
        funding_style = inferred_funding_style
    if funding_style not in FUNDING_STYLES:
        errors.append(
            f"funding_style 必须是 {FUNDING_STYLES} 之一，实际为 {funding_style!r}"
        )

    face_value = data.get("face_value")
    if face_value is None:
        face_value = notional
    if not _is_finite_number(face_value) or face_value <= 0 or face_value > 1e12:
        errors.append(
            f"face_value 必须是 (0, 1e12] 范围内的有限数字，实际为 {face_value!r}"
        )

    issue_price_pct = data.get("issue_price_pct")
    if issue_price_pct is None and funding_style == "funded_note":
        issue_price_pct = 1.0
    if issue_price_pct is not None and (
        not _is_finite_number(issue_price_pct)
        or issue_price_pct <= 0
        or issue_price_pct > 10
    ):
        errors.append(
            "issue_price_pct 必须为空或 (0, 10] 范围内的有限数字，"
            f"实际为 {issue_price_pct!r}"
        )
    if funding_style == "funded_note" and issue_price_pct is None:
        errors.append("funded_note 必须设置 issue_price_pct")
    if (
        funding_style == "funded_note"
        and _is_finite_number(issue_price_pct)
        and not math.isclose(float(issue_price_pct), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        errors.append(
            "Level-0 funded_note 的 issue_price_pct 固定为 1.0；"
            "价格适配必须通过 participation/coupon solve-for"
        )
    if funding_style == "premium_paid" and issue_price_pct is not None:
        errors.append("premium_paid 的发行价由报价求解，issue_price_pct 必须为空")

    default_protected = (
        face_value
        if principal_protected is True and _is_finite_number(face_value)
        else 0.0
    )
    protected_amount = data.get("protected_amount")
    if protected_amount is None:
        protected_amount = default_protected
    if (
        not _is_finite_number(protected_amount)
        or protected_amount < 0
        or protected_amount > 1e12
    ):
        errors.append(
            "protected_amount 必须是 [0, 1e12] 范围内的有限数字，"
            f"实际为 {protected_amount!r}"
        )
    if principal_protected is True and _is_finite_number(protected_amount):
        if protected_amount <= 0:
            errors.append("principal_protected 产品的 protected_amount 必须为正")
    if principal_protected is False and _is_finite_number(protected_amount):
        if protected_amount > 0:
            errors.append(
                "未声明 principal_protected 的产品不能设置正 protected_amount"
            )
    if (
        _is_finite_number(protected_amount)
        and _is_finite_number(default_protected)
        and not math.isclose(
            float(protected_amount),
            float(default_protected),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        errors.append(
            "Level-0 protected_amount 由确定性 payoff floor 推导，不能由策略指定"
        )

    # Level 0 MC note templates are normalised to one unit of face.  Keeping
    # face and exposure equal here is an explicit boundary until the solve-for
    # payoff DSL supports separate participation notionals.
    if (
        product_type in {"autocallable", "snowball"}
        and _is_finite_number(face_value)
        and _is_finite_number(notional)
        and not math.isclose(float(face_value), float(notional), rel_tol=0.0, abs_tol=1e-9)
    ):
        errors.append(
            "Level-0 autocallable/snowball 要求 face_value 与 notional 相等"
        )

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
        funding_style=funding_style,
        face_value=float(face_value) if _is_finite_number(face_value) else None,
        issue_price_pct=(
            float(issue_price_pct) if _is_finite_number(issue_price_pct) else None
        ),
        protected_amount=(
            float(protected_amount) if _is_finite_number(protected_amount) else None
        ),
        barrier_direction=barrier_direction,
        reference_spot=None,
        barrier_touched=False,
        knock_in_active=False,
        elapsed_months=0,
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
