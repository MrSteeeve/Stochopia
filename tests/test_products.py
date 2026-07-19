"""产品与约束测试：客户决策、约束校验、产品解析与保本声明诚实性检查。"""

from __future__ import annotations

import pytest

from mirage.products import (
    DRAFT_TARGET,
    ClientProfile,
    Constraint,
    ProductError,
    ProductSpec,
    check_protection_claim,
    parse_product_draft,
    parse_product_spec,
)


def _spec(**overrides) -> ProductSpec:
    """构造一个默认合法的 ProductSpec，测试中按需覆盖字段。"""
    base = dict(
        product_type="vanilla_call",
        notional=1_000_000.0,
        maturity_months=6,
        strike_pct=1.0,
        barrier_pct=None,
        barrier_type=None,
        coupon_rate=None,
        participation_rate=1.0,
        principal_protected=False,
        target_client="张先生",
        pitch="",
        hedging_plan="",
    )
    base.update(overrides)
    return ProductSpec(**base)


def _client(**overrides) -> ClientProfile:
    base = dict(
        id="c1",
        name="张先生",
        capital=2_000_000.0,
        max_loss_pct=0.1,
        min_return_pct=0.03,
        risk_appetite="balanced",
        preferences="",
    )
    base.update(overrides)
    return ClientProfile(**base)


# ---------------------------------------------------------------------------
# ClientProfile.would_buy
# ---------------------------------------------------------------------------


def test_would_buy_rejects_unprotected_for_conservative():
    """保守型客户遇到不保本产品，第一条规则即拒绝。"""
    client = _client(risk_appetite="conservative")
    product = _spec(principal_protected=False)
    pricing = {"client_price": 10_000.0, "expected_payoff": 50_000.0, "loss_frac": 0.05}
    ok, reason = client.would_buy(product, pricing)
    assert ok is False
    assert "不保本" in reason


def test_would_buy_conservative_accepts_protected_product():
    """保守型客户遇到保本产品，且其余条件满足，应通过第一条检查继续往下判断并最终接受。"""
    client = _client(risk_appetite="conservative", capital=2_000_000.0, max_loss_pct=0.1, min_return_pct=0.02)
    product = _spec(principal_protected=True, notional=1_000_000.0, maturity_months=12)
    pricing = {"client_price": 10_000.0, "loss_frac": 0.0, "hurdle_hit_prob": 0.8}
    ok, reason = client.would_buy(product, pricing)
    assert ok is True
    assert reason == "符合投资标准"


def test_would_buy_rejects_notional_exceeds_capital():
    """名义本金超过客户可投资金额，第二条规则拒绝。"""
    client = _client(capital=500_000.0)
    product = _spec(principal_protected=True, notional=1_000_000.0)
    pricing = {"client_price": 10_000.0, "expected_payoff": 50_000.0, "loss_frac": 0.0}
    ok, reason = client.would_buy(product, pricing)
    assert ok is False
    assert "名义本金" in reason


def test_would_buy_rejects_loss_frac_exceeds_tolerance():
    """潜在损失超过客户承受范围，第三条规则拒绝，原因中带两个百分数。"""
    client = _client(max_loss_pct=0.1)
    product = _spec(notional=1_000_000.0)
    pricing = {"client_price": 10_000.0, "expected_payoff": 50_000.0, "loss_frac": 0.2}
    ok, reason = client.would_buy(product, pricing)
    assert ok is False
    assert "潜在损失" in reason
    assert "20" in reason
    assert "10" in reason


def test_would_buy_rejects_low_hit_prob():
    """实现客户期望年化的把握不足时拒绝，理由用定性口吻（不报小数概率）。"""
    client = _client(min_return_pct=0.05, min_hit_prob=0.6)
    product = _spec(notional=1_000_000.0, maturity_months=12)
    pricing = {"client_price": 10_000.0, "loss_frac": 0.0, "hurdle_hit_prob": 0.25}
    ok, reason = client.would_buy(product, pricing)
    assert ok is False
    assert "年化 5.0%" in reason
    assert "把握较小" in reason
    # 概率只以定性表述出现，门槛数字之外不再出现百分号
    assert reason.count("%") == 1


def test_would_buy_accepts_when_all_checks_pass():
    """所有检查均通过时接受，理由为标准通过语。"""
    client = _client(
        risk_appetite="balanced", capital=2_000_000.0, max_loss_pct=0.2, min_return_pct=0.02
    )
    product = _spec(notional=1_000_000.0, maturity_months=6)
    pricing = {"client_price": 10_000.0, "loss_frac": 0.05, "hurdle_hit_prob": 0.75}
    ok, reason = client.would_buy(product, pricing)
    assert ok is True
    assert reason == "符合投资标准"


# ---------------------------------------------------------------------------
# Constraint.validate
# ---------------------------------------------------------------------------


def _constraint(**overrides) -> Constraint:
    base = dict(
        id="con1",
        activate_round=1,
        source="监管",
        description="delta 不得超限",
        check_field="delta",
        check_type="max",
        check_value=0.5,
    )
    base.update(overrides)
    return Constraint(**base)


def test_constraint_max_pass_and_fail():
    con = _constraint(check_type="max", check_value=0.5, check_field="delta")
    product = _spec()
    ok, _ = con.validate(product, {"delta": 0.5})
    assert ok is True
    ok, reason = con.validate(product, {"delta": 0.6})
    assert ok is False
    assert "delta" in reason
    assert "0.6" in reason


def test_constraint_min_pass_and_fail():
    con = _constraint(check_type="min", check_value=0.2, check_field="vega")
    product = _spec()
    ok, _ = con.validate(product, {"vega": 0.2})
    assert ok is True
    ok, reason = con.validate(product, {"vega": 0.1})
    assert ok is False
    assert "vega" in reason


def test_constraint_max_abs_pass_and_fail():
    con = _constraint(check_type="max_abs", check_value=0.3, check_field="gamma")
    product = _spec()
    ok, _ = con.validate(product, {"gamma": -0.3})
    assert ok is True
    ok, reason = con.validate(product, {"gamma": -0.4})
    assert ok is False
    assert "gamma" in reason


def test_constraint_forbidden_type_pass_and_fail():
    con = _constraint(check_type="forbidden_type", check_value="snowball", check_field="product_type")
    ok, _ = con.validate(_spec(product_type="vanilla_call"), {})
    assert ok is True
    ok, reason = con.validate(_spec(product_type="snowball", coupon_rate=0.1), {})
    assert ok is False
    assert "snowball" in reason


def test_constraint_resolves_field_from_greeks_over_product():
    """check_field 若存在于 greeks 中，优先取 greeks 的值。"""
    con = _constraint(check_type="max", check_value=100.0, check_field="notional")
    product = _spec(notional=1_000_000.0)
    # greeks 里给出一个更小的 notional 值，应该被采用而不是 product.notional
    ok, _ = con.validate(product, {"notional": 50.0})
    assert ok is True


def test_constraint_resolves_field_from_product_when_absent_in_greeks():
    con = _constraint(check_type="max", check_value=2_000_000.0, check_field="notional")
    product = _spec(notional=1_000_000.0)
    ok, _ = con.validate(product, {})
    assert ok is True


def test_constraint_unknown_field_raises():
    con = _constraint(check_field="nonexistent_field")
    product = _spec()
    with pytest.raises(ProductError):
        con.validate(product, {})


def test_constraint_unknown_check_type_raises():
    con = _constraint(check_type="weird_type")
    product = _spec()
    with pytest.raises(ProductError):
        con.validate(product, {"delta": 0.1})


# ---------------------------------------------------------------------------
# parse_product_spec
# ---------------------------------------------------------------------------


def _valid_payload(**overrides) -> dict:
    base = dict(
        product_type="vanilla_call",
        notional=1_000_000,
        maturity_months=6,
        strike_pct=1.0,
        target_client="张先生",
    )
    base.update(overrides)
    return base


def test_parse_product_spec_happy_path_with_defaults():
    """最小合法输入，缺省字段应被填上文档规定的默认值。"""
    spec = parse_product_spec(_valid_payload())
    assert spec.product_type == "vanilla_call"
    assert spec.notional == 1_000_000.0
    assert spec.maturity_months == 6
    assert spec.barrier_pct is None
    assert spec.barrier_type is None
    assert spec.coupon_rate is None
    assert spec.participation_rate == 1.0
    assert spec.principal_protected is False
    assert spec.pitch == ""
    assert spec.hedging_plan == ""


def test_parse_product_spec_accepts_integral_float_maturity():
    spec = parse_product_spec(_valid_payload(maturity_months=12.0))
    assert spec.maturity_months == 12
    assert isinstance(spec.maturity_months, int)


def test_parse_product_spec_aggregates_multiple_errors():
    """多个字段同时非法时，报错信息应把所有问题以分号拼接在一起。"""
    payload = _valid_payload(
        product_type="not_a_type",
        notional=-5,
        maturity_months=100,
        strike_pct=-1,
        target_client="",
    )
    with pytest.raises(ProductError) as exc_info:
        parse_product_spec(payload)
    msg = str(exc_info.value)
    assert "product_type" in msg
    assert "notional" in msg
    assert "maturity_months" in msg
    assert "strike_pct" in msg
    assert "target_client" in msg
    assert msg.count("；") >= 4


def test_parse_product_spec_barrier_both_or_neither():
    """barrier_pct 与 barrier_type 必须同时设置或同时为空。"""
    with pytest.raises(ProductError, match="barrier_pct"):
        parse_product_spec(_valid_payload(product_type="barrier_call", barrier_pct=1.1))
    with pytest.raises(ProductError, match="barrier_pct"):
        parse_product_spec(_valid_payload(product_type="barrier_call", barrier_type="knock_in"))
    # 两者都设置时应当成功
    spec = parse_product_spec(
        _valid_payload(product_type="barrier_call", barrier_pct=1.1, barrier_type="knock_in")
    )
    assert spec.barrier_pct == 1.1
    assert spec.barrier_type == "knock_in"


def test_parse_product_spec_autocallable_needs_coupon():
    """autocallable / snowball 类型必须提供 coupon_rate。"""
    with pytest.raises(ProductError, match="coupon_rate"):
        parse_product_spec(_valid_payload(product_type="autocallable"))
    spec = parse_product_spec(_valid_payload(product_type="autocallable", coupon_rate=0.08))
    assert spec.coupon_rate == 0.08

    with pytest.raises(ProductError, match="coupon_rate"):
        parse_product_spec(_valid_payload(product_type="snowball"))
    spec = parse_product_spec(_valid_payload(product_type="snowball", coupon_rate=0.12))
    assert spec.coupon_rate == 0.12


def test_parse_product_spec_rejects_non_dict():
    with pytest.raises(ProductError):
        parse_product_spec("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_product_spec：非有限数字 / 越界数值不得导致异常逃逸
# ---------------------------------------------------------------------------


def test_parse_product_spec_huge_maturity_months_rejected_without_overflow_error():
    """超出 float 范围的 maturity_months（字面量退化为 inf）应报 ProductError，
    而不是让 int() 转换抛出的 OverflowError 逃逸。"""
    with pytest.raises(ProductError, match="maturity_months"):
        parse_product_spec(_valid_payload(maturity_months=1e309))


@pytest.mark.parametrize("bad_strike", [1e309, float("inf"), float("nan")])
def test_parse_product_spec_non_finite_strike_pct_rejected(bad_strike):
    with pytest.raises(ProductError, match="strike_pct"):
        parse_product_spec(_valid_payload(strike_pct=bad_strike))


def test_parse_product_spec_notional_exceeds_upper_bound():
    with pytest.raises(ProductError, match="notional"):
        parse_product_spec(_valid_payload(notional=1e13))


def test_parse_product_spec_snowball_participation_must_be_one():
    with pytest.raises(ProductError, match="participation_rate"):
        parse_product_spec(
            _valid_payload(product_type="snowball", coupon_rate=0.12, participation_rate=2.0)
        )

    spec = parse_product_spec(
        _valid_payload(product_type="snowball", coupon_rate=0.12, participation_rate=1.0)
    )
    assert spec.participation_rate == 1.0


# ---------------------------------------------------------------------------
# check_protection_claim
# ---------------------------------------------------------------------------


def test_check_protection_claim_snowball_protected_is_inconsistent():
    product = _spec(product_type="snowball", coupon_rate=0.1, principal_protected=True)
    ok, reason = check_protection_claim(product)
    assert ok is False
    assert "雪球" in reason


def test_check_protection_claim_autocall_with_knock_in_protected_is_inconsistent():
    product = _spec(
        product_type="autocallable",
        coupon_rate=0.1,
        barrier_pct=0.8,
        barrier_type="knock_in",
        principal_protected=True,
    )
    ok, reason = check_protection_claim(product)
    assert ok is False
    assert "敲入" in reason


def test_check_protection_claim_protected_vanilla_is_consistent():
    product = _spec(product_type="vanilla_call", principal_protected=True)
    ok, reason = check_protection_claim(product)
    assert ok is True
    assert reason == "一致"


def test_check_protection_claim_custom_cannot_be_checked():
    product = _spec(product_type="custom", principal_protected=True)
    ok, reason = check_protection_claim(product)
    assert ok is True
    assert reason == "无法校验"


# ---------------------------------------------------------------------------
# parse_product_spec：显式 null 处理（TASK 4）
# ---------------------------------------------------------------------------


def test_parse_product_spec_explicit_null_optional_fields_get_defaults():
    """显式 JSON null 的可选字段应退化为默认值，不报错。"""
    payload = _valid_payload(
        participation_rate=None,
        principal_protected=None,
        pitch=None,
        hedging_plan=None,
    )
    spec = parse_product_spec(payload)
    assert spec.participation_rate == 1.0
    assert spec.principal_protected is False
    assert spec.pitch == ""
    assert spec.hedging_plan == ""


def test_parse_product_spec_null_target_client_still_errors():
    """target_client 不是可选字段，显式 null 应报 ProductError。"""
    payload = _valid_payload(target_client=None)
    with pytest.raises(ProductError, match="target_client"):
        parse_product_spec(payload)


# ---------------------------------------------------------------------------
# parse_product_draft（TASK 4）
# ---------------------------------------------------------------------------


def test_parse_product_draft_without_target_client_uses_draft_target():
    """草案中不提供 target_client 时，应自动填充 DRAFT_TARGET。"""
    payload = _valid_payload()
    del payload["target_client"]
    spec = parse_product_draft(payload)
    assert spec.target_client == DRAFT_TARGET


def test_parse_product_draft_with_target_client_keeps_it():
    """草案中提供了 target_client 时，应保留原值。"""
    payload = _valid_payload(target_client="client_b")
    spec = parse_product_draft(payload)
    assert spec.target_client == "client_b"


def test_parse_product_draft_null_target_client_uses_draft_target():
    """草案中 target_client 为 null 时，视同缺失，填 DRAFT_TARGET。"""
    payload = _valid_payload(target_client=None)
    spec = parse_product_draft(payload)
    assert spec.target_client == DRAFT_TARGET


def test_parse_product_draft_invalid_fields_still_raise():
    """草案有非法字段时，仍应报 ProductError。"""
    payload = dict(
        product_type="invalid_type",
        notional=-999,
        maturity_months=6,
        strike_pct=1.0,
    )
    with pytest.raises(ProductError):
        parse_product_draft(payload)
