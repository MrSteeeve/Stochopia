"""产品与约束测试：客户决策、约束校验、产品解析与保本声明诚实性检查。"""

from __future__ import annotations

import pytest

from stochopia.products import (
    DRAFT_TARGET,
    ClientProfile,
    ProductError,
    ProductSpec,
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
    assert spec.barrier_direction is None
    assert spec.coupon_rate is None
    assert spec.participation_rate == 1.0
    assert spec.principal_protected is False
    assert spec.pitch == ""
    assert spec.hedging_plan == ""
    assert spec.reference_spot is None
    assert spec.barrier_touched is False
    assert spec.knock_in_active is False
    assert spec.elapsed_months == 0


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
    assert spec.barrier_direction == "up"


def test_parse_product_spec_barrier_direction_is_fixed_at_issuance():
    """障碍方向缺省时按发行时 barrier_pct 相对 1.0 推断，显式方向则保留。"""
    down = parse_product_spec(
        _valid_payload(
            product_type="barrier_put",
            barrier_pct=0.8,
            barrier_type="knock_in",
        )
    )
    explicit = parse_product_spec(
        _valid_payload(
            product_type="barrier_call",
            barrier_pct=0.8,
            barrier_type="knock_out",
            barrier_direction="up",
        )
    )

    assert down.barrier_direction == "down"
    assert explicit.barrier_direction == "up"


@pytest.mark.parametrize("bad_direction", ["left", 1, True])
def test_parse_product_spec_rejects_invalid_barrier_direction(bad_direction):
    with pytest.raises(ProductError, match="barrier_direction"):
        parse_product_spec(
            _valid_payload(
                product_type="barrier_call",
                barrier_pct=1.2,
                barrier_type="knock_out",
                barrier_direction=bad_direction,
            )
        )


def test_parse_product_spec_rejects_barrier_direction_on_non_barrier_product():
    with pytest.raises(ProductError, match="barrier_direction"):
        parse_product_spec(_valid_payload(barrier_direction="up"))


def test_parse_product_spec_autocallable_needs_coupon():
    """autocallable / snowball 类型必须提供 coupon_rate。"""
    with pytest.raises(ProductError, match="coupon_rate"):
        parse_product_spec(
            _valid_payload(product_type="autocallable", principal_protected=True)
        )
    spec = parse_product_spec(
        _valid_payload(
            product_type="autocallable",
            coupon_rate=0.08,
            principal_protected=True,
        )
    )
    assert spec.coupon_rate == 0.08

    with pytest.raises(ProductError, match="coupon_rate"):
        parse_product_spec(_valid_payload(product_type="snowball"))
    spec = parse_product_spec(_valid_payload(product_type="snowball", coupon_rate=0.12))
    assert spec.coupon_rate == 0.12


def test_parse_product_spec_autocallable_requires_coherent_loss_semantics():
    with pytest.raises(ProductError, match="非保本 autocallable"):
        parse_product_spec(
            _valid_payload(
                product_type="autocallable",
                coupon_rate=0.08,
                principal_protected=False,
            )
        )

    with pytest.raises(ProductError, match="保本 autocallable"):
        parse_product_spec(
            _valid_payload(
                product_type="autocallable",
                coupon_rate=0.08,
                principal_protected=True,
                barrier_pct=0.85,
                barrier_type="knock_in",
            )
        )

    spec = parse_product_spec(
        _valid_payload(
            product_type="autocallable",
            coupon_rate=0.08,
            principal_protected=False,
            barrier_pct=0.85,
            barrier_type="knock_in",
        )
    )
    assert spec.principal_protected is False
    assert spec.barrier_pct == pytest.approx(0.85)


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


@pytest.mark.parametrize("bad_participation", [0, 0.0, -0.1])
def test_parse_product_spec_participation_must_be_strictly_positive(
    bad_participation,
):
    with pytest.raises(ProductError, match=r"participation_rate .+\(0, 10\]"):
        parse_product_spec(
            _valid_payload(participation_rate=bad_participation)
        )


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


def test_parse_product_spec_ignores_untrusted_internal_state_fields():
    """提交 JSON 不得伪造存续期定盘价、历史敲入或已存续时间。"""
    spec = parse_product_spec(
        _valid_payload(
            reference_spot=999.0,
            barrier_touched=True,
            knock_in_active=True,
            elapsed_months=5,
        )
    )

    assert spec.reference_spot is None
    assert spec.barrier_touched is False
    assert spec.knock_in_active is False
    assert spec.elapsed_months == 0


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


@pytest.mark.parametrize("bad_type", [["vanilla_call"], {"x": 1}, 3])
def test_parse_product_spec_rejects_non_string_product_type(bad_type):
    """An LLM emitting a list/object product_type is a ProductError, not a TypeError."""
    with pytest.raises(ProductError, match="product_type"):
        parse_product_spec(_valid_payload(product_type=bad_type))
