"""定价引擎测试：BS 已知值、看涨看跌平价、希腊值、障碍期权 in-out 平价、
蒙特卡洛雪球/自动赎回定价、期望收益与产品级评估。"""

from __future__ import annotations

import math

import pytest

import inspect
import mirage.pricing as pricing_module

from mirage.products import (
    MarketState,
    ProductSpec,
    parse_product_spec,
)

from mirage.pricing import (
    PricingError,
    autocallable_price,
    bs_call,
    bs_greeks,
    bs_put,
    barrier_option,
    best_case_line,
    build_payoff_table,
    evaluate_product,
    expected_payoff,
    hurdle_hit_prob,
    mc_diagnostics,
    price_product,
    snowball_price,
)


def _market(**overrides) -> MarketState:
    base = dict(
        round_num=1,
        index_name="SSE50",
        spot=100.0,
        volatility=0.2,
        risk_free_rate=0.03,
        dividend_yield=0.0,
        recent_trend="flat",
        vix_level="low",
        trend_alpha=0.0,
    )
    base.update(overrides)
    return MarketState(**base)


def _product(**overrides) -> ProductSpec:
    base = dict(
        product_type="vanilla_call",
        notional=1_000_000.0,
        maturity_months=12,
        strike_pct=1.0,
        barrier_pct=None,
        barrier_type=None,
        coupon_rate=None,
        participation_rate=1.0,
        principal_protected=False,
        target_client="retail",
        pitch="示例产品",
        hedging_plan="delta 对冲",
    )
    base.update(overrides)
    return ProductSpec(**base)


# ---------------------------------------------------------------------------
# Black-Scholes 已知值 / 平价
# ---------------------------------------------------------------------------


def test_bs_known_values():
    """标准参数下的 BS 期权价格与公认参考值一致。"""
    call = bs_call(100, 100, 1, 0.05, 0.2, 0.0)
    put = bs_put(100, 100, 1, 0.05, 0.2, 0.0)
    assert call == pytest.approx(10.4506, abs=1e-3)
    assert put == pytest.approx(5.5735, abs=1e-3)


@pytest.mark.parametrize(
    "S,K,T,r,sigma,q",
    [
        (100, 100, 1, 0.05, 0.2, 0.0),
        (100, 110, 0.5, 0.03, 0.3, 0.01),
        (80, 100, 2.0, 0.02, 0.15, 0.02),
        (120, 100, 0.25, 0.04, 0.4, 0.0),
    ],
)
def test_put_call_parity(S, K, T, r, sigma, q):
    """看涨看跌平价：C - P = S*exp(-qT) - K*exp(-rT)。"""
    call = bs_call(S, K, T, r, sigma, q)
    put = bs_put(S, K, T, r, sigma, q)
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-8)


def test_bs_call_put_intrinsic_at_expiry():
    """T=0 时退化为内在价值。"""
    assert bs_call(110, 100, 0, 0.05, 0.2, 0.0) == pytest.approx(10.0)
    assert bs_put(90, 100, 0, 0.05, 0.2, 0.0) == pytest.approx(10.0)
    assert bs_call(90, 100, 0, 0.05, 0.2, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 希腊值
# ---------------------------------------------------------------------------


def test_bs_greeks_call_put_ranges():
    """delta 区间、call-put delta 差、gamma/vega 符号。"""
    S, K, T, r, sigma, q = 100, 100, 1, 0.05, 0.2, 0.01
    call_g = bs_greeks(S, K, T, r, sigma, q, "call")
    put_g = bs_greeks(S, K, T, r, sigma, q, "put")

    assert 0.0 < call_g["delta"] < 1.0
    assert -1.0 < put_g["delta"] < 0.0
    assert call_g["delta"] - put_g["delta"] == pytest.approx(math.exp(-q * T), abs=1e-8)
    assert call_g["gamma"] > 0.0
    assert put_g["gamma"] > 0.0
    assert call_g["gamma"] == pytest.approx(put_g["gamma"], abs=1e-10)
    assert call_g["vega"] > 0.0
    assert put_g["vega"] > 0.0


def test_bs_greeks_degenerate_at_expiry():
    """T<=0 或 sigma<=0 时希腊值退化为 0。"""
    g = bs_greeks(100, 100, 0, 0.05, 0.2, 0.0, "call")
    assert g == {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}


def test_bs_greeks_unknown_option_type_raises():
    from mirage.pricing import PricingError

    with pytest.raises(PricingError):
        bs_greeks(100, 100, 1, 0.05, 0.2, 0.0, "straddle")


# ---------------------------------------------------------------------------
# 障碍期权：in-out 平价、knock-out <= vanilla、已触碰边界
# ---------------------------------------------------------------------------


_BARRIER_CONFIGS = [
    ("call", 80.0),
    ("call", 120.0),
    ("put", 80.0),
    ("put", 120.0),
]


@pytest.mark.parametrize("option_type,barrier", _BARRIER_CONFIGS)
@pytest.mark.parametrize(
    "S,K",
    [
        (100.0, 100.0),  # K on the "far" side for one direction, "near" for the other
        (100.0, 70.0),
        (100.0, 130.0),
    ],
)
def test_barrier_in_out_parity(option_type, barrier, S, K):
    """8 种 up/down x in/out x call/put 组合（以及不同 K 相对障碍位置）均满足 KI+KO=vanilla。"""
    T, r, sigma, q = 1.0, 0.05, 0.2, 0.0
    direction = "down" if barrier < S else "up"
    ki = barrier_option(
        S, K, T, r, sigma, q, barrier, "knock_in", option_type, direction
    )
    ko = barrier_option(
        S, K, T, r, sigma, q, barrier, "knock_out", option_type, direction
    )
    vanilla = bs_call(S, K, T, r, sigma, q) if option_type == "call" else bs_put(S, K, T, r, sigma, q)
    assert abs(ki + ko - vanilla) < 1e-8


@pytest.mark.parametrize("option_type,barrier", _BARRIER_CONFIGS)
def test_barrier_knockout_leq_vanilla(option_type, barrier):
    """敲出期权价格不超过对应香草期权价格。"""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    direction = "down" if barrier < S else "up"
    ko = barrier_option(
        S, K, T, r, sigma, q, barrier, "knock_out", option_type, direction
    )
    vanilla = bs_call(S, K, T, r, sigma, q) if option_type == "call" else bs_put(S, K, T, r, sigma, q)
    assert ko <= vanilla + 1e-12


@pytest.mark.parametrize("direction", ["down", "up"])
def test_barrier_equal_spot_treated_as_breached(direction):
    """barrier == S 视为已经触碰。"""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    vanilla = bs_call(S, K, T, r, sigma, q)
    ki = barrier_option(
        S, K, T, r, sigma, q, S, "knock_in", "call", direction
    )
    ko = barrier_option(
        S, K, T, r, sigma, q, S, "knock_out", "call", direction
    )
    assert ki == pytest.approx(vanilla, abs=1e-10)
    assert ko == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    "direction,S,barrier",
    [("up", 130.0, 120.0), ("down", 70.0, 80.0)],
)
def test_barrier_crossing_uses_fixed_issuance_direction(direction, S, barrier):
    """重估现货跨越固定上/下障碍后必须识别为已触碰，不能按当前 S 反转方向。"""
    K, T, r, sigma, q = 100.0, 1.0, 0.05, 0.2, 0.0
    vanilla = bs_call(S, K, T, r, sigma, q)

    ki = barrier_option(
        S, K, T, r, sigma, q, barrier, "knock_in", "call", direction
    )
    ko = barrier_option(
        S, K, T, r, sigma, q, barrier, "knock_out", "call", direction
    )

    assert ki == pytest.approx(vanilla)
    assert ko == pytest.approx(0.0)


def test_barrier_option_historical_touch_is_inherited():
    """当前现货虽回到障碍内侧，历史已触碰仍使 knock-out 永久失效。"""
    ko = barrier_option(
        100.0,
        100.0,
        1.0,
        0.03,
        0.2,
        0.0,
        120.0,
        "knock_out",
        "call",
        "up",
        already_touched=True,
    )
    assert ko == 0.0


def test_barrier_option_rejects_unknown_direction():
    with pytest.raises(PricingError, match="障碍方向"):
        barrier_option(
            100.0,
            100.0,
            1.0,
            0.03,
            0.2,
            0.0,
            120.0,
            "knock_out",
            "call",
            "sideways",
        )


# ---------------------------------------------------------------------------
# 蒙特卡洛：确定性、雪球定价合理性
# ---------------------------------------------------------------------------


def test_mc_determinism():
    """相同参数（含 seed）的两次调用结果完全一致。"""
    r1 = snowball_price(100, 12, 0.03, 0.2, 0.0, 0.12, n_paths=2000, seed=42)
    r2 = snowball_price(100, 12, 0.03, 0.2, 0.0, 0.12, n_paths=2000, seed=42)
    assert r1 == r2

    a1 = autocallable_price(100, 12, 0.03, 0.2, 0.0, 0.1, barrier_pct=0.75, n_paths=2000, seed=7)
    a2 = autocallable_price(100, 12, 0.03, 0.2, 0.0, 0.1, barrier_pct=0.75, n_paths=2000, seed=7)
    assert a1 == a2


def test_snowball_revaluation_inherits_knock_in_and_reference_spot():
    """已敲入雪球在现货跌至期初 80% 后重估，不得重置历史敲入或改用当前现货归一化。"""
    result = snowball_price(
        80.0,
        12,
        0.0,
        0.0,
        0.0,
        0.12,
        knock_in_pct=0.75,
        knock_out_pct=10.0,
        n_paths=16,
        seed=7,
        fixing_ratio=100.0 / 80.0,
        already_knocked_in=True,
        elapsed_months=6,
    )

    assert result["fair_value"] == pytest.approx(0.8)
    assert result["knock_in_prob"] == pytest.approx(1.0)
    assert result["expected_life_months"] == pytest.approx(12.0)


def test_snowball_revaluation_coupon_uses_total_elapsed_life():
    """剩余只模拟 6 个月，但到期票息仍按合约总存续 12 个月计提。"""
    result = snowball_price(
        100.0,
        12,
        0.0,
        0.0,
        0.0,
        0.12,
        knock_in_pct=0.5,
        knock_out_pct=10.0,
        n_paths=8,
        fixing_ratio=1.0,
        elapsed_months=6,
    )

    assert result["fair_value"] == pytest.approx(1.12)


def test_autocall_revaluation_coupon_counts_elapsed_months_before_future_call():
    """已存续 6 个月后下一月赎回，票息应按总计 7 个月计提。"""
    result = autocallable_price(
        110.0,
        12,
        0.0,
        0.0,
        0.0,
        0.12,
        barrier_pct=0.75,
        autocall_pct=1.05,
        n_paths=8,
        fixing_ratio=100.0 / 110.0,
        elapsed_months=6,
    )

    assert result["fair_value"] == pytest.approx(1.0 + 0.12 * 7.0 / 12.0)
    assert result["expected_life_months"] == pytest.approx(7.0)


def test_snowball_fair_value_plausible_band():
    """标准参数下雪球公允价值（占名义本金比例）应落在合理区间。"""
    result = snowball_price(100, 12, 0.03, 0.2, 0.0, 0.12, n_paths=2000, seed=42)
    assert 0.8 < result["fair_value"] < 1.2


def test_snowball_knock_in_prob_rises_with_sigma():
    """波动率越高，敲入概率越高。"""
    low = snowball_price(100, 12, 0.03, 0.1, 0.0, 0.12, n_paths=2000, seed=42)
    high = snowball_price(100, 12, 0.03, 0.4, 0.0, 0.12, n_paths=2000, seed=42)
    assert high["knock_in_prob"] >= low["knock_in_prob"]


def test_snowball_deep_otm_barrier_matches_coupon_bond():
    """障碍位设置到几乎不可能触及/触碰时，雪球退化为纯票息债券的贴现现值。"""
    result = snowball_price(
        100, 12, 0.03, 0.2, 0.0, 0.12,
        knock_in_pct=0.01, knock_out_pct=10.0, n_paths=2000, seed=42,
    )
    expected_bond = (1.0 + 0.12 * 1.0) * math.exp(-0.03 * 1.0)
    assert result["fair_value"] == pytest.approx(expected_bond, abs=1e-6)
    assert result["knock_in_prob"] == 0.0
    assert result["knock_out_prob"] == 0.0


def test_mc_runtime_under_one_second():
    """10000 条路径的雪球定价应在 1 秒内完成。"""
    import time

    t0 = time.time()
    snowball_price(100, 12, 0.03, 0.2, 0.0, 0.12, n_paths=10000, seed=42)
    elapsed = time.time() - t0
    assert elapsed < 1.0


# ---------------------------------------------------------------------------
# expected_payoff
# ---------------------------------------------------------------------------


def test_expected_payoff_vanilla_call_trend_alpha_increases_payoff():
    """趋势 alpha>0 时，真实世界期望收益应超过 alpha=0（风险中性漂移）时的值。"""
    product = _product(product_type="vanilla_call")
    market0 = _market(trend_alpha=0.0)
    market_alpha = _market(trend_alpha=0.05)

    e0 = expected_payoff(product, market0)
    e_alpha = expected_payoff(product, market_alpha)
    assert e_alpha > e0


def test_expected_payoff_custom_is_none():
    product = _product(product_type="custom")
    assert expected_payoff(product, _market()) is None


def test_expected_payoff_barrier_and_snowball_run():
    """障碍与雪球类型的 expected_payoff 都应返回有限正数。"""
    barrier = _product(
        product_type="barrier_call", barrier_pct=1.2, barrier_type="knock_out"
    )
    snowball = _product(
        product_type="snowball", barrier_pct=0.75, barrier_type="knock_in", coupon_rate=0.12
    )
    market = _market()
    eb = expected_payoff(barrier, market)
    es = expected_payoff(snowball, market)
    assert eb is not None and eb >= 0.0
    assert es is not None and es > 0.0


# ---------------------------------------------------------------------------
# price_product / evaluate_product
# ---------------------------------------------------------------------------


def test_price_product_custom_is_none():
    product = _product(product_type="custom")
    assert price_product(product, _market()) is None


def test_evaluate_product_custom_is_none():
    product = _product(product_type="custom")
    assert evaluate_product(product, _market()) is None


def test_price_product_revaluation_anchors_strike_to_reference_spot():
    """存续期 vanilla 的行权价锚定发行定盘价，且只使用剩余期限定价。"""
    product = _product(
        product_type="vanilla_call",
        maturity_months=12,
        reference_spot=100.0,
        elapsed_months=6,
    )
    market = _market(spot=120.0)

    result = price_product(product, market)
    expected_points = bs_call(120.0, 100.0, 0.5, 0.03, 0.2, 0.0)
    expected_fair_value = product.notional / 100.0 * expected_points
    raw_delta = bs_greeks(120.0, 100.0, 0.5, 0.03, 0.2, 0.0, "call")[
        "delta"
    ]

    assert result["fair_value"] == pytest.approx(expected_fair_value)
    assert result["greeks"]["delta"] == pytest.approx(120.0 / 100.0 * raw_delta)
    assert result["greeks"]["delta"] * product.notional == pytest.approx(
        product.notional * 120.0 / 100.0 * raw_delta
    )


@pytest.mark.parametrize(
    "product",
    [
        _product(product_type="vanilla_call", reference_spot=100.0),
        _product(
            product_type="barrier_call",
            barrier_pct=2.0,
            barrier_type="knock_out",
            barrier_direction="up",
            reference_spot=100.0,
        ),
    ],
)
def test_revaluation_delta_matches_one_percent_fair_value_bump(product):
    """vanilla/barrier delta 是下游可直接乘 N 的 dollar-delta 占比。"""
    market = _market(spot=120.0)
    bump = 0.01 * market.spot
    result = price_product(product, market)
    fair_up = price_product(product, _market(spot=market.spot + bump))[
        "fair_value"
    ]
    fair_down = price_product(product, _market(spot=market.spot - bump))[
        "fair_value"
    ]
    bumped_delta = (fair_up - fair_down) / (0.02 * product.notional)

    assert result["greeks"]["delta"] == pytest.approx(bumped_delta, rel=2e-3)


@pytest.mark.parametrize(
    "direction,spot,barrier_pct,product_type",
    [
        ("up", 130.0, 1.2, "barrier_call"),
        ("down", 70.0, 0.8, "barrier_put"),
    ],
)
def test_product_revaluation_detects_crossed_fixed_barrier(
    direction, spot, barrier_pct, product_type
):
    product = _product(
        product_type=product_type,
        barrier_pct=barrier_pct,
        barrier_type="knock_out",
        barrier_direction=direction,
        reference_spot=100.0,
    )
    market = _market(spot=spot)

    priced = price_product(product, market)
    expected = expected_payoff(product, market)

    assert priced["fair_value"] == pytest.approx(0.0)
    assert expected == pytest.approx(0.0)


def test_stateful_snowball_flows_through_price_expected_and_diagnostics():
    """reference_spot / knock_in_active / elapsed_months 在三条 MC 路径中必须保持一致。"""
    product = _product(
        product_type="snowball",
        barrier_pct=0.75,
        barrier_type="knock_in",
        coupon_rate=0.12,
        reference_spot=100.0,
        knock_in_active=True,
        elapsed_months=6,
    )
    market = _market(
        spot=80.0,
        volatility=0.0,
        risk_free_rate=0.0,
        trend_alpha=0.0,
    )

    priced = price_product(product, market)
    expected = expected_payoff(product, market)
    diag = mc_diagnostics(product, market, n_paths=32, seed=11)

    assert priced["fair_value"] == pytest.approx(0.8 * product.notional)
    assert expected == pytest.approx(0.8 * product.notional)
    assert diag.pv_mean_frac == pytest.approx(0.8)
    assert diag.event_probs["knock_in"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(product_type="vanilla_call"),
        dict(product_type="vanilla_put"),
        dict(product_type="barrier_call", barrier_pct=1.2, barrier_type="knock_out"),
        dict(
            product_type="snowball",
            barrier_pct=0.75,
            barrier_type="knock_in",
            coupon_rate=0.12,
        ),
        dict(
            product_type="autocallable",
            barrier_pct=0.75,
            barrier_type="knock_in",
            coupon_rate=0.1,
        ),
    ],
)
def test_evaluate_product_returns_required_keys(kwargs):
    """evaluate_product 对所有非 custom 类型都返回完整字段。"""
    product = _product(**kwargs)
    result = evaluate_product(product, _market())
    assert result is not None
    for key in (
        "fair_value",
        "greeks",
        "hedging_cost",
        "pricing_details",
        "client_price",
        "expected_payoff",
        "loss_frac",
    ):
        assert key in result
    for key in ("delta", "gamma", "vega", "theta", "vega_pct"):
        assert key in result["greeks"]
    assert result["client_price"] == pytest.approx(1.03 * result["fair_value"])
    assert result["hedging_cost"] == pytest.approx(1.02 * result["fair_value"])


def test_evaluate_product_loss_frac_principal_protected_vanilla():
    """保本产品的 loss_frac 恒为 0。"""
    product = _product(product_type="vanilla_call", principal_protected=True, participation_rate=0.8)
    result = evaluate_product(product, _market())
    assert result["loss_frac"] == 0.0


def test_evaluate_product_loss_frac_nonprotected_vanilla():
    """非保本香草产品的 loss_frac 等于 client_price/notional。"""
    product = _product(product_type="vanilla_call", principal_protected=False)
    result = evaluate_product(product, _market())
    expected = result["client_price"] / product.notional
    assert result["loss_frac"] == pytest.approx(expected)


def test_evaluate_product_loss_frac_snowball_uses_expected_loss_frac():
    """雪球产品的 loss_frac 取自风险中性 MC 的 expected_loss_frac。"""
    product = _product(
        product_type="snowball", barrier_pct=0.75, barrier_type="knock_in", coupon_rate=0.12
    )
    result = evaluate_product(product, _market())
    assert result["loss_frac"] == pytest.approx(result["pricing_details"]["expected_loss_frac"])


def test_evaluate_product_loss_frac_autocallable_without_barrier_is_zero():
    """无障碍（保本地板）自动赎回产品的 loss_frac 应为 0。"""
    product = _product(
        product_type="autocallable",
        barrier_pct=None,
        barrier_type=None,
        coupon_rate=0.1,
        principal_protected=True,
    )
    result = evaluate_product(product, _market())
    assert result["loss_frac"] == 0.0


def test_evaluate_product_loss_frac_autocallable_with_barrier_uses_expected_loss_frac():
    """带障碍的自动赎回产品 loss_frac 取自风险中性 MC 的 expected_loss_frac。"""
    product = _product(
        product_type="autocallable",
        barrier_pct=0.75,
        barrier_type="knock_in",
        coupon_rate=0.1,
    )
    result = evaluate_product(product, _market())
    assert result["loss_frac"] == pytest.approx(result["pricing_details"]["expected_loss_frac"])


def test_price_product_barrier_delta_finite_difference_reasonable():
    """障碍期权 delta 由中心差分计算，量级应是合理的（非零、有界）。"""
    product = _product(product_type="barrier_call", barrier_pct=1.2, barrier_type="knock_out")
    result = price_product(product, _market())
    delta = result["greeks"]["delta"]
    assert -1.5 <= delta <= 1.5


def test_price_product_vega_pct_always_present():
    """vega_pct 恒定存在，供波动率约束使用。"""
    for kwargs in (
        dict(product_type="vanilla_call"),
        dict(product_type="barrier_put", barrier_pct=0.8, barrier_type="knock_in"),
        dict(
            product_type="snowball",
            barrier_pct=0.75,
            barrier_type="knock_in",
            coupon_rate=0.12,
        ),
    ):
        product = _product(**kwargs)
        result = price_product(product, _market())
        assert "vega_pct" in result["greeks"]
        assert isinstance(result["greeks"]["vega_pct"], float)


# ---------------------------------------------------------------------------
# delta 回归（Opus 审查缺陷 1/2 的修复验证）


def test_snowball_delta_is_positive():
    """雪球对现货的敏感度必须为正（修复前 fixing 未锚定导致恒为 0）。"""
    product = _product(
        product_type="snowball",
        barrier_pct=0.75,
        barrier_type="knock_in",
        coupon_rate=0.15,
    )
    delta = price_product(product, _market())["greeks"]["delta"]
    assert delta > 0.005


def test_autocallable_delta_is_positive():
    """带敲入障碍的自动赎回票据 delta 应为正且非零。"""
    product = _product(
        product_type="autocallable",
        barrier_pct=0.80,
        barrier_type="knock_in",
        coupon_rate=0.10,
    )
    delta = price_product(product, _market())["greeks"]["delta"]
    assert delta > 0.05


def test_autocallable_mc_greeks_ignore_participation_rate():
    """MC 产品（autocallable/snowball）的希腊值不应受 participation_rate 缩放：
    票息结构不含参与率杠杆，parse_product_spec 也强制其为 1.0，此处绕过解析器
    直接构造 ProductSpec 验证 price_product 本身不做该缩放。"""
    base_kwargs = dict(
        product_type="autocallable",
        barrier_pct=0.75,
        barrier_type="knock_in",
        coupon_rate=0.1,
    )
    product_p1 = _product(participation_rate=1.0, **base_kwargs)
    product_p2 = _product(participation_rate=2.0, **base_kwargs)
    market = _market()

    greeks_p1 = price_product(product_p1, market)["greeks"]
    greeks_p2 = price_product(product_p2, market)["greeks"]

    assert greeks_p1["delta"] == pytest.approx(greeks_p2["delta"])
    assert greeks_p1["vega"] == pytest.approx(greeks_p2["vega"])
    assert greeks_p1["vega_pct"] == pytest.approx(greeks_p2["vega_pct"])


def test_barrier_far_knockout_delta_matches_vanilla():
    """障碍位极远的 knock_out call 应退化为香草期权：delta 与解析值一致。"""
    product = _product(
        product_type="barrier_call",
        maturity_months=6,
        barrier_pct=0.01,
        barrier_type="knock_out",
    )
    market = _market()
    delta_fd = price_product(product, market)["greeks"]["delta"]
    delta_an = bs_greeks(100.0, 100.0, 0.5, 0.03, 0.2, 0.0, "call")["delta"]
    assert abs(delta_fd - delta_an) < 1e-3


def test_fair_value_stress_keeps_fixed_barrier_direction_after_spot_crossing():
    """上行压力跨越固定 up knock-out 后公允价值归零，且应成为最坏情景。"""
    product = _product(
        product_type="barrier_call",
        strike_pct=0.8,
        barrier_pct=1.1,
        barrier_type="knock_out",
        barrier_direction="up",
        reference_spot=100.0,
    )
    market = _market()
    base_fair_value = price_product(product, market)["fair_value"]

    loss, worst_id = pricing_module._fair_value_stress_loss(product, market)

    assert worst_id == "spot_up_20"
    assert loss == pytest.approx(base_fair_value)


@pytest.mark.parametrize("reference_spot", [None, 100.0])
def test_fair_value_stress_freezes_reference_without_rescaling_terms(
    monkeypatch, reference_spot
):
    """压力副本统一冻结发行参考现货，不得通过缩放比例二次改写条款。"""
    product = _product(
        product_type="barrier_call",
        strike_pct=0.9,
        barrier_pct=1.1,
        barrier_type="knock_out",
        barrier_direction="up",
        reference_spot=reference_spot,
    )
    captured: list[ProductSpec] = []

    def _capture_price(candidate, _market_state):
        captured.append(candidate)
        return {"fair_value": 1.0}

    monkeypatch.setattr(pricing_module, "price_product", _capture_price)
    pricing_module._fair_value_stress_loss(product, _market())

    stressed = captured[1:]
    assert len(stressed) == 3
    assert all(p.barrier_direction == "up" for p in stressed)
    assert all(p.reference_spot == pytest.approx(100.0) for p in stressed)
    assert all(p.strike_pct == pytest.approx(0.9) for p in stressed)
    assert all(p.barrier_pct == pytest.approx(1.1) for p in stressed)


@pytest.mark.parametrize("product_type", ["snowball", "autocallable"])
def test_fair_value_stress_freezes_reference_for_implicit_coupon_barriers(
    monkeypatch, product_type
):
    """发行时 reference=None 也必须在压力副本上锁定隐式 1.03/1.05 敲出水平。"""
    product = _product(
        product_type=product_type,
        barrier_pct=0.75,
        barrier_type="knock_in",
        coupon_rate=0.12,
        reference_spot=None,
    )
    captured: list[ProductSpec] = []

    def _capture_price(candidate, _market_state):
        captured.append(candidate)
        return {"fair_value": 1.0}

    monkeypatch.setattr(pricing_module, "price_product", _capture_price)
    pricing_module._fair_value_stress_loss(product, _market(spot=100.0))

    assert all(p.reference_spot == pytest.approx(100.0) for p in captured[1:])
    assert all(p.barrier_pct == pytest.approx(0.75) for p in captured[1:])


# ---------------------------------------------------------------------------
# hurdle_hit_prob：闭合公式正确性
# ---------------------------------------------------------------------------


def test_hurdle_hit_prob_custom_is_none():
    assert hurdle_hit_prob(_product(product_type="custom"), _market(), 0.05, 1000.0) is None


@pytest.mark.parametrize("participation", [0.0, -1.0])
def test_hurdle_hit_prob_rejects_nonpositive_participation(participation):
    """直接构造 ProductSpec 可绕过解析器，hurdle 边界必须防御除零/反号语义。"""
    product = _product(participation_rate=participation)
    with pytest.raises(PricingError, match="participation_rate > 0"):
        hurdle_hit_prob(product, _market(), 0.05, 50_000.0)


def test_positive_participation_parses_and_prices_end_to_end():
    product = parse_product_spec(
        {
            "product_type": "vanilla_call",
            "notional": 1_000_000,
            "maturity_months": 12,
            "strike_pct": 1.0,
            "participation_rate": 0.5,
            "target_client": "retail",
        }
    )

    pricing = evaluate_product(product, _market())
    probability = hurdle_hit_prob(
        product, _market(), 0.05, pricing["client_price"]
    )

    assert pricing["fair_value"] > 0.0
    assert probability is not None and 0.0 <= probability <= 1.0


def test_hurdle_hit_prob_protected_call_below_floor_is_one():
    """保本看涨：门槛远低于保本地板收益 → 概率 = 1.0。"""
    product = _product(product_type="vanilla_call", principal_protected=True)
    # protected call 地板 payoff_frac = 1.0，年化收益 ≥ 0；
    # 设 hurdle = -0.50 时 required_frac = client_price/notional - 0.5 < 1.0 → 1.0
    market = _market()
    client_price = 1_000_000.0 * 0.05  # 任意合理值
    prob = hurdle_hit_prob(product, market, -0.50, client_price)
    assert prob == 1.0


def test_hurdle_hit_prob_nonprotected_put_above_max_is_zero():
    """非保本看跌：所需收益超过最大可达收益 → 概率 = 0.0。"""
    # ATM 非保本 put：max payoff_frac = participation * K/S0 = 1.0 (strike_pct=1.0, spot=100)
    product = _product(product_type="vanilla_put", principal_protected=False, participation_rate=1.0)
    market = _market()
    # 设 client_price = 0，hurdle = 1.5 → required_frac = 1.5 > max 1.0 → 0.0
    prob = hurdle_hit_prob(product, market, 1.5, 0.0)
    assert prob == 0.0


def test_hurdle_hit_prob_call_monotone_with_alpha():
    """看涨：trend_alpha 越高，超越门槛概率单调不减。"""
    product = _product(product_type="vanilla_call")
    client_price = 50_000.0  # 5% of notional
    hurdle = 0.05
    alphas = [0.0, 0.05, 0.10]
    probs = [
        hurdle_hit_prob(product, _market(trend_alpha=a), hurdle, client_price)
        for a in alphas
    ]
    for p in probs:
        assert p is not None and 0.0 <= p <= 1.0
    # 单调不减
    assert probs[0] <= probs[1] + 1e-9
    assert probs[1] <= probs[2] + 1e-9


def test_hurdle_hit_prob_put_monotone_decreasing_with_alpha():
    """看跌：trend_alpha 越高（股价期望越高），超越下行门槛概率单调不增。"""
    product = _product(product_type="vanilla_put", principal_protected=False)
    client_price = 50_000.0
    hurdle = 0.02
    alphas = [0.0, 0.05, 0.10]
    probs = [
        hurdle_hit_prob(product, _market(trend_alpha=a), hurdle, client_price)
        for a in alphas
    ]
    for p in probs:
        assert p is not None and 0.0 <= p <= 1.0
    # 单调不增
    assert probs[0] >= probs[1] - 1e-9
    assert probs[1] >= probs[2] - 1e-9


def test_hurdle_hit_prob_closed_form_vs_mc():
    """闭合公式与 MC 近似（深处 knock_out ≈ vanilla）一致，容差 0.02。

    用 barrier_call knock_out，障碍置于 barrier_pct=100（几乎永不触碰），
    模拟结果应接近香草看涨闭合公式值。
    """
    hurdle = 0.05
    client_price = 50_000.0  # 5 % of 1e6 notional

    # 闭合公式：vanilla_call
    vanilla = _product(product_type="vanilla_call")
    market = _market(trend_alpha=0.0)
    p_closed = hurdle_hit_prob(vanilla, market, hurdle, client_price)

    # MC 近似：barrier_call knock_out，障碍极远
    barrier_deep = _product(product_type="barrier_call", barrier_pct=100.0, barrier_type="knock_out")
    p_mc = hurdle_hit_prob(barrier_deep, market, hurdle, client_price)

    assert p_closed is not None and p_mc is not None
    assert abs(p_closed - p_mc) < 0.02


def test_hurdle_hit_prob_determinism():
    """相同参数调用两次结果完全一致（随机种子固定）。"""
    product = _product(product_type="snowball", barrier_pct=0.75, barrier_type="knock_in", coupon_rate=0.12)
    market = _market(trend_alpha=0.03)
    p1 = hurdle_hit_prob(product, market, 0.05, 50_000.0)
    p2 = hurdle_hit_prob(product, market, 0.05, 50_000.0)
    assert p1 == p2


def test_barrier_hurdle_and_diagnostics_honor_explicit_fixed_direction():
    """显式 up 障碍低于发行现货时已触碰；MC 不得因 barrier_pct < 1 重推为 down。"""
    product = _product(
        product_type="barrier_call",
        barrier_pct=0.8,
        barrier_type="knock_out",
        barrier_direction="up",
        reference_spot=100.0,
    )
    market = _market()

    probability = hurdle_hit_prob(product, market, 0.01, 0.0)
    diag = mc_diagnostics(product, market, n_paths=64, seed=3)

    assert probability == 0.0
    assert diag.event_probs["touch"] == 1.0
    assert diag.pv_mean_frac == 0.0


# ---------------------------------------------------------------------------
# build_payoff_table
# ---------------------------------------------------------------------------


def test_build_payoff_table_custom_is_none():
    assert build_payoff_table(_product(product_type="custom"), 100.0) is None


def test_build_payoff_table_no_market_state_in_signature():
    """build_payoff_table 签名不含 MarketState 参数（防信息泄漏）。"""
    sig = inspect.signature(build_payoff_table)
    param_annotations = [p.annotation for p in sig.parameters.values()]
    from mirage.products import MarketState as _MS
    assert _MS not in param_annotations


def test_build_payoff_table_protected_vanilla_call_floor_rows():
    """保本 ATM 看涨期权：所有负涨跌情景的到期收益 = 1.0000（保本地板）。"""
    product = _product(
        product_type="vanilla_call",
        principal_protected=True,
        participation_rate=1.0,
        strike_pct=1.0,
    )
    table = build_payoff_table(product, 100.0)
    assert table is not None
    # 解析表格行，检查 -20%/-10%/-5%/0% 各行收益均为 1.0000
    for label in ["-20%", "-10%", "-5%", "0%"]:
        # 找到含该标签的行
        row = next(line for line in table.splitlines() if label in line)
        # 最后一个空格分隔字段是收益值
        value = float(row.strip().split()[-1])
        assert value == pytest.approx(1.0, abs=1e-4), f"{label} 行收益应为 1.0，实际 {value}"


def test_build_payoff_table_barrier_ko_triggered_column():
    """barrier knock_out：已触发列 = 保本时 1.0000，非保本时 0.0000。"""
    # 非保本
    p_noprotect = _product(
        product_type="barrier_call",
        barrier_pct=1.2,
        barrier_type="knock_out",
        principal_protected=False,
    )
    table_np = build_payoff_table(p_noprotect, 100.0)
    assert table_np is not None
    # 所有数据行的第三列（已触发障碍）应为 0.0000
    data_lines = [
        line for line in table_np.splitlines()
        if "%" in line and "-" not in line.strip()[0:1]
    ]
    # 取含 % 的数据行
    data_lines = [l for l in table_np.splitlines() if "%" in l and l.strip() and not l.startswith("-")]
    for row in data_lines:
        parts = row.strip().split()
        triggered_val = float(parts[-1])
        assert triggered_val == pytest.approx(0.0, abs=1e-4)

    # 保本
    p_protect = _product(
        product_type="barrier_call",
        barrier_pct=1.2,
        barrier_type="knock_out",
        principal_protected=True,
    )
    table_p = build_payoff_table(p_protect, 100.0)
    assert table_p is not None
    data_lines_p = [l for l in table_p.splitlines() if "%" in l and l.strip() and not l.startswith("-")]
    for row in data_lines_p:
        parts = row.strip().split()
        triggered_val = float(parts[-1])
        assert triggered_val == pytest.approx(1.0, abs=1e-4)


def test_build_payoff_table_snowball_mentions_knock_lines():
    """雪球情景表包含敲出和敲入场景描述。"""
    product = _product(
        product_type="snowball",
        barrier_pct=0.75,
        barrier_type="knock_in",
        coupon_rate=0.12,
    )
    table = build_payoff_table(product, 100.0)
    assert table is not None
    assert "敲出" in table
    assert "敲入" in table


# ---------------------------------------------------------------------------
# best_case_line
# ---------------------------------------------------------------------------


def test_best_case_line_custom_is_none():
    assert best_case_line(_product(product_type="custom"), _market(), 50_000.0) is None


def test_best_case_line_snowball_contains_coupon_pct():
    """雪球最高情景行包含票息年化百分比。"""
    product = _product(
        product_type="snowball",
        barrier_pct=0.75,
        barrier_type="knock_in",
        coupon_rate=0.12,
    )
    line = best_case_line(product, _market(), 50_000.0)
    assert line is not None
    assert "12.0%" in line


def test_best_case_line_vanilla_call_uncapped_suffix():
    """无封顶香草看涨期权摘要行包含「上不封顶」。"""
    product = _product(product_type="vanilla_call")
    line = best_case_line(product, _market(), 50_000.0)
    assert line is not None
    assert "上不封顶" in line
