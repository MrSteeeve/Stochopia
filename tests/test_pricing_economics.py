"""v2 定价经济学核心测试：MC 诊断、报价经济学、连续损失度量、系数标定。

覆盖 REDESIGN_PLAN 的可信度底线：margin 不再恒等于 1%×fair_value、最大名义
vanilla 非必然最优、同 seed 逐位复现、负 margin 不被截断、suitability 边界、
损失分位与保本产品损失为 0。
"""

from __future__ import annotations

import math

import pytest

from mirage.pricing import (
    ClientLossMeasure,
    MCDiagnostics,
    QuoteEconomics,
    QuotePolicy,
    _DIAG_CACHE,
    _fair_value_stress_loss,
    calibrate_quote_policy,
    client_loss_measure,
    evaluate_product,
    fair_value_stress_profile,
    hurdle_hit_prob,
    mc_diagnostics,
    quote_economics,
    solve_quote_equilibrium,
)
from mirage.products import ClientProfile, MarketState, ProductSpec


def market() -> MarketState:
    return MarketState(
        round_num=1,
        index_name="CSI500",
        spot=6000.0,
        volatility=0.22,
        risk_free_rate=0.02,
        dividend_yield=0.0,
        recent_trend="sideways",
        vix_level="mid",
    )


def product(
    product_type: str = "vanilla_call",
    *,
    notional: float = 1_000_000,
    maturity: int = 6,
    protected: bool = False,
    strike: float = 1.0,
    barrier: float | None = None,
    barrier_type: str | None = None,
    coupon: float | None = None,
) -> ProductSpec:
    return ProductSpec(
        product_type=product_type,
        notional=notional,
        maturity_months=maturity,
        strike_pct=strike,
        barrier_pct=barrier,
        barrier_type=barrier_type,
        coupon_rate=coupon,
        participation_rate=1.0,
        principal_protected=protected,
        target_client="c",
        pitch="透明的指数参与",
        hedging_plan="delta hedge",
    )


def client(
    *,
    allowed: list[str] | None = None,
    protection: bool = False,
    risk: str = "moderate",
    max_maturity: int = 12,
) -> ClientProfile:
    return ClientProfile(
        id="c",
        name="Client",
        capital=20_000_000,
        max_loss_pct=1.0,
        min_return_pct=0.03,
        risk_appetite=risk,
        max_maturity_months=max_maturity,
        principal_protection_required=protection,
        allowed_product_types=allowed,
    )


def snowball(notional: float = 5_000_000) -> ProductSpec:
    return product(
        "snowball", notional=notional, coupon=0.10, barrier=0.85, barrier_type="knock_in"
    )


def econ(prod: ProductSpec, cli: ClientProfile, *, policy: QuotePolicy | None = None,
         capacity: float | None = None, seed: int = 42) -> QuoteEconomics:
    """便捷组装：定价 + 诊断 + 压力损失 -> 报价经济学。"""
    mkt = market()
    policy = policy or QuotePolicy()
    pricing = evaluate_product(prod, mkt)
    diag = mc_diagnostics(prod, mkt, n_paths=2048, seed=seed)
    stress_loss, _ = _fair_value_stress_loss(prod, mkt)
    return quote_economics(
        pricing,
        diag,
        stress_loss=stress_loss,
        post_notional=prod.notional,
        capacity_notional=capacity if capacity is not None else cli.capital,
        policy=policy,
        product=prod,
        client=cli,
    )


def test_margin_not_constant_one_percent_of_fair_value():
    """margin 不再恒等于 1%×fair_value：两个结构不同的产品 margin_rate 不同。"""
    cli = client()
    qe_vanilla = econ(product("vanilla_call"), cli)
    qe_snow = econ(snowball(notional=1_000_000), cli)

    # 与旧口径（恒定 0.01×FV）显著不同
    assert abs(qe_vanilla.dealer_margin - 0.01 * qe_vanilla.fair_value) > 1.0
    # 两个结构的单位名义 margin 不同 -> margin 随结构风险变化，非恒定
    assert not math.isclose(qe_vanilla.margin_rate, qe_snow.margin_rate, abs_tol=1e-6)


def test_largest_notional_vanilla_is_not_margin_optimal():
    """最大名义 vanilla 非必然最优：容量 Q² 项 + suitability 联合压制。"""
    cli = client()
    capacity = cli.capital
    big_vanilla = product("vanilla_call", notional=int(capacity))  # 100% 容量 -> Q=1
    small_protected = product("vanilla_call", notional=1_000_000, protected=True)
    snow = snowball(notional=5_000_000)

    results = {
        "big_vanilla": econ(big_vanilla, cli, capacity=capacity),
        "small_protected": econ(small_protected, cli, capacity=capacity),
        "snowball": econ(snow, cli, capacity=capacity),
    }
    best = max(results, key=lambda k: results[k].dealer_margin)
    assert best != "big_vanilla"
    # 上满名义的 vanilla 被容量冲击打成负 margin
    assert results["big_vanilla"].dealer_margin < 0


def test_same_seed_is_bit_identical():
    """同 seed 两次调用 mc_diagnostics / quote_economics 结果逐位相同。"""
    prod = snowball()
    cli = client()
    mkt = market()

    _DIAG_CACHE.clear()
    d1 = mc_diagnostics(prod, mkt, n_paths=2048, seed=7)
    _DIAG_CACHE.clear()
    d2 = mc_diagnostics(prod, mkt, n_paths=2048, seed=7)

    assert d1.pv_mean_frac == d2.pv_mean_frac
    assert d1.pv_std_frac == d2.pv_std_frac
    assert d1.pv_se_frac == d2.pv_se_frac
    assert d1.expected_loss_frac == d2.expected_loss_frac
    assert d1.p95_loss_frac == d2.p95_loss_frac
    assert d1.expected_life_months == d2.expected_life_months
    assert d1.event_probs == d2.event_probs

    pricing = evaluate_product(prod, mkt)
    stress_loss, _ = _fair_value_stress_loss(prod, mkt)
    kwargs = dict(stress_loss=stress_loss, post_notional=prod.notional,
                  capacity_notional=cli.capital, policy=QuotePolicy(), product=prod, client=cli)
    qe1 = quote_economics(pricing, d1, **kwargs)
    qe2 = quote_economics(pricing, d2, **kwargs)
    assert qe1.dealer_margin == qe2.dealer_margin
    assert qe1.client_price == qe2.client_price
    assert qe1.hedging_cost == qe2.hedging_cost
    assert qe1.breakdown == qe2.breakdown


def test_negative_margin_is_not_clamped():
    """负 margin 可出现且不被截断。"""
    cli = client()
    capacity = cli.capital
    big_vanilla = product("vanilla_call", notional=int(capacity))
    qe = econ(big_vanilla, cli, capacity=capacity)
    assert qe.dealer_margin < 0
    # dealer_margin = N·(r_c·suit − r_h)，未做 max(·,0) 截断
    expected = big_vanilla.notional * (
        qe.breakdown["r_c_effective"] - qe.breakdown["r_h"]
    )
    assert math.isclose(qe.dealer_margin, expected, rel_tol=1e-12, abs_tol=1e-6)


def test_suitability_boundary():
    """suitability 边界：完全适配≈1；白名单外产品显著小于 1。"""
    prod = product("vanilla_call")
    fit = econ(prod, client(allowed=None)).suitability
    assert fit == pytest.approx(1.0, abs=1e-9)

    off_whitelist = econ(prod, client(allowed=["snowball"])).suitability
    assert off_whitelist < 0.5
    assert off_whitelist < fit

    # 要求保本却不保本 -> 适配度进一步压低
    mismatch = econ(prod, client(protection=True)).suitability
    assert mismatch < fit


def test_p95_loss_ge_expected_and_protected_loss_zero():
    """p95_loss_frac ≥ expected_loss_frac；保本产品损失度量为 0。"""
    mkt = market()
    diag = mc_diagnostics(snowball(), mkt, n_paths=4096, seed=42)
    assert diag.p95_loss_frac >= diag.expected_loss_frac
    assert diag.expected_loss_frac >= 0.0

    protected = product("vanilla_call", protected=True)
    diag_p = mc_diagnostics(protected, mkt, n_paths=2048, seed=42)
    assert diag_p.expected_loss_frac == 0.0
    assert diag_p.p95_loss_frac == 0.0

    pricing_p = evaluate_product(protected, mkt)
    measure = client_loss_measure(protected, pricing_p, 0.0, worst_stress_id="")
    assert measure.observed_loss_frac == 0.0


def test_client_loss_measure_takes_max_of_three():
    """连续损失取 expected/premium/stress 三者最大。"""
    mkt = market()
    prod = product("vanilla_call")
    pricing = evaluate_product(prod, mkt)
    stress_loss = 0.30 * prod.notional
    measure = client_loss_measure(prod, pricing, stress_loss, worst_stress_id="spot_down_20")
    assert isinstance(measure, ClientLossMeasure)
    assert measure.stress_loss_frac == pytest.approx(0.30)
    assert measure.observed_loss_frac == max(
        measure.expected_loss_frac, measure.premium_at_risk_frac, measure.stress_loss_frac
    )
    assert measure.worst_stress_id == "spot_down_20"


def test_diagnostics_cache_keyed_without_notional():
    """诊断按去掉 notional 的结构键缓存：同结构不同名义命中同一对象。"""
    mkt = market()
    _DIAG_CACHE.clear()
    d_small = mc_diagnostics(product("vanilla_call", notional=1_000_000), mkt, n_paths=2048, seed=5)
    d_large = mc_diagnostics(product("vanilla_call", notional=9_000_000), mkt, n_paths=2048, seed=5)
    assert d_small is d_large


def test_calibrate_quote_policy_hits_target_band():
    """标定：20–60% 正 margin 且结构排序非退化，报告可冻结。"""
    cli = client()
    mkt = market()
    dev = [
        (product("vanilla_call", notional=int(cli.capital)), mkt, cli),
        (product("vanilla_call", notional=2_000_000), mkt, cli),
        (product("vanilla_call", notional=1_000_000, protected=True), mkt, cli),
        (snowball(notional=5_000_000), mkt, cli),
        (product("barrier_call", notional=3_000_000, barrier=1.10, barrier_type="knock_out"), mkt, cli),
    ]
    calibrated, report = calibrate_quote_policy(dev, seed=42)
    assert isinstance(calibrated, QuotePolicy)
    assert 0.20 <= report["selected_positive_margin_rate"] <= 0.60
    assert report["ranking_nondegenerate"] is True
    assert report["within_target"] is True
    assert "freeze" in report["warning"]


def test_unhedged_stress_is_zero_sum_scenario_by_scenario():
    """同一压力情景下，客户价值 P&L 与 dealer liability P&L 必须严格零和。"""
    profile = fair_value_stress_profile(product("vanilla_call"), market())

    assert profile.scenarios
    for scenario in profile.scenarios:
        assert scenario.client_pnl + scenario.dealer_liability_pnl == pytest.approx(
            0.0, abs=1e-12
        )


def test_quote_equilibrium_recomputes_probability_at_returned_price():
    """保存的 price/probability/suitability 是同一个固定点，而不是两遍近似。"""
    prod = product("vanilla_call")
    cli = client()
    cli.min_hit_prob = 0.8
    mkt = market()
    pricing = evaluate_product(prod, mkt)
    diag = mc_diagnostics(prod, mkt, n_paths=512, seed=19)
    profile = fair_value_stress_profile(prod, mkt)
    policy = QuotePolicy(diagnostic_paths=512)

    equilibrium = solve_quote_equilibrium(
        prod,
        mkt,
        cli,
        pricing,
        diag,
        dealer_stress_loss=profile.dealer_hedged_stress_loss,
        post_notional=prod.notional,
        capacity_notional=cli.capital,
        policy=policy,
    )

    assert equilibrium.converged is True
    recomputed_probability = hurdle_hit_prob(
        prod,
        mkt,
        cli.min_return_pct,
        equilibrium.economics.client_price,
    )
    assert equilibrium.hurdle_hit_prob == pytest.approx(
        recomputed_probability, abs=1e-12
    )
    pricing_at_fixed_point = {
        **pricing,
        "hurdle_hit_prob": recomputed_probability,
    }
    recomputed_economics = quote_economics(
        pricing_at_fixed_point,
        diag,
        dealer_stress_loss=profile.dealer_hedged_stress_loss,
        post_notional=prod.notional,
        capacity_notional=cli.capital,
        policy=policy,
        product=prod,
        client=cli,
    )
    assert recomputed_economics.client_price == pytest.approx(
        equilibrium.economics.client_price,
        abs=max(prod.notional * 1e-8, 1e-8),
    )


def test_funded_note_is_par_and_reports_separate_cash_fields():
    """Level-0 funded note 不再把 exposure、cash、premium 和 protection 混为一项。"""
    prod = product("vanilla_call", protected=True)
    cli = client()
    qe = econ(prod, cli)

    assert qe.funding_style == "funded_note"
    assert qe.face_value == prod.notional
    assert qe.issue_price_pct == pytest.approx(1.0)
    assert qe.cash_outlay == pytest.approx(qe.face_value)
    assert qe.premium is None
    assert qe.protected_amount == pytest.approx(qe.face_value)
    assert qe.dealer_fee == pytest.approx(qe.cash_outlay - qe.fair_value)
