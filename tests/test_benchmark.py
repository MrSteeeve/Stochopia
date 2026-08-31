"""Research benchmark protocol: hard checks, state, leakage, market fallback,
shared domain lattice, continuous loss budget and the v2 quote economics gate."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from stochopia.benchmark import (
    BenchmarkCondition,
    BenchmarkError,
    HardConstraintEngine,
    LongHorizonEnvironment,
    MarketSnapshot,
    PortfolioState,
    ProductDomainSpec,
    RiskBudget,
    TradingDesk,
    WorkflowOutcome,
    calibrate_risk_budget,
    carry_sensitivity,
    client_contract_pass,
    enumerate_domain,
    load_market_snapshots,
    option_implied_forward,
    oracle_best_quote,
    settle_submission,
    validate_domain,
)
from stochopia.env_agents import RoleResponse
from stochopia.products import ClientProfile, ProductSpec


# A deliberately tiny lattice so oracle enumeration and property tests stay fast.
# capital = 20_000_000 -> notionals {.02:400_000, .05:1_000_000, .10:2_000_000}.
SMALL_DOMAIN = ProductDomainSpec(
    product_types=("vanilla_call", "vanilla_put", "snowball"),
    notional_fractions=(.02, .05, .10),
    maturities=(3, 6),
    strikes=(1.00,),
    barriers=(.85,),
    coupons=(.08,),
    participations=(1.0,),
    principal_protected=(False,),
)


def snapshots(episode: str = "CSI500_2023H1") -> list[MarketSnapshot]:
    return [
        MarketSnapshot(
            episode_id=episode,
            round_num=1,
            as_of=date(2023, 1, 31),
            underlying="CSI500",
            spot=6000.0,
            risk_free_rate=0.02,
            realized_vol_20d=0.22,
            realized_vol_60d=0.20,
            atm_iv_1m=0.22,
            atm_iv_3m=0.24,
            regime="sideways",
            source="synthetic-test-fixture",
        ),
        MarketSnapshot(
            episode_id=episode,
            round_num=2,
            as_of=date(2023, 2, 28),
            underlying="CSI500",
            spot=5800.0,
            risk_free_rate=0.02,
            realized_vol_20d=0.25,
            realized_vol_60d=0.21,
            regime="high_vol_downtrend",
            source="synthetic-test-fixture",
        ),
    ]


def client(
    max_loss: float = 1.0,
    *,
    min_hit_prob: float = 0.5,
    allowed: list[str] | None = None,
    protection: bool = False,
    risk: str = "moderate",
    max_maturity: int = 12,
) -> ClientProfile:
    return ClientProfile(
        id="institutional",
        name="Synthetic Institutional Client",
        capital=20_000_000,
        max_loss_pct=max_loss,
        min_return_pct=0.03,
        risk_appetite=risk,
        max_maturity_months=max_maturity,
        principal_protection_required=protection,
        allowed_product_types=allowed,
        min_hit_prob=min_hit_prob,
        preferences="yield with transparent downside",
    )


def permissive_client() -> ClientProfile:
    """Accepts anything hard-executable: loss budget wide, hurdle threshold off."""
    return client(max_loss=1.0, min_hit_prob=0.0, allowed=None)


def budget(scale: float = 1.0) -> RiskBudget:
    return RiskBudget(
        notional=100_000_000 * scale,
        net_delta=100_000_000 * scale,
        gross_delta=200_000_000 * scale,
        net_vega=20_000_000 * scale,
        stress_loss=100_000_000 * scale,
    )


def product(*, maturity: int = 6, protected: bool = False, notional: float = 1_000_000,
            product_type: str = "vanilla_call") -> ProductSpec:
    return ProductSpec(
        product_type=product_type,
        notional=notional,
        maturity_months=maturity,
        strike_pct=1.0,
        barrier_pct=None,
        barrier_type=None,
        coupon_rate=None,
        participation_rate=1.0,
        principal_protected=protected,
        target_client="institutional",
        pitch="透明的指数上涨参与",
        hedging_plan="delta hedge with listed ETF",
    )


def make_env(*, full: bool = False, dynamic: bool = True, max_loss: float = 1.0,
             min_hit_prob: float = 0.0, cli: ClientProfile | None = None,
             domain: ProductDomainSpec | None = None):
    return LongHorizonEnvironment(
        snapshots(),
        cli or client(max_loss, min_hit_prob=min_hit_prob),
        budget(),
        BenchmarkCondition(full_information=full, dynamic=dynamic),
        domain=domain or SMALL_DOMAIN,
    )


# ---------------------------------------------------------------------------
# Unchanged information-boundary and loader behaviour
# ---------------------------------------------------------------------------


def test_market_volatility_fallback_and_no_raw_chain_leakage():
    first, second = snapshots()
    assert first.pricing_volatility() == (0.24, "atm_iv_3m")
    assert second.pricing_volatility() == (0.25, "realized_vol_20d")
    brief = first.public_brief()
    assert "option_chain" not in brief
    assert brief["volatility_source"] == "atm_iv_3m"
    assert first.pricing_volatility(1) == (0.22, "atm_iv_1m")


def test_option_implied_forward_absorbs_carry():
    result = option_implied_forward(call=105.0, put=95.0, strike=6000.0, rate=0.02, years=0.5)
    assert result == pytest.approx(6010.10050167)


def test_trend_alpha_reaches_runtime_quote_and_does_not_cross_contaminate_cache():
    cli = permissive_client()
    prod = product(maturity=3)
    desk = TradingDesk(HardConstraintEngine(budget()), domain=SMALL_DOMAIN)
    base = snapshots()[0]
    down = replace(base, trend_alpha=-0.10)
    up = replace(base, trend_alpha=0.10)

    quote_down = desk.quote(prod, down, cli, PortfolioState(), "down", 1)
    quote_up = desk.quote(prod, up, cli, PortfolioState(), "up", 1)

    assert quote_down.hurdle_hit_prob is not None
    assert quote_up.hurdle_hit_prob is not None
    assert quote_up.hurdle_hit_prob > quote_down.hurdle_hit_prob


def test_partial_information_topic_gate_and_budget():
    env = make_env(full=False)
    assert "client_constraints" not in env.get_round_brief()
    assert env.get_round_brief()["client_id"] == "institutional"
    assert env.query_client("capital")["answer"] == 20_000_000
    env.query_client("maturity")
    env.query_client("protection")
    with pytest.raises(BenchmarkError, match="budget exhausted"):
        env.query_client("preferences")
    invalid = make_env()
    with pytest.raises(BenchmarkError, match="unknown client topic"):
        invalid.query_client("What is your loss tolerance?")
    assert invalid.query_count == 0


def test_full_information_discloses_constraints():
    env = make_env(full=True, min_hit_prob=0.5)
    disclosed = env.get_round_brief()["client_constraints"]
    assert disclosed["capital"] == 20_000_000
    assert disclosed["max_maturity_months"] == 12
    # Full legacy cells must at least include every field that can change
    # deterministic settlement; otherwise Full is not a superset of Partial.
    assert disclosed["accepting_new_products"] is True
    assert disclosed["risk_appetite"] == "moderate"
    assert disclosed["min_return_pct"] == pytest.approx(0.03)
    assert disclosed["min_hit_prob"] == pytest.approx(0.5)


def test_partial_quote_redacts_exact_client_mandate_values():
    partial = make_env(full=False)
    payload = partial.request_quote(product())
    checks = {item["check_id"]: item for item in payload["checks"]}

    for check_id in (
        "CLIENT_ACCEPTING",
        "CLIENT_CAPITAL",
        "CLIENT_MATURITY",
        "CLIENT_PRODUCT_WHITELIST",
        "CLIENT_LOSS_BUDGET_V2",
        "CLIENT_PROTECTION",
    ):
        assert checks[check_id]["observed"] is None
        assert checks[check_id]["limit"] is None
        assert "query" in checks[check_id]["reason"]

    # Product/domain and desk-risk checks are not hidden client preferences.
    assert checks["DOMAIN"]["limit"] == SMALL_DOMAIN.version
    assert checks["PORTFOLIO_NOTIONAL"]["limit"] == budget().notional

    full = make_env(full=True)
    full_checks = {
        item["check_id"]: item for item in full.request_quote(product())["checks"]
    }
    assert full_checks["CLIENT_CAPITAL"]["limit"] == 20_000_000
    assert full_checks["CLIENT_MATURITY"]["limit"] == 12


def test_partial_client_topics_cover_settlement_truth_without_bulk_disclosure():
    env = make_env(full=False)
    assert env.query_client("purchase_status")["answer"] is True
    assert env.query_client("risk_appetite")["answer"] == "moderate"
    hurdle = env.query_client("return_hurdle")["answer"]
    assert hurdle == {"min_return_pct": 0.03, "min_hit_prob": 0.0}


def test_quote_budget_and_state_binding():
    env = make_env()
    first = env.request_quote(product())
    assert first["hard_pass"] is True
    assert first["valid_for_state"] == env.state_version
    env.request_quote(product(notional=2_000_000))
    env.request_quote(product(notional=400_000))
    with pytest.raises(BenchmarkError, match="quote budget exhausted"):
        env.request_quote(product(notional=1_000_000))


def test_load_market_snapshots_validates_contiguous_rounds(tmp_path):
    csv_path = tmp_path / "market.csv"
    csv_path.write_text(
        "episode_id,round,date,underlying,spot,risk_free_rate,realized_vol_20d,source\n"
        "E1,1,2023-01-31,CSI500,6000,0.02,0.20,fixture\n"
        "E1,3,2023-03-31,CSI500,6100,0.02,0.21,fixture\n",
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkError, match="contiguous"):
        load_market_snapshots(csv_path)


# ---------------------------------------------------------------------------
# v2: shared domain lattice and one-step attainment <= 1
# ---------------------------------------------------------------------------


def test_domain_membership_is_agent_oracle_symmetric():
    cli = permissive_client()
    for candidate in enumerate_domain(cli, SMALL_DOMAIN):
        assert validate_domain(candidate, cli, SMALL_DOMAIN)[0].passed
    # 0.055 of capital is not one of the frozen notional fractions.
    off = product(notional=1_100_000)
    domain_check = validate_domain(off, cli, SMALL_DOMAIN)[0]
    assert domain_check.check_id == "DOMAIN"
    assert not domain_check.passed


def test_out_of_domain_product_is_rejected_without_spending_quote_budget():
    env = make_env(cli=permissive_client())
    with pytest.raises(BenchmarkError, match="DOMAIN"):
        env.request_quote(product(notional=1_100_000))  # out of lattice

    assert env.quote_count == 0
    assert env.quotes == {}


def test_one_step_attainment_never_exceeds_one():
    cli = permissive_client()
    snap = snapshots()[0]
    bud = budget()
    port = PortfolioState()
    desk = TradingDesk(HardConstraintEngine(bud), domain=SMALL_DOMAIN)

    candidates = list(enumerate_domain(cli, SMALL_DOMAIN))
    quotes = [desk.quote(p, snap, cli, port, "sv", i + 1) for i, p in enumerate(candidates)]
    feasible = [(p, q) for p, q in zip(candidates, quotes) if q.hard_pass]
    assert feasible, "the small domain must yield at least one hard-feasible product"

    oracle = oracle_best_quote(SMALL_DOMAIN, snap, cli, port, bud)
    assert oracle is not None
    oracle_margin = oracle[1].dealer_margin

    # Every quotable agent action attains at most the oracle margin.
    for _, quote in feasible:
        assert quote.dealer_margin <= oracle_margin + 1e-9
    best_agent_margin = max(quote.dealer_margin for _, quote in feasible)
    assert best_agent_margin == pytest.approx(oracle_margin)


def test_voluntary_best_submission_attainment_le_one():
    """Submitting the oracle-best product via the environment attains ratio <= 1."""
    cli = permissive_client()
    snap = snapshots()[0]
    bud = budget()
    oracle = oracle_best_quote(SMALL_DOMAIN, snap, cli, PortfolioState(), bud)
    assert oracle is not None
    best_product, best_quote = oracle

    env = LongHorizonEnvironment(
        [snap], cli, bud, BenchmarkCondition(full_information=True, dynamic=True),
        domain=SMALL_DOMAIN,
    )
    payload = env.request_quote(best_product)
    result = env.submit_design(payload["quote_id"], "oracle-best voluntary submission")
    assert result["accepted"] is True
    assert result["dealer_margin"] / best_quote.dealer_margin <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# v2: continuous CLIENT_LOSS_BUDGET_V2
# ---------------------------------------------------------------------------


def test_client_loss_budget_v2_is_continuous_not_boolean():
    env = make_env(full=True, max_loss=1.0)
    payload = env.request_quote(product(protected=False))
    loss_check = next(c for c in payload["checks"] if c["check_id"] == "CLIENT_LOSS_BUDGET_V2")
    # A standalone option's actual cash outlay is its premium.  That premium
    # can be lost in full even though exposure notional is much larger.
    assert loss_check["status"] == "PASS"
    assert loss_check["observed"] == pytest.approx(1.0)


def test_client_loss_budget_v2_fails_tight_client_and_cannot_be_overridden():
    env = make_env(max_loss=0.01)
    payload = env.request_quote(product(protected=False))
    failed = {c["check_id"] for c in payload["checks"] if c["status"] == "FAIL"}
    assert "CLIENT_LOSS_BUDGET_V2" in failed
    submitted = env.submit_design(payload["quote_id"], "soft explanation cannot waive hard checks")
    assert submitted["accepted"] is False
    assert submitted["hard_executable"] is False


# ---------------------------------------------------------------------------
# v2: deterministic client contract gate
# ---------------------------------------------------------------------------


def test_client_contract_pass_blocks_hard_executable_but_unsuitable_product():
    # High hurdle threshold is a contract-only gate (no HARD check mirrors it),
    # so a plain call is hard-executable yet still fails to settle.
    cli = client(max_loss=1.0, min_hit_prob=0.9, allowed=None)
    env = LongHorizonEnvironment(
        snapshots(), cli, budget(), BenchmarkCondition(full_information=True, dynamic=True),
        domain=SMALL_DOMAIN,
    )
    payload = env.request_quote(product())
    assert payload["hard_pass"] is True  # hard-executable
    result = env.submit_design(payload["quote_id"])
    assert result["hard_executable"] is True
    assert result["client_contract_pass"] is False
    assert result["accepted"] is False
    assert any(c["check_id"] == "CONTRACT_HURDLE" for c in result["contract_failures"])


def test_partial_settlement_redacts_contract_thresholds():
    cli = client(max_loss=1.0, min_hit_prob=0.9, allowed=None)
    env = LongHorizonEnvironment(
        snapshots(), cli, budget(), BenchmarkCondition(False, True),
        domain=SMALL_DOMAIN,
    )
    payload = env.request_quote(product())
    result = env.submit_design(payload["quote_id"])
    hurdle = next(
        item for item in result["contract_failures"]
        if item["check_id"] == "CONTRACT_HURDLE"
    )
    assert hurdle["observed"] is None
    assert hurdle["limit"] is None
    assert "query" in hurdle["reason"]


def test_client_contract_pass_pure_function_mirrors_would_buy_dimensions():
    cli = client(allowed=["snowball"])  # vanilla off whitelist
    pricing = {
        "loss_frac": 0.05,
        "stress_loss": 0.05 * 1_000_000,
        "pricing_details": {},
        "hurdle_hit_prob": 0.9,
    }
    ok, checks = client_contract_pass(product(), pricing, cli)
    assert ok is False
    failed = {c.check_id for c in checks if not c.passed}
    assert "CONTRACT_WHITELIST" in failed


# ---------------------------------------------------------------------------
# v2: dealer_margin economics no longer 1% x fair_value
# ---------------------------------------------------------------------------


def test_dealer_margin_is_not_one_percent_of_fair_value():
    env = make_env(cli=permissive_client())
    call = env.request_quote(product())
    env2 = make_env(cli=permissive_client())
    snow = env2.request_quote(
        ProductSpec("snowball", 1_000_000, 6, 1.0, 0.85, "knock_in", 0.08, 1.0, False,
                    "institutional", "雪球", "delta hedge")
    )
    # Not the old constant 1% x fair_value path.
    assert abs(call["dealer_margin"] - 0.01 * call["fair_value"]) > 1.0
    # Margin per unit notional varies with structure risk (was identically constant).
    assert call["margin_rate"] != snow["margin_rate"]
    # public payload exposes margin_rate / suitability but never the breakdown.
    assert "margin_rate" in call and "suitability" in call
    assert "breakdown" not in call


# ---------------------------------------------------------------------------
# Dynamic / static state carry, oracle, calibration and carry sensitivity
# ---------------------------------------------------------------------------


def test_dynamic_state_persists_and_matures():
    env = make_env(dynamic=True, cli=permissive_client())
    payload = env.request_quote(product(maturity=3))
    assert env.submit_design(payload["quote_id"])["accepted"] is True
    assert len(env.portfolio.positions) == 1
    assert env.advance_round() == []
    assert env.get_round_brief()["portfolio_summary"]["outstanding_products"] == 1
    assert env.portfolio.positions[0].remaining_months == 2


def test_dynamic_state_revalues_against_next_market_snapshot():
    env = make_env(dynamic=True, cli=permissive_client())
    payload = env.request_quote(product(maturity=3))
    assert env.submit_design(payload["quote_id"])["accepted"] is True
    position = env.portfolio.positions[0]
    issue_delta = position.delta_dollars
    issue_fv = position.current_fair_value
    assert position.absolute_strike == pytest.approx(6000.0)

    env.advance_round()

    assert position.remaining_months == 2
    assert position.elapsed_months == 1
    assert position.last_valuation_round == 2
    assert position.current_fair_value != pytest.approx(issue_fv)
    assert position.delta_dollars != pytest.approx(issue_delta)
    # Contract levels remain anchored to the issue fixing, not the new 5800 spot.
    assert position.absolute_strike == pytest.approx(6000.0)


def test_dynamic_barrier_knockout_closes_position_after_spot_crossing():
    states = snapshots()
    states[1] = replace(states[1], spot=7000.0)
    env = LongHorizonEnvironment(
        states, permissive_client(), budget(), BenchmarkCondition(False, True),
    )
    barrier = ProductSpec(
        "barrier_call", 1_000_000, 3, 1.0, 1.10, "knock_out", None, 1.0,
        False, "institutional", "up-and-out", "delta hedge",
        barrier_direction="up",
    )
    payload = env.request_quote(barrier)
    assert env.submit_design(payload["quote_id"])["accepted"] is True
    position = env.portfolio.positions[0]

    closed = env.advance_round()

    assert closed == [position.position_id]
    assert position.knock_out_state is True
    assert position.status == "knocked_out"
    assert env.portfolio.positions == []


def test_protected_barrier_knockout_keeps_bond_floor_liability():
    states = snapshots()
    states[1] = replace(states[1], spot=7000.0)
    env = LongHorizonEnvironment(
        states, permissive_client(), budget(), BenchmarkCondition(False, True),
    )
    protected = ProductSpec(
        "barrier_call", 1_000_000, 3, 1.0, 1.10, "knock_out", None, 1.0,
        True, "institutional", "protected up-and-out", "delta hedge",
        barrier_direction="up",
    )
    payload = env.request_quote(protected)
    assert env.submit_design(payload["quote_id"])["accepted"] is True
    position = env.portfolio.positions[0]

    closed = env.advance_round()

    assert closed == []
    assert position.knock_out_state is True
    assert position.status == "active"
    assert position.current_fair_value > 0.9 * protected.notional
    assert position.delta_dollars == pytest.approx(0.0)
    assert env.portfolio.totals()["notional"] == protected.notional


def test_quote_product_is_snapshotted_against_mutation_before_submit():
    env = make_env(dynamic=True, cli=permissive_client())
    mutable = product(maturity=3)
    payload = env.request_quote(mutable)

    mutable.notional = 9_000_000
    mutable.maturity_months = 60
    mutable.strike_pct = 10.0
    result = env.submit_design(payload["quote_id"])

    assert result["accepted"] is True
    issued = env.portfolio.positions[0].product
    assert issued.notional == 1_000_000
    assert issued.maturity_months == 3
    assert issued.strike_pct == 1.0


def test_quote_boundary_rejects_forged_lifecycle_state():
    env = make_env(cli=permissive_client())
    forged = replace(product(maturity=3), elapsed_months=3, reference_spot=6000.0)

    with pytest.raises(BenchmarkError, match="environment-maintained"):
        env.request_quote(forged)

    assert env.quote_count == 0


def test_quote_boundary_rejects_wrong_target_without_spending_budget():
    env = make_env(cli=permissive_client())
    wrong_target = replace(product(), target_client="some-other-client")

    with pytest.raises(BenchmarkError, match="TARGET_CLIENT_MATCH"):
        env.request_quote(wrong_target)

    assert env.quote_count == 0
    assert env.quotes == {}


def test_maturity_closes_position_into_cashflow_and_zero_sum_pnl_ledger():
    one_month_domain = ProductDomainSpec(
        product_types=("vanilla_call",),
        notional_fractions=(.05,),
        maturities=(1,),
        strikes=(1.0,),
        barriers=(.85,),
        coupons=(.08,),
        participations=(1.0,),
        principal_protected=(False,),
    )
    env = make_env(
        dynamic=True,
        cli=permissive_client(),
        domain=one_month_domain,
    )
    one_month = product(maturity=1, protected=False, notional=1_000_000)
    quote = env.request_quote(one_month)
    result = env.submit_design(quote["quote_id"])
    assert result["accepted"] is True
    assert len(env.portfolio.cashflow_ledger) == 1
    assert env.portfolio.cashflow_ledger[0].event_type == "issuance"
    assert result["client_account"]["available_cash"] == pytest.approx(
        result["client_account"]["initial_cash"] - quote["cash_outlay"]
    )
    assert result["client_account"]["locked_cash"] == pytest.approx(
        quote["cash_outlay"]
    )
    assert result["dealer_account"]["model_equity"] == pytest.approx(
        quote["dealer_fee"]
    )

    closed = env.advance_round()

    assert len(closed) == 1
    assert env.portfolio.positions == []
    assert len(env.portfolio.closed_positions) == 1
    assert len(env.portfolio.lifecycle_events) == 1
    event = env.portfolio.lifecycle_events[0]
    expected_cashflows = 2 + int(abs(event.dealer_hedge_pnl) > 1e-12)
    assert len(env.portfolio.cashflow_ledger) == expected_cashflows
    assert event.close_reason == "matured"
    assert event.client_realized_pnl + event.dealer_liability_realized_pnl == pytest.approx(
        0.0, abs=1e-12
    )
    assert event.dealer_hedged_realized_pnl == pytest.approx(
        event.dealer_liability_realized_pnl + event.dealer_hedge_pnl
    )
    assert event.transaction_costs is None
    assert event.dealer_total_pnl is None

    client_account = env.portfolio.client_account
    assert client_account is not None
    assert client_account.locked_cash == pytest.approx(0.0)
    assert client_account.realised_pnl == pytest.approx(event.client_realized_pnl)
    assert client_account.available_cash == pytest.approx(
        client_account.initial_cash + event.client_realized_pnl
    )

    dealer_account = env.portfolio.dealer_account
    assert dealer_account is not None
    assert dealer_account.realised_liability_pnl == pytest.approx(
        event.dealer_liability_realized_pnl
    )
    assert dealer_account.realised_hedged_pnl == pytest.approx(
        event.dealer_hedged_realized_pnl
    )
    dealer_snapshot = dealer_account.snapshot(
        active_liability_fair_value=0.0,
        active_stress=0.0,
    )
    assert dealer_snapshot["model_equity"] == pytest.approx(
        event.dealer_hedged_realized_pnl
    )
    assert dealer_snapshot["risk_capital_equity"] == pytest.approx(
        dealer_account.risk_capital_limit + event.dealer_hedged_realized_pnl
    )


def test_realised_client_loss_reduces_next_round_investable_cash():
    account_domain = ProductDomainSpec(
        product_types=("vanilla_call",),
        notional_fractions=(.05, 1.0),
        maturities=(1,),
        strikes=(1.0,),
        barriers=(.85,),
        coupons=(.08,),
        participations=(1.0,),
        principal_protected=(False, True),
    )
    env = make_env(
        full=True,
        dynamic=True,
        cli=permissive_client(),
        domain=account_domain,
    )
    loss_quote = env.request_quote(
        product(maturity=1, protected=False, notional=1_000_000)
    )
    assert env.submit_design(loss_quote["quote_id"])["accepted"] is True
    env.advance_round()
    assert env.portfolio.client_account is not None
    assert env.portfolio.client_account.available_cash < 20_000_000

    next_quote = env.request_quote(
        product(maturity=1, protected=True, notional=20_000_000)
    )
    capital_check = next(
        item
        for item in next_quote["checks"]
        if item["check_id"] == "CLIENT_CAPITAL"
    )
    assert capital_check["status"] == "FAIL"
    assert capital_check["observed"] == pytest.approx(20_000_000)
    assert capital_check["limit"] == pytest.approx(
        env.portfolio.client_account.available_cash
    )


def test_round_overrides_are_resolved_on_every_benchmark_round():
    scheduled = replace(
        permissive_client(),
        round_overrides=[{
            "rounds": [2],
            "accepting_new_products": False,
            "max_maturity_months": 3,
            "current_focus": "liquidity",
        }],
    )
    env = make_env(cli=scheduled)
    assert env.client.accepting_new_products is True

    env.advance_round()

    assert env.client.accepting_new_products is False
    assert env.client.max_maturity_months == 3
    assert env.client.current_focus == "liquidity"


def test_invalid_round_override_fails_before_episode_starts():
    malformed = replace(
        permissive_client(),
        round_overrides=[{"rounds": [2], "not_a_client_field": 1}],
    )
    with pytest.raises(BenchmarkError, match="unsupported fields"):
        make_env(cli=malformed)


def test_static_state_resets_between_rounds():
    env = make_env(dynamic=False, cli=permissive_client())
    payload = env.request_quote(product())
    assert env.submit_design(payload["quote_id"])["accepted"] is True
    # Static cells never carry accepted products into the next decision.
    assert env.portfolio.positions == []
    env.advance_round()
    assert "portfolio_summary" not in env.get_round_brief()


def test_oracle_over_shared_lattice_is_hard_feasible_and_maximal():
    cli = permissive_client()
    snap = snapshots()[0]
    bud = budget()
    result = oracle_best_quote(SMALL_DOMAIN, snap, cli, PortfolioState(), bud)
    assert result is not None
    _, quote = result
    assert quote.hard_pass is True


def test_budget_calibration_is_explicit_and_freezable():
    products = [product(notional=value) for value in (400_000, 1_000_000, 2_000_000)]
    base = RiskBudget(1_000_000, 500_000, 1_000_000, 100_000, 500_000)
    report = calibrate_risk_budget(
        products,
        [(snapshots()[0], client(), PortfolioState())],
        base,
        target=(0.2, 0.5),
        factors=(0.5, 1.0, 2.0, 5.0),
    )
    assert report["selected_factor"] in {0.5, 1.0, 2.0, 5.0}
    assert "freeze" in report["warning"]


def test_carry_sensitivity_reports_pre_registered_tolerance():
    report = carry_sensitivity(product(maturity=6), snapshots()[0], client(), PortfolioState(), budget())
    assert report["shock_bp"] == 25.0
    assert len(report["rows"]) == 3
    assert report["max_fv_change_pct_notional"] >= 0


# ---------------------------------------------------------------------------
# v2: settle_submission is a pure (no-I/O) composition of the three env roles
# ---------------------------------------------------------------------------


def _role(role_id, action, *, degraded=False) -> RoleResponse:
    return RoleResponse(
        role_id=role_id, status="error" if degraded else "ok", action=action,
        payload={}, narrative="", cited_fact_ids=(), raw_hash="", degraded=degraded,
    )


def test_settle_submission_workflow_deal_requires_all_three_affirmatives():
    outcome = settle_submission(
        True,
        _role("trading_desk", "issue"),
        _role("risk_control", "approve"),
        _role("client", "accept"),
    )
    assert isinstance(outcome, WorkflowOutcome)
    assert outcome.workflow_deal is True
    assert outcome.degraded is False


def test_settle_submission_blocks_when_hard_or_any_role_dissents():
    # hard-executable but the desk declines -> no workflow deal.
    dissent = settle_submission(
        True, _role("trading_desk", "decline"),
        _role("risk_control", "approve"), _role("client", "accept"),
    )
    assert dissent.workflow_deal is False
    # all affirm but the product is not hard-executable -> still no deal.
    not_hard = settle_submission(
        False, _role("trading_desk", "issue"),
        _role("risk_control", "approve"), _role("client", "accept"),
    )
    assert not_hard.workflow_deal is False


def test_settle_submission_missing_or_degraded_role_marks_degraded():
    # A missing role (None) is a non-affirmative, degraded signal.
    missing = settle_submission(True, None, _role("risk_control", "approve"), _role("client", "accept"))
    assert missing.workflow_deal is False
    assert missing.degraded is True
    assert missing.desk_action == "decline"
    # A degraded affirmative still counts its action but flags degraded.
    degraded = settle_submission(
        True, _role("trading_desk", "issue", degraded=True),
        _role("risk_control", "approve"), _role("client", "accept"),
    )
    assert degraded.degraded is True
