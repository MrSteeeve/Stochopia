"""End-to-end scripted LLM run through the research tool protocol."""

from __future__ import annotations

import json
import re
from dataclasses import FrozenInstanceError
from datetime import date
from pathlib import Path

import pytest
import yaml

from stochopia.benchmark import (
    BenchmarkCondition,
    BenchmarkError,
    LongHorizonEnvironment,
    MarketSnapshot,
    PortfolioState,
    ProductDomainSpec,
    RiskBudget,
    WorkflowOutcome,
    oracle_best_quote,
)
from stochopia.benchmark_runner import (
    PROTOCOL_POLICIES,
    RUNNER_SYSTEM_PROMPT,
    EpisodeTrace,
    RoundTrace,
    compute_metrics,
    run_episode,
)
from stochopia.env_agents import FrozenEnvAgent
from stochopia.llm import BaseLLMClient, MockLLMClient
from stochopia.pricing import PricingError, QuotePolicy
from stochopia.products import ClientProfile
from stochopia.role_config import InferenceSpec, RetryPolicy, RoleSpecV2

ROOT = Path(__file__).resolve().parents[1]


# A one-candidate lattice keeps runner tests fast and makes the expected
# one-step frontier transparent. It exactly contains product_payload().
RUNNER_DOMAIN = ProductDomainSpec(
    product_types=("vanilla_call",),
    notional_fractions=(.10,),
    maturities=(3,),
    strikes=(.95,),
    barriers=(),
    coupons=(),
    participations=(1.0,),
    principal_protected=(True,),
    version="runner-test-domain-v1",
)


def product_payload(notional=1_000_000):
    return {
        "product_type": "vanilla_call",
        "notional": notional,
        "maturity_months": 3,
        "strike_pct": 0.95,
        "barrier_pct": None,
        "barrier_type": None,
        "coupon_rate": None,
        "participation_rate": 1.0,
        "principal_protected": True,
        "target_client": "c1",
        "pitch": "clear risk disclosure",
        "hedging_plan": "ETF delta hedge",
    }


def _client_profile():
    return ClientProfile(
        "c1", "client", 10_000_000, 1.0, 0.02, "moderate",
        max_maturity_months=12, min_hit_prob=0.25,
    )


def _risk_budget():
    return RiskBudget(50_000_000, 50_000_000, 100_000_000, 10_000_000, 50_000_000)


def environment():
    states = [
        MarketSnapshot("E", 1, date(2023, 1, 31), "CSI500", 6000, 0.02, realized_vol_20d=0.2, source="fixture"),
        MarketSnapshot("E", 2, date(2023, 2, 28), "CSI500", 6100, 0.02, realized_vol_20d=0.2, source="fixture"),
    ]
    return LongHorizonEnvironment(
        states, _client_profile(), _risk_budget(), BenchmarkCondition(False, True),
        domain=RUNNER_DOMAIN,
    )


def single_round_environment():
    states = [
        MarketSnapshot("E", 1, date(2023, 1, 31), "CSI500", 6000, 0.02, realized_vol_20d=0.2, source="fixture"),
    ]
    return LongHorizonEnvironment(
        states, _client_profile(), _risk_budget(), BenchmarkCondition(False, True),
        domain=RUNNER_DOMAIN,
    )


class _ForcedChoiceClient(MockLLMClient):
    """脚本化客户端：预算内的动作按序回放，一旦看到追加的“最后机会”消息，
    改由 `on_forced_prompt` 动态生成回复（例如引用刚被环境报价的 quote_id）。"""

    def __init__(self, scripted: list[str], on_forced_prompt):
        super().__init__()
        self._scripted = list(scripted)
        self._on_forced_prompt = on_forced_prompt
        self.forced_prompt_payloads: list[dict] = []

    async def chat(self, messages, temperature=None, max_tokens=None, seed=None) -> str:
        self.total_usage["calls"] += 1
        last_content = messages[-1]["content"]
        if isinstance(last_content, str) and last_content.startswith("【最后机会】"):
            payload = json.loads(last_content.split("\n", 1)[1])
            self.forced_prompt_payloads.append(payload)
            return self._on_forced_prompt(payload)
        if not self._scripted:
            return "{}"
        if len(self._scripted) > 1:
            return self._scripted.pop(0)
        return self._scripted[0]


@pytest.mark.asyncio
async def test_one_shot_runner_and_metrics():
    response = json.dumps({"action": "submit_product", "product": product_payload(), "explanation": "risk disclosed"})
    trace = await run_episode(environment(), MockLLMClient([response, response]), strategy="one_shot")
    assert len(trace.rounds) == 2
    assert all(item.accepted for item in trace.rounds)
    assert all(item.hard_executable for item in trace.rounds)
    assert all(item.client_contract_pass for item in trace.rounds)
    assert all(item.contract_failures == [] for item in trace.rounds)
    assert all(item.submission_origin == "voluntary" for item in trace.rounds)
    metrics = compute_metrics(trace, oracle_margins=[100_000, 100_000])
    assert metrics["hard_feasibility_rate"] == 1.0
    assert metrics["hard_execution_rate"] == 1.0
    assert metrics["settlement_acceptance_rate"] == 1.0
    assert metrics["hard_execution_rate_given_submission"] == 1.0
    assert metrics["contract_acceptance_rate_given_hard_pass"] == 1.0
    assert metrics["submission_rate"] == 1.0
    assert metrics["voluntary_submission_rate"] == 1.0
    assert metrics["forced_prompt_rate"] == 0.0
    assert metrics["no_submission_rate"] == 0.0
    assert metrics["forced_prompt_margin"] == 0.0
    # Funded-note fixtures issue at par.  Until payoff solve-for is added their
    # inception margin may legitimately be negative; it must not be clamped.
    assert metrics["total_dealer_margin"] != 0.0
    assert metrics["mean_dealer_margin"] == pytest.approx(metrics["total_dealer_margin"] / 2)
    assert metrics["mean_dealer_margin_per_voluntary_accepted_trade"] == pytest.approx(
        metrics["total_dealer_margin"] / 2
    )


def test_protocol_policies_are_immutable_and_one_shot_is_strict():
    policy = PROTOCOL_POLICIES["one_shot"]
    assert policy.allowed_actions == frozenset({"submit_product", "skip_round"})
    assert policy.action_budget == 1
    assert policy.protocol_repair is False
    assert policy.forced_prompt is False
    with pytest.raises(FrozenInstanceError):
        policy.action_budget = 2  # type: ignore[misc]


def test_packaged_runner_prompt_matches_parser_contract():
    assert "purchase_status|risk_appetite|return_hurdle" in RUNNER_SYSTEM_PROMPT
    assert "participation_rate：正数" in RUNNER_SYSTEM_PROMPT
    assert 'barrier_direction：障碍期权可选 "down" | "up"' in RUNNER_SYSTEM_PROMPT
    assert '"action":"consult"' in RUNNER_SYSTEM_PROMPT
    assert "cash_outlay" in RUNNER_SYSTEM_PROMPT


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    json.dumps({"action": "query_client", "topic": "capital"}),
    json.dumps({"action": "request_quote", "product": product_payload()}),
    "not-json",
])
async def test_one_shot_forbidden_or_malformed_action_has_no_repair_or_forced_prompt(raw):
    valid_second_turn = json.dumps({"action": "skip_round"})
    client = MockLLMClient([raw, valid_second_turn])
    env = single_round_environment()

    trace = await run_episode(
        env, client, strategy="one_shot", max_actions_per_round=99,
    )

    round_trace = trace.rounds[0]
    assert client.total_usage["calls"] == 1
    assert len(round_trace.actions) == 1
    assert round_trace.actions[0].action == "protocol_error"
    assert round_trace.submitted is False
    assert round_trace.submission_origin == "none"
    assert env.query_count == 0
    assert env.quote_count == 0
    assert not any(action.action.startswith("forced_prompt") for action in round_trace.actions)


@pytest.mark.asyncio
async def test_one_shot_skip_is_the_only_non_submission_terminal_action():
    client = MockLLMClient([json.dumps({"action": "skip_round"}), "not-used"])
    trace = await run_episode(single_round_environment(), client, strategy="one_shot")
    round_trace = trace.rounds[0]
    assert client.total_usage["calls"] == 1
    assert [action.action for action in round_trace.actions] == ["skip_round"]
    assert round_trace.submitted is False
    assert round_trace.submission_origin == "none"


@pytest.mark.asyncio
async def test_zero_participation_is_recorded_as_protocol_error_not_episode_crash():
    invalid_product = product_payload()
    invalid_product["participation_rate"] = 0
    response = json.dumps({"action": "submit_product", "product": invalid_product})
    client = MockLLMClient([response, json.dumps({"action": "skip_round"})])

    trace = await run_episode(single_round_environment(), client, strategy="one_shot")

    round_trace = trace.rounds[0]
    assert client.total_usage["calls"] == 1
    assert round_trace.submitted is False
    assert [action.action for action in round_trace.actions] == ["protocol_error"]
    assert "participation_rate" in round_trace.actions[0].response["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary_error", [
    PricingError("pricing boundary"),
    ArithmeticError("numeric boundary"),
], ids=["pricing", "arithmetic"])
async def test_tool_boundary_pricing_errors_are_recorded_not_raised(monkeypatch, boundary_error):
    env = single_round_environment()

    def fail_quote(_product):
        raise boundary_error

    monkeypatch.setattr(env, "request_quote", fail_quote)
    response = json.dumps({"action": "submit_product", "product": product_payload()})
    trace = await run_episode(env, MockLLMClient([response]), strategy="one_shot")

    round_trace = trace.rounds[0]
    assert [action.action for action in round_trace.actions] == ["protocol_error"]
    assert "boundary" in round_trace.actions[0].response["error"]
    assert round_trace.submitted is False


@pytest.mark.asyncio
async def test_frontier_uses_environment_nondefault_domain_and_quote_policy():
    policy = QuotePolicy(
        a_f=QuotePolicy().a_f * 4,
        a_v=QuotePolicy().a_v * 4,
        a_p=QuotePolicy().a_p * 4,
        a_b=QuotePolicy().a_b * 4,
    )
    states = [
        MarketSnapshot(
            "E", 1, date(2023, 1, 31), "CSI500", 6000, 0.02,
            realized_vol_20d=0.2, source="fixture",
        ),
    ]
    client_profile = _client_profile()
    client_profile.min_hit_prob = 0.0
    premium_domain = ProductDomainSpec(
        product_types=("vanilla_call",),
        notional_fractions=(.10,),
        maturities=(3,),
        strikes=(.95,),
        barriers=(),
        coupons=(),
        participations=(1.0,),
        principal_protected=(False,),
        version="runner-test-premium-domain-v1",
    )
    env = LongHorizonEnvironment(
        states,
        client_profile,
        _risk_budget(),
        BenchmarkCondition(False, True),
        domain=premium_domain,
        quote_policy=policy,
    )
    expected = oracle_best_quote(
        premium_domain,
        env.snapshot,
        env.client,
        PortfolioState(),
        _risk_budget(),
        policy=policy,
    )
    default_policy_frontier = oracle_best_quote(
        premium_domain,
        env.snapshot,
        env.client,
        PortfolioState(),
        _risk_budget(),
        policy=QuotePolicy(),
    )
    assert expected is not None
    assert default_policy_frontier is not None
    assert expected[1].dealer_margin != pytest.approx(default_policy_frontier[1].dealer_margin)

    premium_product = product_payload()
    premium_product["principal_protected"] = False
    response = json.dumps({
        "action": "submit_product",
        "product": premium_product,
        "explanation": "same custom-policy candidate",
    })
    trace = await run_episode(env, MockLLMClient([response]), strategy="one_shot")
    round_trace = trace.rounds[0]

    assert round_trace.one_step_frontier_margin == pytest.approx(expected[1].dealer_margin)
    assert round_trace.oracle_margin == pytest.approx(expected[1].dealer_margin)
    assert round_trace.dealer_margin == pytest.approx(expected[1].dealer_margin)
    assert compute_metrics(trace)["one_step_attainment"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_round_trace_records_contract_rejection_after_hard_execution():
    client_profile = _client_profile()
    client_profile.min_hit_prob = 0.999999
    states = [
        MarketSnapshot(
            "E", 1, date(2023, 1, 31), "CSI500", 6000, 0.02,
            realized_vol_20d=0.2, source="fixture",
        ),
    ]
    env = LongHorizonEnvironment(
        states,
        client_profile,
        _risk_budget(),
        BenchmarkCondition(False, True),
        domain=RUNNER_DOMAIN,
    )
    response = json.dumps({
        "action": "submit_product",
        "product": product_payload(),
        "explanation": "hard executable but misses client hurdle",
    })

    trace = await run_episode(env, MockLLMClient([response]), strategy="one_shot")
    round_trace = trace.rounds[0]

    assert round_trace.submitted is True
    assert round_trace.hard_executable is True
    assert round_trace.client_contract_pass is False
    assert round_trace.accepted is False
    assert "CONTRACT_HURDLE" in round_trace.contract_failures
    metrics = compute_metrics(trace)
    assert metrics["hard_execution_rate_given_submission"] == 1.0
    assert metrics["hard_execution_rate"] == 1.0
    assert metrics["contract_acceptance_rate_given_hard_pass"] == 0.0
    assert metrics["settlement_acceptance_rate"] == 0.0


@pytest.mark.asyncio
async def test_no_submission_when_model_never_submits_or_skips():
    """预算耗尽后模型仍不提交合法动作（forced_prompt 里依旧不回 submit_design/skip_round）：
    该轮不成交，margin=0，portfolio 不更新，submission_origin="none"。"""
    query = json.dumps({"action": "query_client", "topic": "capital"})
    client = MockLLMClient([query])
    trace = await run_episode(environment(), client, strategy="ledger_archive", max_actions_per_round=1)
    for round_trace in trace.rounds:
        assert round_trace.submitted is False
        assert round_trace.accepted is False
        assert round_trace.submission_origin == "none"
        assert round_trace.dealer_margin == 0.0
        # No request_quote ever happened, so there is nothing to impute either.
        assert round_trace.imputed_counterfactual_margin is None
    metrics = compute_metrics(trace)
    assert metrics["total_dealer_margin"] == 0.0
    assert metrics["mean_dealer_margin"] == 0.0
    assert metrics["forced_prompt_margin"] == 0.0
    assert metrics["hard_feasibility_rate"] == 0.0
    assert metrics["voluntary_submission_rate"] == 0.0
    assert metrics["forced_prompt_rate"] == 0.0
    assert metrics["no_submission_rate"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["quote_and_revise", "ledger_archive"])
async def test_forced_prompt_submission_is_tagged_and_excluded_from_primary_margin(strategy):
    """预算内只 request_quote（不提交），追加的最后机会消息里模型才 submit_design：
    origin="forced_prompt"，真实成交、portfolio 更新，但不计入 total_dealer_margin。"""
    quote_action = json.dumps({"action": "request_quote", "product": product_payload()})

    def on_forced_prompt(payload: dict) -> str:
        assert payload["allowed_actions"] == ["submit_design", "skip_round"]
        quote_id = payload["available_quotes"][0]["quote_id"]
        return json.dumps({"action": "submit_design", "quote_id": quote_id, "explanation": "last-chance submit"})

    client = _ForcedChoiceClient([quote_action], on_forced_prompt)
    env = single_round_environment()
    trace = await run_episode(env, client, strategy=strategy, max_actions_per_round=1)

    assert len(client.forced_prompt_payloads) == 1
    round_trace = trace.rounds[0]
    assert round_trace.submitted is True
    assert round_trace.accepted is True
    assert round_trace.hard_executable is True
    assert round_trace.client_contract_pass is True
    assert round_trace.contract_failures == []
    assert round_trace.submission_origin == "forced_prompt"
    assert round_trace.dealer_margin != 0.0
    assert round_trace.submitted_product is not None
    assert round_trace.submitted_explanation == "last-chance submit"
    assert round_trace.client_brief_snapshot is not None
    # A real submission executed against the environment: portfolio does update.
    assert len(env.portfolio.positions) == 1

    metrics = compute_metrics(trace)
    assert metrics["total_dealer_margin"] == 0.0
    assert metrics["mean_dealer_margin"] == 0.0
    assert metrics["forced_prompt_margin"] == round_trace.dealer_margin
    assert metrics["voluntary_submission_rate"] == 0.0
    assert metrics["forced_prompt_rate"] == 1.0
    assert metrics["no_submission_rate"] == 0.0
    # hard_feasibility_rate keeps its existing "accepted regardless of origin" semantics.
    assert metrics["hard_feasibility_rate"] == 1.0


@pytest.mark.asyncio
async def test_forced_prompt_skip_leaves_round_unsettled():
    """预算内 request_quote 建了可行档案，但最后机会里模型选择 skip_round：
    不成交、origin="none"，但 imputed_counterfactual_margin 仍记录该可行报价的 margin，
    且绝不触碰 portfolio。"""
    quote_action = json.dumps({"action": "request_quote", "product": product_payload()})

    def on_forced_prompt(payload: dict) -> str:
        assert payload["available_quotes"], "archive should hold the previously requested quote"
        return json.dumps({"action": "skip_round"})

    client = _ForcedChoiceClient([quote_action], on_forced_prompt)
    env = single_round_environment()
    trace = await run_episode(env, client, strategy="ledger_archive", max_actions_per_round=1)

    round_trace = trace.rounds[0]
    assert round_trace.submitted is False
    assert round_trace.accepted is False
    assert round_trace.submission_origin == "none"
    assert round_trace.dealer_margin == 0.0
    assert round_trace.imputed_counterfactual_margin is not None
    assert round_trace.imputed_counterfactual_margin != 0.0
    assert round_trace.submitted_product is None
    assert env.portfolio.positions == []


@pytest.mark.asyncio
async def test_forced_prompt_invalid_action_counts_as_none():
    """最后机会里模型返回预算内以外的动作（例如又发一次 request_quote）：无效，仍算 origin="none"。"""
    quote_action = json.dumps({"action": "request_quote", "product": product_payload()})

    def on_forced_prompt(_payload: dict) -> str:
        return json.dumps({"action": "request_quote", "product": product_payload()})

    client = _ForcedChoiceClient([quote_action], on_forced_prompt)
    env = single_round_environment()
    trace = await run_episode(env, client, strategy="ledger_archive", max_actions_per_round=1)

    round_trace = trace.rounds[0]
    assert round_trace.submitted is False
    assert round_trace.submission_origin == "none"
    assert env.portfolio.positions == []


def test_compute_metrics_splits_mixed_origins():
    rounds = [
        RoundTrace(
            round_num=1, submitted=True, accepted=True,
            hard_executable=True, client_contract_pass=True,
            submission_origin="voluntary", dealer_margin=100.0,
            one_step_frontier_margin=200.0,
        ),
        RoundTrace(
            round_num=2, submitted=True, accepted=True,
            hard_executable=True, client_contract_pass=True,
            submission_origin="forced_prompt", dealer_margin=50.0, oracle_margin=100.0,
        ),
        RoundTrace(
            round_num=3, submitted=False, accepted=False,
            submission_origin="none", dealer_margin=0.0, oracle_margin=80.0,
            imputed_counterfactual_margin=40.0,
        ),
        RoundTrace(
            round_num=4, submitted=True, accepted=False,
            hard_executable=True, client_contract_pass=False,
            contract_failures=["CONTRACT_HURDLE"],
            submission_origin="voluntary", dealer_margin=0.0, oracle_margin=60.0,
        ),
    ]
    trace = EpisodeTrace(episode_id="E", condition="partial_dynamic", strategy="ledger_archive", rounds=rounds, usage={})
    metrics = compute_metrics(trace)

    assert metrics["total_dealer_margin"] == 100.0
    assert metrics["mean_dealer_margin"] == 100.0
    assert metrics["mean_dealer_margin_per_voluntary_accepted_trade"] == 100.0
    assert metrics["forced_prompt_margin"] == 50.0
    assert metrics["voluntary_submission_rate"] == pytest.approx(2 / 4)
    assert metrics["forced_prompt_rate"] == pytest.approx(1 / 4)
    assert metrics["no_submission_rate"] == pytest.approx(1 / 4)
    assert metrics["hard_execution_rate_given_submission"] == 1.0
    assert metrics["hard_execution_rate"] == pytest.approx(3 / 4)
    assert metrics["contract_acceptance_rate_given_hard_pass"] == pytest.approx(2 / 3)
    assert metrics["settlement_acceptance_rate"] == pytest.approx(2 / 4)
    # Deprecated alias: final accepted rounds regardless of origin (1 and 2).
    assert metrics["hard_feasibility_rate"] == pytest.approx(2 / 4)
    # one_step_attainment: only the voluntary+accepted round (round 1) enters the ratio.
    assert metrics["one_step_attainment"] == pytest.approx(100.0 / 200.0)
    assert "oracle_margin_attainment" not in metrics
    assert "forced_submission_rate" not in metrics


def test_compute_metrics_separates_hard_contract_and_settlement_gates():
    rounds = [
        RoundTrace(
            round_num=1, submitted=True, accepted=True,
            hard_executable=True, client_contract_pass=True,
            submission_origin="voluntary", dealer_margin=10.0,
        ),
        RoundTrace(
            round_num=2, submitted=True, accepted=False,
            hard_executable=True, client_contract_pass=False,
            contract_failures=["CONTRACT_HURDLE"], submission_origin="voluntary",
        ),
        RoundTrace(
            round_num=3, submitted=True, accepted=False,
            hard_executable=False, client_contract_pass=True,
            hard_failures=["PORTFOLIO_NOTIONAL"], submission_origin="voluntary",
        ),
        RoundTrace(round_num=4),
    ]
    metrics = compute_metrics(EpisodeTrace("E", "full_static", "one_shot", rounds, {}))

    assert metrics["hard_execution_rate_given_submission"] == pytest.approx(2 / 3)
    assert metrics["hard_execution_rate"] == pytest.approx(2 / 4)
    assert metrics["contract_acceptance_rate_given_hard_pass"] == pytest.approx(1 / 2)
    assert metrics["settlement_acceptance_rate"] == pytest.approx(1 / 4)
    assert metrics["hard_feasibility_rate"] == metrics["settlement_acceptance_rate"]


def test_conditional_execution_metrics_are_none_without_denominators():
    trace = EpisodeTrace("E", "full_static", "one_shot", [RoundTrace(round_num=1)], {})
    metrics = compute_metrics(trace)
    assert metrics["hard_execution_rate_given_submission"] is None
    assert metrics["hard_execution_rate"] == 0.0
    assert metrics["contract_acceptance_rate_given_hard_pass"] is None
    assert metrics["settlement_acceptance_rate"] == 0.0


def test_compute_metrics_rejects_attainment_above_one_step_frontier():
    trace = EpisodeTrace(
        "E",
        "full_static",
        "one_shot",
        [RoundTrace(
            round_num=1,
            submitted=True,
            accepted=True,
            hard_executable=True,
            client_contract_pass=True,
            submission_origin="voluntary",
            dealer_margin=100.000001,
            one_step_frontier_margin=100.0,
        )],
        {},
    )
    with pytest.raises(BenchmarkError, match="exceeds the one-step frontier"):
        compute_metrics(trace)


@pytest.mark.asyncio
async def test_ledger_archive_no_forced_completion_on_invalid_responses():
    """两次响应都不是合法提交（第二次连最后机会也不合法）：不再有环境替提交，
    该轮保持 origin="none"，但 archive 中的可行报价仍被记为 imputed 诊断值。"""
    quote = json.dumps({"action": "request_quote", "product": product_payload()})
    client = MockLLMClient([quote, '{"action":"unknown"}'])
    trace = await run_episode(environment(), client, strategy="ledger_archive", max_actions_per_round=2)
    round_trace = trace.rounds[0]
    assert round_trace.submitted is False
    assert round_trace.accepted is False
    assert round_trace.submission_origin == "none"
    assert round_trace.dealer_margin == 0.0
    assert round_trace.imputed_counterfactual_margin is not None
    assert round_trace.imputed_counterfactual_margin != 0.0


# ===========================================================================
# v2 dialogue layer: env-role consult, LLM client query, dual settlement.
# All of these are opt-in via env_agents; the tests above (env_agents=None)
# are the byte-for-byte fallback-equivalence evidence.
# ===========================================================================


class _RaisingClient(BaseLLMClient):
    """Always raises: exercises the FrozenEnvAgent degraded fallback path."""

    async def chat(self, messages, temperature=None, max_tokens=None, seed=None) -> str:
        raise RuntimeError("simulated provider outage")


class _EchoQuoteDeskClient(BaseLLMClient):
    """A desk that issues, echoing back the quote_id from the injected facts.

    A fixed MockLLMClient can't cite the run-time quote_id, so this mock reads
    it from the request payload (exactly what a real desk LLM would do) to
    exercise a genuine workflow issue with valid grounding.
    """

    async def chat(self, messages, temperature=None, max_tokens=None, seed=None) -> str:
        self.total_usage["calls"] += 1
        match = re.search(r'"quote_id":\s*"(Q-[0-9a-f]+)"', messages[-1]["content"])
        if match:
            return json.dumps({
                "action": "issue", "quote_id": match.group(1),
                "hedge_tags": [], "suggestions": [], "narrative": "",
            })
        return json.dumps({
            "action": "request_revision", "quote_id": None,
            "hedge_tags": [], "suggestions": [], "narrative": "",
        })


def _env_spec(role: str, schema: str, failure: str, seed_offset: int) -> RoleSpecV2:
    return RoleSpecV2(
        id=role, protocol_version="stochopia.csi.legacy.v1", role=role,
        system_prompt_file=Path("dummy.md"), system_prompt_sha256="",
        inference=InferenceSpec("mock", 0.0, 500, 30.0, "derived", seed_offset),
        retry=RetryPolicy(transport_retries=0, format_retries=1),
        output_schema=schema, tools=(), max_calls_per_round=3, history_scope="episode",
        numeric_authority="supplied_facts_only", failure_policy=failure,
    )


def _make_agents(*, client=None, risk=None, desk=None) -> dict[str, FrozenEnvAgent]:
    agents: dict[str, FrozenEnvAgent] = {}
    if client is not None:
        agents["client"] = FrozenEnvAgent(
            _env_spec("client", "stochopia.role.client-response.v1", "abstain", 2000), client, system_prompt="SYS")
    if risk is not None:
        agents["risk_control"] = FrozenEnvAgent(
            _env_spec("risk_control", "stochopia.role.risk-response.v1", "escalate", 3000), risk, system_prompt="SYS")
    if desk is not None:
        agents["trading_desk"] = FrozenEnvAgent(
            _env_spec("trading_desk", "stochopia.role.trading-desk-response.v1", "decline", 4000), desk, system_prompt="SYS")
    return agents


def env_with_agents(agents, *, full: bool = False) -> LongHorizonEnvironment:
    states = [
        MarketSnapshot("E", 1, date(2023, 1, 31), "CSI500", 6000, 0.02, realized_vol_20d=0.2, source="fixture"),
    ]
    return LongHorizonEnvironment(
        states, _client_profile(), _risk_budget(),
        BenchmarkCondition(full, True), domain=RUNNER_DOMAIN, env_agents=agents,
    )


_CLIENT_ACCEPT = '{"action":"accept","disclose_fields":[],"reason_codes":[],"narrative":""}'
_RISK_APPROVE = '{"action":"approve","check_refs":[],"suggestions":[],"narrative":""}'


@pytest.mark.asyncio
async def test_consult_with_draft_returns_block_without_spending_quote_budget():
    desk = MockLLMClient(
        ['{"action":"request_revision","quote_id":null,"hedge_tags":[],"suggestions":["缩短期限"],"narrative":""}'])
    env = env_with_agents(_make_agents(desk=desk))
    draft = {k: v for k, v in product_payload().items() if k != "target_client"}
    resp = await env.consult("trading_desk", "这个结构能对冲吗？", draft)
    assert resp["degraded"] is False
    assert resp["deterministic_block"] is not None
    assert "hard_pass" in resp["deterministic_block"]
    assert resp["action"] == "request_revision"
    # A consultative quote never spends the request_quote budget nor archives.
    assert env.quote_count == 0
    assert env.quotes == {}
    assert env.consult_count == 1


@pytest.mark.asyncio
async def test_text_consult_goes_to_llm_without_deterministic_block():
    desk = MockLLMClient(
        ['{"action":"request_revision","quote_id":null,"hedge_tags":[],"suggestions":[],"narrative":""}'])
    env = env_with_agents(_make_agents(desk=desk))
    resp = await env.consult("trading_desk", "现在流动性怎么样？")
    assert resp["deterministic_block"] is None
    assert resp["degraded"] is False
    assert desk.total_usage["calls"] == 1


@pytest.mark.asyncio
async def test_consult_budget_and_missing_role_return_error_never_raise():
    desk = MockLLMClient(
        ['{"action":"request_revision","quote_id":null,"hedge_tags":[],"suggestions":[],"narrative":""}'])
    env = env_with_agents(_make_agents(desk=desk))
    # Missing role: graceful error dict.
    missing = await env.consult("risk_control", "预检一下")
    assert missing["degraded"] is True
    assert "error" in missing
    # Exhaust the consult budget (default 3).
    for _ in range(3):
        await env.consult("trading_desk", "?")
    exhausted = await env.consult("trading_desk", "?")
    assert exhausted["degraded"] is True
    assert "budget" in exhausted["error"]


@pytest.mark.asyncio
async def test_consult_degraded_does_not_break_the_round():
    agents = _make_agents(
        client=MockLLMClient([_CLIENT_ACCEPT]),
        risk=MockLLMClient([_RISK_APPROVE]),
        desk=_RaisingClient(),
    )
    env = env_with_agents(agents)
    consult = json.dumps({"action": "consult", "role": "trading_desk", "message": "?"})
    submit = json.dumps({"action": "submit_product", "product": product_payload(), "explanation": "ok"})
    trace = await run_episode(env, MockLLMClient([consult, submit]),
                              strategy="ledger_archive", max_actions_per_round=3)
    rt = trace.rounds[0]
    assert rt.consult_count == 1
    assert rt.degraded_consult_count == 1
    # The round still reached a submission after the degraded consult.
    assert rt.submitted is True


@pytest.mark.asyncio
async def test_workflow_deal_true_when_all_roles_affirm():
    env = env_with_agents(_make_agents(
        client=MockLLMClient([_CLIENT_ACCEPT]),
        risk=MockLLMClient([_RISK_APPROVE]),
        desk=_EchoQuoteDeskClient(),
    ))
    submit = json.dumps({"action": "submit_product", "product": product_payload(), "explanation": "ok"})
    trace = await run_episode(env, MockLLMClient([submit]),
                              strategy="ledger_archive", max_actions_per_round=2)
    rt = trace.rounds[0]
    assert rt.accepted is True
    assert rt.workflow is not None
    assert rt.workflow.workflow_deal is True
    assert (rt.workflow.desk_action, rt.workflow.risk_action, rt.workflow.client_action) == (
        "issue", "approve", "accept")
    assert rt.workflow.degraded is False


@pytest.mark.asyncio
async def test_workflow_escalate_blocks_deal_but_not_primary_settlement():
    escalate = '{"action":"escalate","check_refs":[],"suggestions":[],"narrative":""}'
    env = env_with_agents(_make_agents(
        client=MockLLMClient([_CLIENT_ACCEPT]),
        risk=MockLLMClient([escalate]),
        desk=_EchoQuoteDeskClient(),
    ))
    submit = json.dumps({"action": "submit_product", "product": product_payload(), "explanation": "ok"})
    trace = await run_episode(env, MockLLMClient([submit]),
                              strategy="ledger_archive", max_actions_per_round=2)
    rt = trace.rounds[0]
    # Primary settlement is deterministic and unaffected by the risk LLM.
    assert rt.accepted is True
    assert rt.dealer_margin != 0.0
    assert rt.workflow.workflow_deal is False
    assert rt.workflow.risk_action == "escalate"


@pytest.mark.asyncio
async def test_partial_query_client_routes_to_llm_with_injected_numbers():
    # capital=10_000_000 is injected as a grounding fact; the client echoes it.
    grounded = ('{"action":"answer","disclose_fields":["capital"],"reason_codes":[],'
                '"narrative":"我的可投资金大约 10000000 元"}')
    env = env_with_agents(_make_agents(client=MockLLMClient([grounded])), full=False)
    assert env.has_client_llm() is True
    resp = await env.query_client_llm("capital")
    assert resp["degraded"] is False
    assert resp["action"] == "answer"
    assert "10000000" in resp["reply"]
    assert "capital" in resp["disclosed_fields"]


@pytest.mark.asyncio
async def test_llm_client_query_degrades_on_ungrounded_number():
    ungrounded = '{"action":"answer","disclose_fields":[],"reason_codes":[],"narrative":"我有 777 元"}'
    env = env_with_agents(_make_agents(client=MockLLMClient([ungrounded, ungrounded])), full=False)
    resp = await env.query_client_llm("capital")
    assert resp["degraded"] is True


@pytest.mark.asyncio
async def test_full_information_query_client_stays_deterministic():
    client_c = MockLLMClient([_CLIENT_ACCEPT])
    env = env_with_agents(_make_agents(client=client_c), full=True)
    assert env.has_client_llm() is False
    resp = env.query_client("capital")
    assert resp["answer"] == "already disclosed in the full client profile"
    assert client_c.total_usage["calls"] == 0


def test_partial_query_client_without_agents_is_deterministic():
    env = single_round_environment()  # env_agents=None
    assert env.has_client_llm() is False
    assert env.query_client("capital")["answer"] == 10_000_000


def test_compute_metrics_secondary_dialogue_metrics():
    rounds = [
        RoundTrace(
            round_num=1, submitted=True, accepted=True, submission_origin="voluntary",
            dealer_margin=100.0, consult_count=2, degraded_consult_count=1,
            workflow=WorkflowOutcome(True, "issue", "approve", "accept", False),
        ),
        RoundTrace(
            round_num=2, submitted=True, accepted=False, submission_origin="voluntary",
            dealer_margin=0.0, consult_count=1, degraded_consult_count=0,
            workflow=WorkflowOutcome(True, "issue", "approve", "accept", False),
        ),
    ]
    trace = EpisodeTrace("E", "partial_dynamic", "ledger_archive", rounds, {})
    m = compute_metrics(trace)
    assert m["workflow_deal_rate"] == 1.0
    # round 2: workflow_deal True but the primary settlement rejected -> divergence.
    assert m["llm_action_vs_formal_truth"] == pytest.approx(0.5)
    assert m["degraded_consult_rate"] == pytest.approx(1 / 3)


def test_secondary_metrics_are_none_without_dialogue_layer():
    rounds = [RoundTrace(round_num=1, submitted=True, accepted=True,
                         submission_origin="voluntary", dealer_margin=1.0)]
    trace = EpisodeTrace("E", "full_static", "one_shot", rounds, {})
    m = compute_metrics(trace)
    assert m["workflow_deal_rate"] is None
    assert m["llm_action_vs_formal_truth"] is None
    assert m["degraded_consult_rate"] is None


def test_failure_metric_does_not_double_count_submission_alias_lists():
    rounds = [
        RoundTrace(
            round_num=1,
            hard_failures=["PORTFOLIO_NOTIONAL"],
            all_quote_failures=["PORTFOLIO_NOTIONAL"],
        ),
        RoundTrace(
            round_num=2,
            hard_failures=["PORTFOLIO_NOTIONAL"],
            all_quote_failures=["PORTFOLIO_NOTIONAL"],
        ),
    ]
    metrics = compute_metrics(
        EpisodeTrace("E", "full_dynamic", "quote_and_revise", rounds, {})
    )

    assert metrics["failure_counts"] == {"PORTFOLIO_NOTIONAL": 2}
    assert metrics["repeated_hard_violations"] == 1


# --- CLI wiring: fail-fast roles config and env-agent construction ---------


def test_cli_roles_config_load_failure_is_fail_fast(tmp_path):
    from stochopia.benchmark_cli import _load_env_role_specs
    from stochopia.llm import load_model_registry

    registry = load_model_registry(ROOT / "config" / "models.yaml")
    (tmp_path / "p.md").write_text("prompt", encoding="utf-8")
    bad = tmp_path / "roles.yaml"
    bad.write_text(yaml.safe_dump({
        "protocol_version": "stochopia.csi.legacy.v1",
        "roles": {"client_main": {
            "role": "client", "model_ref": "mock", "system_prompt_file": "p.md",
            "output_schema": "stochopia.role.client-response.v1", "numeric_authority": "supplied_facts_only",
            "bogus_field": 1,
        }},
    }, allow_unicode=True), encoding="utf-8")
    with pytest.raises(SystemExit):
        _load_env_role_specs(bad, registry)


def test_cli_builds_env_agents_from_real_config():
    from stochopia.benchmark_cli import _load_env_role_specs, _make_env_agents, _roles_config_meta
    from stochopia.llm import load_model_registry

    registry = load_model_registry(ROOT / "config" / "models.yaml")
    specs = _load_env_role_specs(ROOT / "config" / "benchmark_roles.yaml", registry)
    agents = _make_env_agents(specs, registry, None)
    assert set(agents) == {"client", "risk_control", "trading_desk"}
    npc_lineup_id, roles_sha = _roles_config_meta(ROOT / "config" / "benchmark_roles.yaml")
    assert npc_lineup_id == "stochopia.npc-lineup.fixed.v1"
    assert len(roles_sha) == 64
