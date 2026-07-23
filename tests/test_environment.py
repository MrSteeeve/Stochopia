"""Contract tests for the v3-spine/Level-0 synchronous environment."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import pytest

from mirage.benchmark import MarketSnapshot, ProductDomainSpec, RiskBudget
from mirage.environment import (
    AskClient,
    EpisodeTask,
    InvalidAction,
    MirageStructurerEnv,
    RequestQuote,
    ScalarizationSpec,
    ScalarizedMirageEnv,
    Skip,
    SubmitDesign,
    SubmitProduct,
    TrajectoryRecorder,
    V3_PUBLIC_CLIENT_ALIAS,
)
from mirage.pricing import QuotePolicy
from mirage.products import ClientProfile, ProductSpec


def _task() -> EpisodeTask:
    snapshots = (
        MarketSnapshot(
            "v3-test",
            1,
            date(2023, 1, 31),
            "CSI500",
            6000.0,
            0.02,
            realized_vol_20d=0.20,
            source="test",
        ),
        MarketSnapshot(
            "v3-test",
            2,
            date(2023, 2, 28),
            "CSI500",
            6100.0,
            0.02,
            realized_vol_20d=0.21,
            source="test",
        ),
        MarketSnapshot(
            "v3-test",
            3,
            date(2023, 3, 31),
            "CSI500",
            6050.0,
            0.02,
            realized_vol_20d=0.22,
            source="test",
        ),
    )
    client = ClientProfile(
        id="client-hidden",
        name="Hidden Client",
        capital=10_000_000.0,
        max_loss_pct=1.0,
        min_return_pct=0.02,
        risk_appetite="moderate",
        max_maturity_months=12,
        min_hit_prob=0.25,
    )
    return EpisodeTask(
        snapshots=snapshots,
        client=client,
        risk_budget=RiskBudget(
            notional=50_000_000.0,
            net_delta=50_000_000.0,
            gross_delta=100_000_000.0,
            net_vega=10_000_000.0,
            stress_loss=50_000_000.0,
        ),
        domain=ProductDomainSpec(),
        quote_policy=QuotePolicy(diagnostic_paths=256),
        task_seed=17,
        schema="mirage.environment.test.v3",
        version="test-task-v1",
    )


def _product() -> ProductSpec:
    return ProductSpec(
        product_type="vanilla_call",
        notional=1_000_000.0,
        maturity_months=3,
        strike_pct=0.95,
        barrier_pct=None,
        barrier_type=None,
        coupon_rate=None,
        participation_rate=1.0,
        principal_protected=True,
        target_client=V3_PUBLIC_CLIENT_ALIAS,
        pitch="clear risk disclosure",
        hedging_plan="listed futures delta hedge",
    )


def test_reset_is_deterministic_and_rebuilds_state():
    env = MirageStructurerEnv(_task())
    first_observation, first_info = env.reset()
    env.step(AskClient("capital"))

    second_observation, second_info = env.reset()

    assert second_observation == first_observation
    assert second_info["state_hash"] == first_info["state_hash"]
    assert second_info["seed"] == 17
    assert second_observation.step_index == 0
    assert second_observation.disclosed_client == {}


def test_public_action_schema_is_complete_and_matches_runtime_domain():
    env = MirageStructurerEnv(_task())
    observation, _ = env.reset()
    schema = env.action_schema

    assert schema["schema_version"] == "mirage.environment.action-schema.v3"
    assert schema["product_domain_version"] == "csi-domain-v3-action-contract"
    assert len(schema["client_topic_enum"]) == 9
    assert set(schema["client_topic_enum"]) == {
        "capital",
        "loss_tolerance",
        "maturity",
        "product_types",
        "protection",
        "preferences",
        "purchase_status",
        "risk_appetite",
        "return_hurdle",
    }
    assert "custom" not in schema["allowed_product_types"]
    assert _product().notional in schema["notional_rule"]["allowed_values"]
    assert _product().maturity_months in schema["maturities"]
    assert _product().strike_pct in schema["strikes"]
    assert schema["combination_rules"]["autocallable"]["protected"][
        "barrier_pct"
    ] == [None]
    assert "request_quote" in observation.available_actions

    transition = env.step(RequestQuote(_product()))
    assert transition.constraint_signals.action_valid is True
    assert transition.constraint_signals.hard_pass is True


def test_public_action_schema_does_not_encode_hidden_client_capital():
    first = _task()
    rich_client = ClientProfile(
        **{
            **asdict(first.client),
            "id": "another-hidden-client",
            "capital": 40_000_000.0,
        }
    )
    second = EpisodeTask(
        snapshots=first.snapshots,
        client=rich_client,
        risk_budget=first.risk_budget,
        domain=ProductDomainSpec(),
        quote_policy=first.quote_policy,
        task_seed=99,
        schema=first.schema,
        version=first.version,
    )
    first_env = MirageStructurerEnv(first)
    second_env = MirageStructurerEnv(second)
    first_observation, _ = first_env.reset()
    second_observation, _ = second_env.reset()

    assert first_env.action_schema == second_env.action_schema
    assert first_env.action_schema["target_client"] == V3_PUBLIC_CLIENT_ALIAS
    assert (
        first_env.action_schema["notional_rule"]["base_amount"]
        == second_env.action_schema["notional_rule"]["base_amount"]
        == 10_000_000.0
    )
    assert first_observation.public_task_id == second_observation.public_task_id
    encoded = json.dumps(asdict(second_observation), sort_keys=True)
    assert "another-hidden-client" not in encoded
    assert "40000000" not in encoded


def test_task_manifest_detaches_reset_from_later_client_mutation():
    task = _task()
    original_capital = task.client.capital
    original_hash = task.task_hash
    task.client.capital = 1.0

    env = MirageStructurerEnv(task, expose_privileged_info=True)
    _, info = env.reset()

    assert info["privileged_state"]["client"]["capital"] == original_capital
    assert task.task_hash == original_hash


def test_typed_quote_then_submit_auto_advances_round():
    env = MirageStructurerEnv(_task())
    env.reset()

    quoted = env.step(RequestQuote(_product()))
    quote_id = quoted.tool_result["payload"]["quote_id"]

    assert quote_id == "quote-r1-n1"
    assert (
        quoted.tool_result["payload"]["valid_for_state"]
        == quoted.observation.state_version
    )
    assert quoted.action.type == "request_quote"
    assert quoted.constraint_signals.hard_pass is True
    assert quoted.reward_components.quote_cost.value < 0
    assert quoted.reward_components.quote_cost.available is True
    assert quoted.reward_components.client_utility.available is True
    assert quoted.reward_components.client_utility.value == pytest.approx(0.0)
    assert quoted.terminated is False

    submitted = env.step(SubmitDesign(quote_id, "risks disclosed"))

    assert submitted.action.type == "submit_design"
    assert submitted.tool_result["payload"]["quote_id"] == quote_id
    assert submitted.constraint_signals.accepted is True
    assert submitted.reward_components.dealer_economics.available is True
    assert submitted.reward_components.dealer_economics.value == pytest.approx(
        submitted.tool_result["payload"]["dealer_margin"]
    )
    assert submitted.observation.round_num == 2
    assert submitted.observation.last_event["advanced_to_round"] == 2
    assert submitted.terminated is False
    assert submitted.state_hash_before == quoted.state_hash_after


def test_submit_product_is_atomic_typed_quote_and_submission():
    env = MirageStructurerEnv(_task())
    env.reset()

    transition = env.step(SubmitProduct(_product(), "one-shot"))

    assert transition.tool_result["type"] == "quote_and_submission"
    assert transition.constraint_signals.accepted is True
    assert transition.reward_components.quote_cost.value < 0
    assert transition.observation.round_num == 2
    encoded = json.dumps(transition.tool_result, sort_keys=True)
    assert "client_account" not in encoded
    assert "dealer_account" not in encoded
    assert "client-hidden" not in encoded


def test_rejected_submission_cannot_earn_counterfactual_dealer_margin():
    base = _task()
    restrictive_budget = RiskBudget(
        notional=100.0,
        net_delta=base.risk_budget.net_delta,
        gross_delta=base.risk_budget.gross_delta,
        net_vega=base.risk_budget.net_vega,
        stress_loss=base.risk_budget.stress_loss,
    )
    task = EpisodeTask(
        snapshots=base.snapshots,
        client=base.client,
        risk_budget=restrictive_budget,
        domain=base.domain,
        quote_policy=base.quote_policy,
        task_seed=base.task_seed,
        schema=base.schema,
        version=base.version,
    )
    env = MirageStructurerEnv(task)
    env.reset()

    transition = env.step(SubmitProduct(_product(), "not accepted"))

    assert transition.constraint_signals.accepted is False
    assert transition.tool_result["submission"]["dealer_margin"] == 0.0
    assert transition.reward_components.dealer_economics.value == pytest.approx(
        0.0
    )
    assert (
        transition.reward_components.dealer_economics.provenance
        == "accepted-inception-margin-v2"
    )


def test_horizon_liquidation_closes_accounts_and_enters_reward_summary():
    env = MirageStructurerEnv(_task(), expose_privileged_info=True)
    env.reset()

    issued = env.step(SubmitProduct(_product(), "one-shot"))
    terminal = env.step(Skip("finish horizon"))

    assert issued.terminated is False
    assert issued.reward_components.capital_efficiency.value < 0
    assert issued.reward_components.risk_change.value <= 0
    assert issued.reward_components.relationship_delta.available is False
    assert terminal.terminated is True
    assert terminal.tool_result["open_at_horizon"] == []
    assert terminal.tool_result["horizon_liquidations"]
    assert "horizon_liquidated_position_ids" not in terminal.tool_result
    assert "position_id" not in json.dumps(terminal.tool_result)
    assert "event_id" not in json.dumps(terminal.tool_result)
    assert "v3-test" not in json.dumps(terminal.tool_result)
    assert terminal.reward_components.terminal_lifecycle_pnl.available is True
    assert terminal.reward_components.capital_efficiency.value < 0

    portfolio = terminal.info["privileged_state"]["portfolio"]
    assert portfolio["positions"] == []
    assert portfolio["client_account"]["locked_cash"] == pytest.approx(0.0)
    assert portfolio["dealer_account"]["realised_hedged_pnl"] == pytest.approx(
        terminal.reward_components.terminal_lifecycle_pnl.value
    )
    summary = terminal.info["episode_summary"]
    assert summary["reward"]["total_steps"] == 2
    assert summary["reward"]["raw_totals"][
        "terminal_lifecycle_pnl"
    ] == pytest.approx(
        terminal.reward_components.terminal_lifecycle_pnl.value
    )
    assert summary["constraints"]["steps"] == 2
    assert summary["constraints"]["valid_actions"] == 2


def test_query_cost_is_emitted_on_every_query_step():
    env = MirageStructurerEnv(_task())
    env.reset()

    capital = env.step(AskClient("capital"))
    maturity = env.step(AskClient("maturity"))

    assert capital.reward_components.query_cost.value == pytest.approx(-0.01)
    assert maturity.reward_components.query_cost.value == pytest.approx(-0.01)
    assert sum(
        t.reward_components.query_cost.value for t in (capital, maturity)
    ) == pytest.approx(-0.02)
    assert maturity.previous_observation == capital.observation
    assert maturity.state_hash_before == capital.state_hash_after
    assert maturity.observation.disclosed_client == {
        "capital": 10_000_000.0,
        "maturity": 12,
    }


def test_direct_typed_product_pricing_error_becomes_constraint_signal():
    """Typed callers can bypass JSON parsing; numeric failures must not crash step()."""
    env = MirageStructurerEnv(_task())
    env.reset()
    invalid = ProductSpec(
        **{
            **asdict(_product()),
            "participation_rate": 0.0,
        }
    )

    transition = env.step(SubmitProduct(invalid, "invalid boundary"))

    assert transition.constraint_signals.action_valid is False
    assert "participation_rate" in transition.constraint_signals.error
    assert transition.reward_components.operational_cost.value < 0
    assert transition.terminated is False


def test_policy_cannot_forge_environment_maintained_product_state():
    env = MirageStructurerEnv(_task())
    env.reset()
    forged = ProductSpec(**{**asdict(_product()), "elapsed_months": 3})

    transition = env.step(SubmitProduct(forged, "pretend matured"))

    assert transition.constraint_signals.action_valid is False
    assert "environment-maintained" in transition.constraint_signals.error
    assert "elapsed_months" in transition.constraint_signals.error
    assert transition.observation.action_budget["quotes_left"] == 3


def test_quote_binds_product_snapshot_against_post_quote_mutation():
    env = MirageStructurerEnv(_task(), expose_privileged_info=True)
    env.reset()
    mutable = _product()
    quoted = env.step(RequestQuote(mutable))
    quote_id = quoted.tool_result["payload"]["quote_id"]

    mutable.notional = 9_000_000.0
    mutable.maturity_months = 60
    mutable.strike_pct = 10.0
    submitted = env.step(SubmitDesign(quote_id, "mutated after quote"))

    assert submitted.constraint_signals.accepted is True
    portfolio = submitted.info["privileged_state"]["portfolio"]
    assert portfolio["positions"][0]["product"]["notional"] == 1_000_000.0
    assert portfolio["positions"][0]["product"]["maturity_months"] == 3
    assert quoted.action.product.notional == 1_000_000.0


def test_skip_advances_then_terminates_on_final_round():
    env = MirageStructurerEnv(_task())
    env.reset()

    first = env.step(Skip("no fit"))
    second = env.step(Skip("still no fit"))

    assert first.terminated is False
    assert first.observation.round_num == 2
    assert second.terminated is True
    assert second.truncated is False
    assert second.observation.terminated is True
    assert second.observation.available_actions == ()
    with pytest.raises(RuntimeError, match="ended"):
        env.step(Skip())


def test_privileged_state_is_hidden_by_default_and_evaluator_opt_in_is_explicit():
    env = MirageStructurerEnv(_task())
    observation, info = env.reset()

    public = asdict(observation)
    encoded = json.dumps(public, ensure_ascii=False, sort_keys=True)
    assert "privileged_state" not in public
    assert "client_constraints" not in encoded
    assert "10,000,000" not in encoded
    assert "10000000" not in encoded
    assert observation.disclosed_client == {}
    assert "privileged_state" not in info

    evaluator = MirageStructurerEnv(_task(), expose_privileged_info=True)
    _, evaluator_info = evaluator.reset()
    assert evaluator_info["privileged_state"]["client"]["capital"] == 10_000_000.0
    assert evaluator_info["privileged_state"]["client"]["id"] == "client-hidden"


def test_invalid_actions_hit_a_finite_round_time_limit():
    env = MirageStructurerEnv(_task(), max_steps_per_round=2)
    env.reset()

    first = env.step(InvalidAction("bad"))
    second = env.step(InvalidAction("still bad"))

    assert first.truncated is False
    assert second.terminated is False
    assert second.truncated is True
    assert second.observation.terminated is False
    assert second.observation.truncated is True
    assert second.observation.available_actions == ()
    assert second.tool_result["time_limit"]["max_steps_per_round"] == 2
    with pytest.raises(RuntimeError, match="ended"):
        env.step(Skip())


def test_available_action_mask_removes_exhausted_query_and_quote_actions():
    env = MirageStructurerEnv(_task(), max_steps_per_round=12)
    observation, _ = env.reset()
    assert {"ask_client", "request_quote", "submit_product"} <= set(
        observation.available_actions
    )

    for topic in ("capital", "maturity", "protection"):
        transition = env.step(AskClient(topic))
    assert "ask_client" not in transition.observation.available_actions

    for _ in range(3):
        transition = env.step(RequestQuote(_product()))
    assert "request_quote" not in transition.observation.available_actions
    assert "submit_product" not in transition.observation.available_actions
    assert "submit_design" in transition.observation.available_actions


def test_state_hash_commits_reward_config_and_reset_options():
    base = MirageStructurerEnv(_task(), query_cost=-0.01)
    costly = MirageStructurerEnv(_task(), query_cost=-99.0)
    _, base_info = base.reset(options={"run_label": "a"})
    _, costly_info = costly.reset(options={"run_label": "a"})
    assert base_info["state_hash"] != costly_info["state_hash"]

    _, changed_options = base.reset(options={"run_label": "b"})
    assert base_info["state_hash"] != changed_options["state_hash"]
    assert (
        TrajectoryRecorder(base).metadata.environment_config_hash
        != TrajectoryRecorder(costly).metadata.environment_config_hash
    )


def test_unimplemented_curriculum_option_fails_closed():
    env = MirageStructurerEnv(_task())

    with pytest.raises(ValueError, match="TaskGenerator"):
        env.reset(options={"curriculum": "easy"})


def test_scalarized_wrapper_requires_explicit_non_double_counting_spec():
    with pytest.raises(ValueError, match="cannot combine"):
        ScalarizationSpec(
            {
                "dealer_economics": 1.0,
                "terminal_lifecycle_pnl": 1.0,
            }
        )

    wrapped = ScalarizedMirageEnv(
        MirageStructurerEnv(_task()),
        ScalarizationSpec({"query_cost": 1.0}),
    )
    _, reset_info = wrapped.reset(seed=5)
    observation, reward, terminated, truncated, info = wrapped.step(
        AskClient("capital")
    )

    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False
    assert observation.disclosed_client["capital"] == 10_000_000.0
    assert reset_info["scalarization_hash"] == info["scalarization_hash"]
    assert info["reward_components"]["query_cost"]["value"] == pytest.approx(
        -0.01
    )


def test_trajectory_records_full_transitions_and_verifies_chain(tmp_path):
    env = MirageStructurerEnv(_task(), expose_privileged_info=True)
    _, reset_info = env.reset()
    recorder = TrajectoryRecorder(env, initial_state_hash=reset_info["state_hash"])

    first = env.step(Skip("round one"))
    second = env.step(Skip("round two"))
    recorder.record(first, policy_metadata={"policy": "test"})
    recorder.record(second, policy_metadata={"policy": "test"})

    assert recorder.verify_hash_chain() is True
    path = recorder.save(tmp_path / "trajectory.json")
    assert TrajectoryRecorder.verify(path) is True

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["metadata"]["schema_version"] == "mirage.environment.test.v3"
    assert saved["metadata"]["environment_version"] == env.environment_version
    assert saved["metadata"]["pricing_version"]
    assert saved["metadata"]["task_version"] == "test-task-v1"
    assert saved["metadata"]["package_version"] == "0.2.0"
    assert saved["metadata"]["installed_distribution_version"]
    assert saved["metadata"]["environment_config_hash"]
    assert saved["environment_configuration"] == env.configuration
    assert saved["metadata"]["implementation_hash"]
    assert isinstance(saved["metadata"]["worktree_dirty"], (bool, type(None)))
    assert saved["metadata"]["worktree_state_hash"]
    assert saved["dependency_versions"]["python"]
    assert saved["metadata"]["dependency_versions_hash"]
    assert saved["run_id_seed"] == 17
    assert saved["public_action_schema"]["product_domain_version"]
    assert saved["entries"][0]["transition"]["info"]["privileged_state"]

    saved["entries"][1]["transition"]["state_hash_before"] = "tampered"
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert TrajectoryRecorder.verify(path) is False


def test_trajectory_snapshot_is_detached_and_suffix_deletion_is_detected(tmp_path):
    env = MirageStructurerEnv(_task())
    _, reset_info = env.reset()
    recorder = TrajectoryRecorder(env, initial_state_hash=reset_info["state_hash"])
    first = env.step(Skip("one"))
    recorder.record(first, policy_metadata={"mutable": {"value": 1}})
    second = env.step(Skip("two"))
    recorder.record(second)

    # Mutating the caller-owned transition after record() cannot rewrite history.
    first.info["state_hash"] = "caller-mutated"
    assert recorder.entries[0].transition["info"]["state_hash"] != "caller-mutated"

    path = recorder.save(tmp_path / "trajectory.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert TrajectoryRecorder.verify(path) is False
