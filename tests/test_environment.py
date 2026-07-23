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
    Skip,
    SubmitDesign,
    SubmitProduct,
    TrajectoryRecorder,
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
        target_client="client-hidden",
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

    assert quoted.action.type == "request_quote"
    assert quoted.constraint_signals.hard_pass is True
    assert quoted.reward_components.quote_cost.value < 0
    assert quoted.reward_components.quote_cost.available is True
    assert quoted.reward_components.client_utility.available is False
    assert quoted.terminated is False

    submitted = env.step(SubmitDesign(quote_id, "risks disclosed"))

    assert submitted.action.type == "submit_design"
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
    _, base_info = base.reset(options={"curriculum": "a"})
    _, costly_info = costly.reset(options={"curriculum": "a"})
    assert base_info["state_hash"] != costly_info["state_hash"]

    _, changed_options = base.reset(options={"curriculum": "b"})
    assert base_info["state_hash"] != changed_options["state_hash"]
    assert (
        TrajectoryRecorder(base).metadata.environment_config_hash
        != TrajectoryRecorder(costly).metadata.environment_config_hash
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
    assert saved["metadata"]["environment_config_hash"]
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
