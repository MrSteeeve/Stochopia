"""DSL quote binding, account flows, automatic termination and economic replay."""

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("structurerdsl")

from examples.dsl_roundtrip import fixture_payload
from examples.structuring_roundtrip import synthetic_task
from stochopia.adapters.structurer_dsl import DSLContract, DSLProductSpec
from stochopia.environment import (
    Skip, SubmitDesign, StochopiaStructurerEnv, TrajectoryRecorder,
    aggregate_evaluations, replay_and_evaluate, parse_environment_action,
)
from stochopia.environment.agent_runner import AgentActionError
from stochopia.environment.dsl import RequestDSLQuote, StochopiaDSLEnv, parse_dsl_action


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("case,settlement,reason", [
    ("vanilla", 100_000, "matured"), ("spread", 100_000, "matured"),
    ("asian", 83_333.3333333333, "matured"), ("range", 1_030_000, "matured"),
    ("autocall", 1_010_000, "autocalled"),
])
def test_quote_account_settlement_and_replay(tmp_path, case, settlement, reason):
    env = StochopiaDSLEnv(synthetic_task(), dsl_paths=512)
    env.reset()
    recorder = TrajectoryRecorder(env)
    quoted = env.step(parse_dsl_action({"action": "request_dsl_quote", "contract": fixture_payload(case)}))
    recorder.record(quoted)
    quote = quoted.tool_result["payload"]
    assert quote["client_price"] == quote["fair_value"]
    assert quote["dealer_margin"] == 0
    submitted = env.step(SubmitDesign(quote["quote_id"]))
    recorder.record(submitted)
    assert submitted.constraint_signals.accepted is True
    while not env.done:
        recorder.record(env.step(Skip()))
    portfolio = env._require_state().backend.portfolio
    assert len(portfolio.lifecycle_events) == 1
    assert portfolio.closed_positions[0].status == reason
    assert portfolio.lifecycle_events[0].settlement_amount == pytest.approx(settlement)
    assert portfolio.client_account.available_cash == pytest.approx(10_000_000 - quote["cash_outlay"] + settlement)
    assert portfolio.client_account.locked_cash == 0
    assert len([cf for cf in portfolio.cashflow_ledger if cf.event_type == "issuance"]) == 1
    assert len([cf for cf in portfolio.cashflow_ledger if cf.event_type == "settlement"]) == 1
    path = recorder.save(tmp_path / "trajectory.json")
    evaluated = replay_and_evaluate(path, replay=True)
    assert evaluated["replay_verified"] is True
    assert evaluated["format"] == "stochopia.dsl-experiment-evaluation.v1"
    assert recorder.metadata.action_schema_version == env.action_schema["schema_version"]
    assert "torch" in json.loads(path.read_text())["dependency_versions"]
    with pytest.raises(ValueError, match="v3 Level-0"):
        aggregate_evaluations([evaluated])


def test_quote_is_bound_before_caller_mutation():
    env = StochopiaDSLEnv(synthetic_task(), dsl_paths=128)
    env.reset()
    product = DSLProductSpec.from_contract(DSLContract.from_payload(fixture_payload("vanilla")))
    quoted = env.step(RequestDSLQuote(product))
    product.contract = DSLContract.from_payload(fixture_payload("autocall"))
    submitted = env.step(SubmitDesign(quoted.tool_result["payload"]["quote_id"]))
    assert submitted.constraint_signals.accepted is True
    position = env._require_state().backend.portfolio.positions[0]
    assert position.product.contract.program.body.op == "premium"
    assert position.initial_fixing == 6000
    assert position.observed_spots == (6000, 6100)


def test_horizon_liquidation_only_pays_remaining_value():
    task = replace(synthetic_task(), snapshots=synthetic_task().snapshots[:2])
    env = StochopiaDSLEnv(task, dsl_paths=256)
    env.reset()
    quoted = env.step(parse_dsl_action({"action": "request_dsl_quote", "contract": fixture_payload("asian")}))
    transition = env.step(SubmitDesign(quoted.tool_result["payload"]["quote_id"]))
    assert transition.terminated
    portfolio = env._require_state().backend.portfolio
    position = portfolio.closed_positions[0]
    assert position.status == "horizon_liquidated"
    assert position.settlement_amount == position.current_fair_value
    assert len(portfolio.lifecycle_events) == 1


def test_month_grid_and_client_routing():
    task = synthetic_task()
    # EpisodeTask enforces the monthly grid, so an ordered sequence with a gap fails.
    with pytest.raises(ValueError, match="monthly"):
        replace(task, snapshots=(task.snapshots[0], replace(task.snapshots[2], round_num=2)))
    env = StochopiaDSLEnv(task, dsl_paths=128)
    env.reset()
    payload = {**fixture_payload("vanilla"), "target_client": "other"}
    result = env.step(parse_dsl_action({"action": "request_dsl_quote", "contract": payload}))
    assert result.constraint_signals.action_valid is False
    assert "TARGET_CLIENT_MATCH" in result.constraint_signals.error


def test_level0_does_not_accept_experimental_action():
    env = StochopiaStructurerEnv(synthetic_task())
    env.reset()
    action = {"action": "request_dsl_quote", "contract": fixture_payload("vanilla")}
    with pytest.raises(AgentActionError):
        parse_environment_action(action)
    typed = parse_dsl_action(action)
    transition = env.step(typed)
    assert transition.constraint_signals.action_valid is False


@pytest.mark.parametrize("raw", [
    {"action": "request_dsl_quote", "contract": [1, 2]},
    {"action": "request_dsl_quote", "contract": {
        "version": "stochopia.dsl-contract.monthly.v1", "notional": 1e6,
        "program": {"maturity": 3, "underlyings": ["X"], "body": ["bogus_op"]},
    }},
    "[1, 2]",
])
def test_malformed_dsl_actions_are_policy_errors(raw):
    with pytest.raises(AgentActionError):
        parse_dsl_action(raw if isinstance(raw, str) else json.dumps(raw))


def test_non_positive_dsl_contract_is_quote_unavailable_not_invalid(monkeypatch):
    import stochopia.environment.dsl as dsl_module

    env = StochopiaDSLEnv(synthetic_task(), dsl_paths=128)
    env.reset()
    real = dsl_module.valuation
    monkeypatch.setattr(
        dsl_module, "valuation", lambda *a, **k: {**real(*a, **k), "fair_value": 0.0}
    )
    quotes_left = env._require_state().backend.max_quotes_per_round
    transition = env.step(parse_dsl_action({"action": "request_dsl_quote", "contract": fixture_payload("vanilla")}))

    assert transition.tool_result["type"] == "quote_unavailable"
    assert transition.constraint_signals.action_valid is True
    assert transition.constraint_signals.quote_unavailable is True
    assert transition.reward_components.operational_cost.available is False
    assert transition.observation.action_budget["quotes_left"] == quotes_left
    summary = transition.info["episode_summary"]["constraints"]
    assert summary["invalid_actions"] == 0
    assert summary["unavailable_quotes"] == 1
