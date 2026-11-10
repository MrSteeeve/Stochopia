"""End-to-end contracts for v3 task suites and economic replay evaluation."""

from __future__ import annotations

import json
from datetime import date

import pytest

from stochopia.benchmark import MarketSnapshot, RiskBudget
from stochopia.environment import (
    EpisodeTask,
    StochopiaStructurerEnv,
    RequestQuote,
    Skip,
    SubmitDesign,
    TaskSuite,
    TrajectoryRecorder,
    aggregate_evaluations,
    replay_and_evaluate,
)
from stochopia.products import ClientProfile, ProductSpec


def _task() -> EpisodeTask:
    snapshots = tuple(
        MarketSnapshot(
            episode_id="evaluation-test",
            round_num=round_num,
            as_of=date(2023, round_num, 28),
            underlying="CSI500",
            spot=6000.0 + round_num,
            risk_free_rate=0.02,
            realized_vol_20d=0.20,
            source="test",
        )
        for round_num in (1, 2, 3)
    )
    return EpisodeTask(
        snapshots=snapshots,
        client=ClientProfile(
            id="client-hidden",
            name="Hidden Client",
            capital=10_000_000.0,
            max_loss_pct=1.0,
            min_return_pct=0.02,
            risk_appetite="moderate",
        ),
        risk_budget=RiskBudget(
            notional=50_000_000.0,
            net_delta=50_000_000.0,
            gross_delta=100_000_000.0,
            net_vega=10_000_000.0,
            stress_loss=50_000_000.0,
        ),
        task_seed=29,
    )


def _record_complete_trajectory(path):
    environment = StochopiaStructurerEnv(_task())
    environment.reset(seed=23)
    recorder = TrajectoryRecorder(
        environment,
        run_metadata={
            "policy_kind": "test",
            "policy_name": "skip-policy",
        },
    )
    recorder.record(environment.step(Skip("first decision")))
    recorder.record(environment.step(Skip("second decision")))
    recorder.save(path)
    return path


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
        target_client="current_client",
        pitch="risk disclosed",
        hedging_plan="listed futures delta hedge",
    )


def test_task_suite_is_self_contained(tmp_path):
    suite = TaskSuite(
        name="unit",
        version="1",
        split="test",
        tasks=(_task(),),
        metadata={"purpose": "regression"},
    )
    path = suite.save(tmp_path / "suite.json")
    loaded = TaskSuite.load(path)

    assert loaded.suite_hash == suite.suite_hash
    assert loaded.tasks[0].task_hash == _task().task_hash
    assert loaded.to_dict()["tasks"][0]["client"]["id"] == "client-hidden"

    # Edited suites load as edited: the suite hash is an identity, not a seal.
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tasks"][0]["client"]["capital"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    edited = TaskSuite.load(path)
    assert edited.tasks[0].client.capital == 1.0
    assert edited.suite_hash != suite.suite_hash


def test_trajectory_evaluator_replays_economics_and_emits_metrics(tmp_path):
    trajectory = _record_complete_trajectory(tmp_path / "trajectory.json")

    evaluation = replay_and_evaluate(
        trajectory,
        output_path=tmp_path / "evaluation.json",
        suite_hash="suite-a",
        replay=True,
    )

    assert evaluation["replay_verified"] is True
    assert replay_and_evaluate(trajectory, suite_hash="suite-a")["metrics"] == evaluation["metrics"]
    assert evaluation["suite_hash"] == "suite-a"
    assert evaluation["policy_name"] == "skip-policy"
    assert evaluation["metrics"]["completion_rate"] == pytest.approx(1.0)
    assert evaluation["metrics"]["steps"] == pytest.approx(2.0)
    assert evaluation["metrics"]["submission_rate"] == pytest.approx(0.0)
    assert len(evaluation["policy_config_hash"]) == 64


def test_quote_checks_are_not_double_counted_as_submissions(tmp_path):
    environment = StochopiaStructurerEnv(_task())
    environment.reset(seed=23)
    recorder = TrajectoryRecorder(
        environment,
        run_metadata={
            "policy_kind": "test",
            "policy_name": "quote-policy",
        },
    )
    quoted = environment.step(RequestQuote(_product()))
    recorder.record(quoted)
    recorder.record(
        environment.step(
            SubmitDesign(quoted.tool_result["payload"]["quote_id"])
        )
    )
    recorder.record(environment.step(Skip("finish")))
    path = recorder.save(tmp_path / "quoted.json")

    evaluation = replay_and_evaluate(path, suite_hash="suite-a")

    assert evaluation["metrics"]["submission_rate"] == pytest.approx(1 / 3)
    assert evaluation["metrics"][
        "hard_execution_rate_given_submission"
    ] == pytest.approx(1.0)


def test_partial_run_without_steps_is_scored_from_recorded_outcome(tmp_path):
    environment = StochopiaStructurerEnv(_task())
    environment.reset(seed=71)
    recorder = TrajectoryRecorder(
        environment,
        run_metadata={
            "policy_kind": "test",
            "policy_name": "failed-policy",
        },
    )
    recorder.mark_run_outcome(
        "infrastructure_error",
        {"error_type": "TimeoutError"},
    )
    path = recorder.save(tmp_path / "partial.json")

    evaluation = replay_and_evaluate(path, suite_hash="suite-a")

    assert evaluation["replay_verified"] is False
    assert evaluation["status"] == "partial"
    assert evaluation["metrics"]["steps"] == pytest.approx(0.0)
    assert evaluation["metrics"]["completion_rate"] == pytest.approx(0.0)
    assert evaluation["metrics"]["infrastructure_failure_rate"] == pytest.approx(
        1.0
    )
    assert evaluation["metrics"]["partial_run_rate"] == pytest.approx(1.0)


def test_aggregate_requires_one_frozen_suite_and_clusters_by_task(tmp_path):
    trajectory = _record_complete_trajectory(tmp_path / "trajectory.json")
    first = replay_and_evaluate(trajectory, suite_hash="suite-a")
    second = replay_and_evaluate(trajectory, suite_hash="suite-a")

    aggregate = aggregate_evaluations(
        [first, second],
        n_resamples=50,
        seed=3,
    )

    summary = aggregate["policies"]["skip-policy"]
    assert aggregate["evaluation_count"] == 2
    assert summary["runs"] == 2
    assert summary["tasks"] == 1
    assert summary["metrics"]["completion_rate"]["n_tasks"] == 1

    other_suite = replay_and_evaluate(trajectory, suite_hash="suite-b")
    with pytest.raises(ValueError, match="mixed suites"):
        aggregate_evaluations(
            [first, other_suite],
            n_resamples=10,
            seed=3,
        )


def test_contract_acceptance_rate_only_counts_hard_pass_submissions():
    from stochopia.environment.evaluation import _metrics_from_transitions

    transitions = [
        {"constraint_signals": {"accepted": False, "hard_pass": False, "client_contract_pass": True}},
        {"constraint_signals": {"accepted": True, "hard_pass": True, "client_contract_pass": True}},
    ]
    metrics = _metrics_from_transitions(transitions)[0]
    assert metrics["contract_acceptance_rate_given_hard_pass"] == pytest.approx(1.0)


def test_conditional_rates_are_none_without_denominator(tmp_path):
    trajectory = _record_complete_trajectory(tmp_path / "trajectory.json")
    evaluation = replay_and_evaluate(trajectory, suite_hash="suite-a")
    metrics = evaluation["metrics"]
    for name in (
        "hard_execution_rate_given_submission",
        "contract_acceptance_rate_given_hard_pass",
        "settlement_acceptance_rate",
    ):
        assert metrics[name] is None

    aggregate = aggregate_evaluations([evaluation], n_resamples=10, seed=0)
    pooled = aggregate["policies"]["skip-policy"]["metrics"]
    assert "settlement_acceptance_rate" not in pooled
    assert "submission_rate" in pooled


def test_aggregate_rejects_mixed_environment_or_pricing_versions(tmp_path):
    trajectory = _record_complete_trajectory(tmp_path / "trajectory.json")
    first = replay_and_evaluate(trajectory, suite_hash="suite-a")
    for key in ("environment_version", "pricing_version"):
        stale = dict(first)
        stale[key] = "old-version"
        with pytest.raises(ValueError, match=f"mixed {key}"):
            aggregate_evaluations([first, stale], n_resamples=10, seed=0)


def test_evaluator_scores_edited_manifest_without_hash_self_check(tmp_path):
    trajectory = _record_complete_trajectory(tmp_path / "trajectory.json")
    payload = json.loads(trajectory.read_text(encoding="utf-8"))
    payload["task_manifest"]["client"]["capital"] = 9_000_000.0
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(payload), encoding="utf-8")

    evaluation = replay_and_evaluate(edited, suite_hash="suite-a")

    # task_hash is a functional ID of the content actually scored.
    assert evaluation["task_hash"] != payload["metadata"]["task_hash"]
    assert evaluation["metrics"]["steps"] == pytest.approx(2.0)
