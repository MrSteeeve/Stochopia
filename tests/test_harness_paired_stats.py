from dataclasses import replace
from math import sqrt

import pytest

from stochopia.harness.contract import (
    AnalysisSpec, ArmKind, Direction, EvaluationTask, FitRecord, PairedEvalResult, ReportStatus, ScenarioSpec,
    StructuredPrediction, UnitObservation,
)
from stochopia.harness.paired_stats import align_observations, estimate_delta, evaluate_pairs
from stochopia.harness.report import assess_candidate, judge_prediction, summarize_candidate


@pytest.fixture
def analysis() -> AnalysisSpec:
    return AnalysisSpec(
        analysis_id="curve", scenario_id="shock", readout="clamped_price", window="after",
        unit="tick", estimator="mean_path_rmse", acceptance={"max_delta": -0.4},
        min_effect=0.2, equivalence_band=(-0.2, 0.2), min_pairs=2, min_train_repeats=1,
        pairing="day_exhaustive", alpha=0.05, n_resamples=4, seed=42,
    )


@pytest.fixture
def paths(analysis: AnalysisSpec) -> tuple[UnitObservation, ...]:
    days = (
        ("A", (4.0, 2.0), (8.0, 2.0), (6.0, 3.0)),
        ("B", (6.0, 3.0), (8.0, 3.0), (7.0, 4.0)),
    )
    rows = []
    for day, truth, controls, revised in days:
        for candidate, response in (("base", controls), ("price", revised)):
            rows.append(UnitObservation(
                unit_id=day, scenario_id=analysis.scenario_id, train_repeat_id="fit-1", arm=ArmKind.CLAMPED,
                window=analysis.window, readout=analysis.readout, unit=analysis.unit, candidate_id=candidate,
                response_path=response, truth_path=truth,
            ))
    return tuple(rows)


def test_exhaustive_bootstrap_recomputes_the_mean_curve(
    analysis: AnalysisSpec, paths: tuple[UnitObservation, ...],
) -> None:
    delta = estimate_delta(analysis, reversed(paths), parent_id="base", candidate_id="price")
    assert delta.estimate == pytest.approx(sqrt(13 / 8) - 3 / sqrt(2))
    assert delta.interval == pytest.approx((-1.2172340824494587, -0.4466399550822213))
    assert delta.n_pairs == 2
    assert assess_candidate((analysis,), {analysis.analysis_id: delta}) is ReportStatus.SUPPORTED


def test_alignment_excludes_duplicate_misaligned_and_missing_rows(
    analysis: AnalysisSpec, paths: tuple[UnitObservation, ...],
) -> None:
    a_parent, a_price, b_parent, b_price = paths
    invalid_b = replace(b_price, response_path=(1.0,), truth_path=(1.0,))
    aligned = align_observations(
        analysis, (a_parent, a_price, b_parent, invalid_b, replace(a_price, unit_id="C"),
                   replace(a_price, unit_id="C")),
        parent_id="base", candidate_id="price", required_unit_ids=("A", "B", "C"),
    )
    assert aligned.days == ("A",)
    assert any("misaligned" in reason and "B" in reason for reason in aligned.excluded)
    assert any("duplicate" in reason and "C" in reason for reason in aligned.excluded)
    assert any("missing" in reason and "C" in reason for reason in aligned.excluded)


def test_below_repeat_threshold_keeps_estimate_and_marks_verdict_inconclusive(
    analysis: AnalysisSpec, paths: tuple[UnitObservation, ...],
) -> None:
    analysis = replace(analysis, min_train_repeats=2)
    delta = estimate_delta(analysis, paths, parent_id="base", candidate_id="price")
    assert delta.estimate == pytest.approx(sqrt(13 / 8) - 3 / sqrt(2))
    assert delta.interval is None
    assert "training repeats" in delta.reason
    prediction = StructuredPrediction(
        prediction_id="p", exp_id="e", round_index=1, group_id="evidence", scenario_id="shock",
        explanation_id="reason", candidate_id="price", parent_id="base", analysis_id="curve",
        dev_snapshot_id="before", readout=analysis.readout, unit=analysis.unit, window=analysis.window,
        direction=Direction.DECREASE, effect_interval=(-1.5, -0.4), conditions="shock", falsifier="misses",
    )
    verdict = judge_prediction(prediction, analysis, delta, result_id="frozen-price")
    assert verdict.status is verdict.direction_status is verdict.magnitude_status is ReportStatus.INCONCLUSIVE
    assert verdict.interval_width is None


def test_day_draw_is_shared_by_repeats(analysis: AnalysisSpec, paths: tuple[UnitObservation, ...]) -> None:
    analysis = replace(analysis, estimator="mean", min_train_repeats=2)
    scalar = []
    for repeat, direction in (("fit-1", 1), ("fit-2", -1)):
        for row in paths:
            value = {"A": 1.0, "B": 5.0}[row.unit_id] if row.candidate_id == "price" else 0.0
            scalar.append(replace(row, train_repeat_id=repeat, response_path=(), truth_path=(),
                                  values={"value": direction * value}))
    result = estimate_delta(analysis, scalar, parent_id="base", candidate_id="price")
    assert result.n_pairs == 2
    assert result.train_repeat_estimates == {"fit-1": 3.0, "fit-2": -3.0}
    assert result.interval == (0.0, 0.0)


@pytest.fixture
def task_setup() -> tuple[EvaluationTask, ScenarioSpec, FitRecord, FitRecord]:
    scenario = ScenarioSpec(
        scenario_id="shock", diagnostic_question="Why did price jump?", environment_kind="tier0_sim",
        environment={"world": "plain"}, action_plan={"size": 2}, window="after", parent_id="base",
        candidate_ids=("base", "price"),
    )
    parent = FitRecord(
        fit_id="fit-base", candidate_id="base", train_repeat_id="fit-1", train_data_ref="train",
        config={"backbone": "gru"}, seed=1, model_ref="fit-base.params",
    )
    candidate = replace(parent, fit_id="fit-price", candidate_id="price", parent_id="base", model_ref="fit-price.params")
    task = EvaluationTask(
        task_id="task", split="test", scenario_id="shock", parent_fit_id=parent.fit_id,
        candidate_fit_id=candidate.fit_id, unit_ids=("A", "B"), action_ref="schedule", random_inputs_ref="seed",
    )
    return task, scenario, parent, candidate


def test_task_report_distinguishes_unsupported_readout_from_incomplete_pair(
    analysis: AnalysisSpec, paths: tuple[UnitObservation, ...],
    task_setup: tuple[EvaluationTask, ScenarioSpec, FitRecord, FitRecord],
) -> None:
    task, scenario, parent, candidate = task_setup
    absent = replace(analysis, analysis_id="flow", readout="flow")
    incomplete = evaluate_pairs(task, scenario, parent, candidate, (paths[0],), (analysis, absent))
    unavailable = incomplete.paired_diffs[absent.analysis_id]
    assert (unavailable.n_pairs, unavailable.estimate, unavailable.interval) == (0, None, None)
    assert "no model observation" in unavailable.reason

    incomplete = PairedEvalResult.from_dict(incomplete.to_dict())
    delta = summarize_candidate((analysis, absent), (incomplete,), required_unit_ids=task.unit_ids)
    assert delta[absent.analysis_id] == unavailable
    assert delta[analysis.analysis_id].n_pairs == 0
    assert delta[analysis.analysis_id].interval is None
    assert assess_candidate((analysis,), delta) is ReportStatus.INCONCLUSIVE
    assert assess_candidate((absent,), delta) is ReportStatus.UNAVAILABLE

    forecast = StructuredPrediction(
        prediction_id="p", exp_id="e", round_index=1, group_id="evidence", scenario_id=scenario.scenario_id,
        explanation_id="reason", candidate_id=candidate.candidate_id, parent_id=parent.candidate_id,
        analysis_id=absent.analysis_id, dev_snapshot_id="before", readout=absent.readout, unit=absent.unit,
        window=absent.window, direction=Direction.UNCHANGED, conditions="shock", falsifier="misses",
    )
    verdict = judge_prediction(forecast, absent, unavailable, result_id=incomplete.result_id)
    assert verdict.status is ReportStatus.UNAVAILABLE
    assert verdict.reason == unavailable.reason


def test_task_estimates_only_clean_observations_and_audits_duplicate_arm(
    analysis: AnalysisSpec, paths: tuple[UnitObservation, ...],
    task_setup: tuple[EvaluationTask, ScenarioSpec, FitRecord, FitRecord],
) -> None:
    task, scenario, parent, candidate = task_setup
    extra_parent = replace(paths[0], arm=ArmKind.FREE)
    extra_candidate = replace(paths[1], arm=ArmKind.FREE)
    result = evaluate_pairs(
        task, scenario, parent, candidate, (*paths, extra_parent, extra_candidate, extra_candidate), (analysis,),
    )
    delta = result.paired_diffs[analysis.analysis_id]
    assert result.observations == paths
    assert delta.n_pairs == 2
    assert delta.estimate == pytest.approx(sqrt(13 / 8) - 3 / sqrt(2))
    assert "fit-1/A: duplicate observation keys" in delta.reason
    assert "free" in delta.reason


def test_duplicate_reference_is_excluded_from_all_model_arms(
    analysis: AnalysisSpec, paths: tuple[UnitObservation, ...],
    task_setup: tuple[EvaluationTask, ScenarioSpec, FitRecord, FitRecord],
) -> None:
    task, scenario, parent, candidate = task_setup
    duplicate = replace(paths[0], arm=ArmKind.TRUTH, candidate_id=None, response_path=())
    result = evaluate_pairs(task, scenario, parent, candidate, (*paths, duplicate, duplicate), (analysis,))
    delta = result.paired_diffs[analysis.analysis_id]
    assert {row.unit_id for row in result.observations} == {"B"}
    assert delta.n_pairs == 1
    assert delta.interval is None
    assert "fit-1/A: duplicate observation keys" in delta.reason
