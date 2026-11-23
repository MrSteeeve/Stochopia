"""Validation and JSON round trips for the P0 contract."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from stochopia.harness.contract import (
    AnalysisSpec,
    ArmKind,
    CandidateRef,
    DataSplit,
    Direction,
    EvaluationTask,
    ExperimentManifest,
    FinalChoice,
    FitRecord,
    FrozenManifest,
    GroupSpec,
    PairedDelta,
    PairedEvalResult,
    PredictionVerdict,
    ReportStatus,
    ScenarioSpec,
    SearchTrialRecord,
    SelectionRecord,
    StructuredPrediction,
    UnitObservation,
)


@pytest.fixture
def manifest() -> ExperimentManifest:
    scenario = ScenarioSpec(
        scenario_id="question-a",
        diagnostic_question="Is the leading error in the background flow?",
        environment_kind="tier0_sim",
        environment={"name": "tier0-propagator", "flow": {"agents": 2}},
        action_plan={"side": 1, "n_children": 2, "spacing": 1, "n_relax": 2},
        window="execution",
        parent_id="base",
        candidate_ids=("base", "flow", "price"),
    )
    evidence_group = GroupSpec(
        group_id="evidence",
        kind="evidence_llm",
        model="research-model",
        model_version="2026-11",
        prompt_template_id="p0-v1",
        llm_budget={"max_calls": 2, "max_tokens": 2000},
        allowed_evidence=("day-level-flow",),
    )
    return ExperimentManifest(
        exp_id="round-trip",
        scenarios=(scenario,),
        candidates=(
            CandidateRef(candidate_id="base", backbone="gru", recipe="baseline"),
            CandidateRef(
                candidate_id="flow",
                backbone="gru",
                recipe="change_flow",
                implementation={"parameters": {"lags": [1, 2]}},
                parent_id="base",
                channels=("flow",),
            ),
            CandidateRef(
                candidate_id="price",
                backbone="gru",
                recipe="change_price",
                parent_id="base",
                channels=("price",),
            ),
        ),
        groups=(
            evidence_group,
            replace(evidence_group, group_id="metrics", kind="metrics_llm", allowed_evidence=()),
            GroupSpec(group_id="fixed", kind="no_llm", selection_rule="fixed-order"),
        ),
        splits={
            "train": DataSplit(
                source={"kind": "simulated", "world": "tier0-propagator", "namespace": "train"},
                unit_ids=("train-a", "train-b"),
                unit_seeds={"train-a": 101, "train-b": 102},
            ),
            "dev": DataSplit(
                source={"kind": "simulated", "world": "tier0-propagator", "namespace": "dev"},
                unit_ids=("dev-a", "dev-b"),
                unit_seeds={"dev-a": 201, "dev-b": 202},
            ),
        },
        train_seeds={"repeat-1": 11, "repeat-2": 12},
        train_plan={"learning_rate": 0.001, "steps": 2},
        measurement_version="tier0-v1",
        fit_version="fit-v1",
        seeds={"rollout": 31, "delta_ci": 32},
        budget={"search": {"trials": 2}, "final_check": {"trials": 2}},
        analysis=(
            AnalysisSpec(
                analysis_id="clamped-rmse",
                scenario_id="question-a",
                readout="clamped_rmse",
                window="execution",
                unit="tick",
                estimator="mean_path_rmse",
                acceptance={"max_delta": -0.4},
                min_effect=0.2,
                equivalence_band=(-0.2, 0.2),
                min_pairs=2,
                min_train_repeats=2,
                pairing="day",
                alpha=0.05,
                n_resamples=20,
                seed=32,
            ),
        ),
        prediction_candidate_set={"question-a": ("flow", "price")},
        unselected_sample_rule="check both candidates",
    )


@pytest.fixture
def prediction() -> StructuredPrediction:
    return StructuredPrediction(
        prediction_id="evidence-p1",
        exp_id="round-trip",
        round_index=1,
        group_id="evidence",
        scenario_id="question-a",
        explanation_id="flow-error",
        candidate_id="flow",
        parent_id="base",
        analysis_id="clamped-rmse",
        dev_snapshot_id="snapshot-1",
        readout="clamped_rmse",
        unit="tick",
        window="execution",
        direction=Direction.DECREASE,
        effect_interval=(-1.0, -0.5),
        conditions="Common development day and action schedule",
        falsifier="The paired interval is disjoint from the forecast",
    )


def test_manifest_json_round_trip_rebuilds_nested_records(manifest):
    manifest.validate()
    payload = json.loads(json.dumps(manifest.to_dict(), allow_nan=False))
    restored = ExperimentManifest.from_dict(payload)
    restored.validate()

    assert restored == manifest
    assert isinstance(restored.scenarios[0], ScenarioSpec)
    assert isinstance(restored.splits["train"], DataSplit)
    assert isinstance(restored.groups[0], GroupSpec)
    assert isinstance(restored.candidates[1], CandidateRef)
    assert isinstance(restored.analysis[0], AnalysisSpec)
    assert restored.scenarios[0].candidate_ids == ("base", "flow", "price")


def test_nested_data_is_frozen_and_json_export_is_detached(manifest):
    source = {"parameters": {"lags": [1, 2]}}
    candidate = replace(manifest.candidates[1], implementation=source)
    source["parameters"]["lags"].append(3)

    with pytest.raises(FrozenInstanceError):
        candidate.candidate_id = "other"
    with pytest.raises(TypeError):
        candidate.implementation["parameters"]["lags"] = (7,)
    with pytest.raises(TypeError):
        manifest.budget["search"]["trials"] = 200

    payload = candidate.to_dict()
    assert payload["implementation"]["parameters"]["lags"] == [1, 2]
    payload["implementation"]["parameters"]["lags"].append(4)
    assert candidate.to_dict()["implementation"]["parameters"]["lags"] == [1, 2]


def test_json_record_rejects_unknown_nested_fields(manifest):
    payload = manifest.to_dict()
    payload["candidates"][1]["unexpected_field"] = True
    with pytest.raises((TypeError, ValueError)):
        ExperimentManifest.from_dict(payload).validate()


def test_development_manifest_rejects_test_split(manifest):
    payload = manifest.to_dict()
    payload["splits"]["test"] = {
        "source": {"kind": "simulated", "world": "tier0-propagator", "namespace": "hidden"},
        "unit_ids": ["hidden-a"],
        "unit_seeds": {"hidden-a": 300},
    }
    with pytest.raises(ValueError):
        ExperimentManifest.from_dict(payload).validate()


@pytest.mark.parametrize("overlap", ("unit-id", "derived-seed"))
def test_development_manifest_rejects_overlap(manifest, overlap):
    payload = manifest.to_dict()
    dev = payload["splits"]["dev"]
    if overlap == "unit-id":
        dev["unit_ids"][0] = "train-a"
        dev["unit_seeds"]["train-a"] = dev["unit_seeds"].pop("dev-a")
    else:
        dev["unit_seeds"]["dev-a"] = 101
    with pytest.raises(ValueError):
        ExperimentManifest.from_dict(payload).validate()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model_version", "other-version"),
        ("prompt_template_id", "other-prompt"),
        ("llm_budget", {"max_calls": 3, "max_tokens": 2000}),
        ("allowed_evidence", ["day-level-flow"]),
    ],
)
def test_development_manifest_rejects_mismatched_llm_groups(manifest, key, value):
    payload = manifest.to_dict()
    payload["groups"][1][key] = value
    with pytest.raises(ValueError):
        ExperimentManifest.from_dict(payload).validate()


def test_development_manifest_requires_one_of_each_group(manifest):
    payload = manifest.to_dict()
    payload["groups"][2] = {**payload["groups"][1], "group_id": "second-metrics"}
    with pytest.raises(ValueError):
        ExperimentManifest.from_dict(payload).validate()


def test_scenario_menu_requires_its_registered_parent(manifest):
    payload = manifest.to_dict()
    payload["scenarios"][0]["candidate_ids"].remove("base")
    with pytest.raises(ValueError):
        ExperimentManifest.from_dict(payload).validate()


def test_scenario_menu_covers_flow_and_price_channels(manifest):
    payload = manifest.to_dict()
    payload["candidates"][2]["channels"] = ["flow"]
    with pytest.raises(ValueError):
        ExperimentManifest.from_dict(payload).validate()


def test_prediction_round_trip_restores_direction_and_signed_interval(prediction):
    prediction.validate()
    payload = json.loads(json.dumps(prediction.to_dict(), allow_nan=False))
    restored = StructuredPrediction.from_dict(payload)
    restored.validate()

    assert restored == prediction
    assert restored.direction is Direction.DECREASE
    assert restored.effect_interval == (-1.0, -0.5)


@pytest.mark.parametrize(
    ("direction", "interval"),
    [
        ("increase", [-1.0, -0.5]),
        ("decrease", [-0.5, 0.5]),
        ("decrease", [-0.5, -1.0]),
        ("decrease", [-float("inf"), -0.5]),
        ("unchanged", [-0.2, 0.2]),
    ],
)
def test_prediction_rejects_conflicting_or_invalid_interval(prediction, direction, interval):
    payload = prediction.to_dict()
    payload.update(direction=direction, effect_interval=interval)
    with pytest.raises(ValueError):
        StructuredPrediction.from_dict(payload).validate()


def test_unchanged_prediction_uses_the_registered_band(prediction):
    near_zero = replace(prediction, direction=Direction.UNCHANGED, effect_interval=None)
    near_zero.validate()


def test_abstention_requires_a_reason_but_not_an_effect_claim(prediction):
    abstention = replace(
        prediction,
        abstained=True,
        abstain_reason="The available observations do not identify the direction",
        direction=None,
        effect_interval=None,
        readout=None,
        unit=None,
        window=None,
        conditions=None,
        falsifier=None,
    )
    restored = StructuredPrediction.from_dict(abstention.to_dict())
    restored.validate()
    assert restored == abstention

    with pytest.raises(ValueError):
        replace(abstention, abstain_reason=None).validate()


@pytest.fixture
def frozen(manifest, prediction) -> FrozenManifest:
    scene = manifest.scenarios[0]
    search_orders = {
        "evidence": ("price",),
        "metrics": ("flow", "price"),
        "fixed": ("flow", "price"),
    }
    selections = tuple(
        SelectionRecord(
            selection_id=f"{group_id}-r{round_index}",
            exp_id=manifest.exp_id,
            round_index=round_index,
            sequence=round_index,
            group_id=group_id,
            scenario_id=scene.scenario_id,
            parent_id=scene.parent_id,
            dev_snapshot_id=f"snapshot-{round_index}",
            menu_id="tier0-menu",
            action="select",
            candidate_id=candidate_id,
            reason="Choose the next unobserved candidate",
        )
        for group_id, order in search_orders.items()
        for round_index, candidate_id in enumerate(order, start=1)
    )
    test_split = DataSplit(
        source={"kind": "simulated", "world": "tier0-propagator", "namespace": "test"},
        unit_ids=("test-a", "test-b"),
        unit_seeds={"test-a": 301, "test-b": 302},
    )
    return FrozenManifest(
        exp_id=manifest.exp_id,
        batch_id="batch-1",
        experiment=manifest,
        test_split=test_split,
        seeds={"rollout": 41, "delta_ci": 42},
        tasks=tuple(
            EvaluationTask(
                task_id=f"test-{candidate_id}",
                split="test",
                scenario_id=scene.scenario_id,
                parent_fit_id="fit-base",
                candidate_fit_id=f"fit-{candidate_id}",
                unit_ids=test_split.unit_ids,
                action_ref="execution-schedule",
                random_inputs_ref="shared-test-rollout",
            )
            for candidate_id in ("flow", "price")
        ),
        selections=selections,
        predictions=tuple(
            replace(
                prediction,
                prediction_id=f"{group_id}-{candidate_id}-forecast",
                group_id=group_id,
                candidate_id=candidate_id,
            )
            for group_id in ("evidence", "metrics")
            for candidate_id in manifest.prediction_candidate_set[scene.scenario_id]
        ),
        search_trials=tuple(
            SearchTrialRecord(
                trial_id=f"trial-{selection.selection_id}",
                exp_id=manifest.exp_id,
                group_id=selection.group_id,
                scenario_id=scene.scenario_id,
                sequence=selection.sequence,
                candidate_id=selection.candidate_id,
                purpose="search",
                charged_to_search=True,
                budget_units=1,
                cost={"trials": 1},
                status="completed",
                selection_id=selection.selection_id,
                development_status=(
                    ReportStatus.SUPPORTED
                    if selection.candidate_id == "price"
                    else ReportStatus.REFUTED
                ),
            )
            for selection in selections
        ),
        final_choices=tuple(
            FinalChoice(
                group_id=group_id,
                scenario_id=scene.scenario_id,
                candidate_id=order[-1],
                selection_id=f"{group_id}-r{len(order)}",
                stopping_reason="Selected candidate met the development requirement",
            )
            for group_id, order in search_orders.items()
        ),
        fits=tuple(
            FitRecord(
                fit_id=f"fit-{candidate.candidate_id}",
                candidate_id=candidate.candidate_id,
                train_repeat_id="repeat-1",
                train_data_ref="registered-train-days",
                config={"backbone": candidate.backbone, "recipe": candidate.recipe},
                seed=manifest.train_seeds["repeat-1"],
                model_ref=f"weights/{candidate.candidate_id}",
                parent_id=candidate.parent_id,
            )
            for candidate in manifest.candidates
        ),
    )


def test_frozen_manifest_json_round_trip_retains_choices_and_test_plan(frozen):
    frozen.validate()
    payload = json.loads(json.dumps(frozen.to_dict(), allow_nan=False))
    restored = FrozenManifest.from_dict(payload)

    assert restored == frozen
    assert restored.test_split.unit_ids == ("test-a", "test-b")
    assert {choice.candidate_id for choice in restored.final_choices} == {"price"}
    assert [selection.candidate_id for selection in restored.selections] == [
        "price", "flow", "price", "flow", "price",
    ]
    assert {task.task_id for task in restored.tasks} == {"test-flow", "test-price"}


def test_frozen_manifest_rejects_aliased_dev_day_seed(frozen):
    test_split = replace(frozen.test_split, unit_seeds={"test-a": 201, "test-b": 302})
    with pytest.raises(ValueError, match="overlap"):
        replace(frozen, test_split=test_split).validate()


def test_frozen_manifest_rejects_development_task(frozen):
    first, second = frozen.tasks
    with pytest.raises(ValueError, match="test"):
        replace(frozen, tasks=(replace(first, split="dev"), second)).validate()


def test_frozen_tasks_cover_first_choices_even_when_final_choices_match(frozen):
    study = replace(frozen.experiment, prediction_candidate_set={"question-a": ("price",)})
    predictions = tuple(row for row in frozen.predictions if row.candidate_id == "price")
    no_flow_forecast = replace(frozen, experiment=study, predictions=predictions)
    no_flow_forecast.validate()

    with pytest.raises(ValueError, match="cover"):
        replace(no_flow_forecast, tasks=(frozen.tasks[1],)).validate()


@pytest.mark.parametrize("update", ({"candidate_id": "flow"}, {"selection_id": "metrics-r2"}))
def test_frozen_final_choice_must_match_its_own_group_selection(frozen, update):
    first, *others = frozen.final_choices
    with pytest.raises(ValueError, match="choice|selection"):
        replace(frozen, final_choices=(replace(first, **update), *others)).validate()


@pytest.mark.parametrize("update", ({"round_index": 2}, {"dev_snapshot_id": "other-snapshot"}))
def test_frozen_groups_share_the_first_round_and_snapshot(frozen, update):
    first, *others = frozen.selections
    with pytest.raises(ValueError, match="round|snapshot"):
        replace(frozen, selections=(replace(first, **update), *others)).validate()


def test_frozen_initial_prediction_matches_the_shared_snapshot(frozen):
    first, *others = frozen.predictions
    wrong_snapshot = replace(first, dev_snapshot_id="other-snapshot")
    with pytest.raises(ValueError, match="snapshot"):
        replace(frozen, predictions=(wrong_snapshot, *others)).validate()


def test_common_predictions_require_a_shared_analysis(frozen):
    analysis, = frozen.experiment.analysis
    recovery = replace(analysis, analysis_id="recovery-rmse", window="recovery")
    predictions = tuple(
        replace(row, analysis_id=recovery.analysis_id, window=recovery.window)
        if row.group_id == "metrics" and row.candidate_id == "price"
        else row
        for row in frozen.predictions
    )
    experiment = replace(frozen.experiment, analysis=(analysis, recovery))
    with pytest.raises(ValueError, match="shared analysis"):
        replace(frozen, experiment=experiment, predictions=predictions).validate()


def test_frozen_tasks_require_matched_training_repeats(frozen):
    *others, price_fit = frozen.fits
    new_price_fit = replace(price_fit, train_repeat_id="repeat-2", seed=12)
    with pytest.raises(ValueError, match="training repeat"):
        replace(frozen, fits=(*others, new_price_fit)).validate()


def test_frozen_tasks_require_matched_training_data(frozen):
    *others, price_fit = frozen.fits
    new_price_fit = replace(price_fit, train_data_ref="different-train-days")
    with pytest.raises(ValueError, match="train"):
        replace(frozen, fits=(*others, new_price_fit)).validate()


@pytest.fixture
def paired_result(frozen) -> PairedEvalResult:
    task = frozen.tasks[1]
    return PairedEvalResult(
        result_id="test-price-result",
        task_id=task.task_id,
        split=task.split,
        scenario_id=task.scenario_id,
        candidate_id="price",
        parent_id="base",
        parent_fit_id=task.parent_fit_id,
        candidate_fit_id=task.candidate_fit_id,
        action_ref=task.action_ref,
        random_inputs_ref=task.random_inputs_ref,
        arms=(ArmKind.CLAMPED,),
        status="completed",
        cost={"train_seconds": 1.0, "eval_seconds": 0.5},
        observations=tuple(
            UnitObservation(
                unit_id=day,
                scenario_id=task.scenario_id,
                train_repeat_id="repeat-1",
                arm=ArmKind.CLAMPED,
                window="execution",
                readout="clamped_rmse",
                unit="tick",
                candidate_id=candidate_id,
                response_path=(response,),
                truth_path=(1.0,),
            )
            for day in task.unit_ids
            for candidate_id, response in (("base", 1.0), ("price", 2.0))
        ),
        paired_diffs={
            "clamped_rmse": PairedDelta(
                readout="clamped_rmse",
                estimator="mean_path_rmse",
                unit="tick",
                estimate=1.0,
                interval=(1.0, 1.0),
                n_pairs=2,
                method="percentile_bootstrap",
                seed=42,
                confidence_level=0.95,
                train_repeat_estimates={"repeat-1": 1.0},
            ),
        },
    )


def test_paired_result_json_round_trip_preserves_observation_arm_and_delta(paired_result):
    payload = json.loads(json.dumps(paired_result.to_dict(), allow_nan=False))
    restored = PairedEvalResult.from_dict(payload)

    assert restored == paired_result
    assert restored.observations[0].arm is ArmKind.CLAMPED
    assert restored.paired_diffs["clamped_rmse"].interval == (1.0, 1.0)


def test_paired_result_rejects_duplicate_day_observation_key(paired_result):
    with pytest.raises(ValueError, match="observation"):
        replace(
            paired_result,
            observations=(*paired_result.observations, paired_result.observations[0]),
        ).validate()


@pytest.fixture
def verdict(paired_result) -> PredictionVerdict:
    return PredictionVerdict(
        prediction_id="evidence-price-forecast",
        status=ReportStatus.REFUTED,
        reason="Direction matched but the expected magnitude was larger",
        result_id=paired_result.result_id,
        measured=paired_result.paired_diffs["clamped_rmse"],
        direction_status=ReportStatus.SUPPORTED,
        magnitude_status=ReportStatus.REFUTED,
        interval_width=0.0,
    )


def test_prediction_verdict_json_round_trip_keeps_separate_assessments(verdict):
    payload = json.loads(json.dumps(verdict.to_dict(), allow_nan=False))
    restored = PredictionVerdict.from_dict(payload)

    assert restored == verdict
    assert restored.status is ReportStatus.REFUTED
    assert restored.direction_status is ReportStatus.SUPPORTED
    assert restored.magnitude_status is ReportStatus.REFUTED


@pytest.mark.parametrize("update", ({"status": ReportStatus.SUPPORTED}, {"magnitude_status": None}))
def test_prediction_verdict_requires_consistent_assessments(verdict, update):
    with pytest.raises(ValueError):
        replace(verdict, **update).validate()
