"""Executable acceptance cases for the P0 interface contract, section 6."""

from __future__ import annotations

import csv
import importlib
import json
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

from stochopia.harness.contract import (
    AnalysisSpec,
    ArmKind,
    Direction,
    EvaluationTask,
    ExperimentManifest,
    FitRecord,
    FrozenManifest,
    PairedEvalResult,
    ReportStatus,
    ScenarioSpec,
    SelectionRecord,
    StructuredPrediction,
    UnitObservation,
)

# Reuse the valid contract baseline and its three-group search history.
from test_harness_contract import frozen, manifest, prediction  # noqa: F401
from test_harness_golden_example import _PATHS

if TYPE_CHECKING:
    from tasks.stochopia_tier0.entry import CheckRequest, Entry
    from tasks.stochopia_tier0.store import DevelopmentExport, DevelopmentStore


# --- Shared fixtures ---


@pytest.fixture
def openhyra(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModuleType]:
    default = Path(__file__).resolve().parents[2] / "OpenHyra"
    configured = os.environ.get("OPENHYRA_REPO")
    checkout = Path(configured or default).resolve()
    if not (checkout / "tasks" / "stochopia_tier0" / "entry.py").is_file():
        reason = f"P0 cross-repository acceptance requires an OpenHyra checkout at {checkout}"
        if configured:
            pytest.fail(reason)
        pytest.skip(reason)
    monkeypatch.syspath_prepend(str(checkout))
    return {
        name: importlib.import_module(f"tasks.stochopia_tier0.{name}")
        for name in ("entry", "store", "verifier")
    }


@pytest.fixture
def golden(frozen: FrozenManifest) -> FrozenManifest:
    rmse = replace(
        frozen.experiment.analysis[0],
        min_train_repeats=1,
        pairing="day_exhaustive",
        n_resamples=4,
    )
    endpoint = replace(
        rmse, analysis_id="end-response", readout="end_response", window="end",
        estimator="mean", acceptance={"role": "prediction"},
    )
    predictions = []
    for row in frozen.predictions:
        if row.candidate_id == "flow":
            direction, effect = Direction.UNCHANGED, None
        elif row.group_id == "evidence":
            direction, effect = Direction.INCREASE, (0.5, 1.5)
        else:
            direction, effect = Direction.DECREASE, (-1.5, -0.5)
        predictions.append(replace(
            row, analysis_id=endpoint.analysis_id, readout=endpoint.readout,
            window=endpoint.window, direction=direction, effect_interval=effect,
        ))
    golden = replace(
        frozen, experiment=replace(frozen.experiment, analysis=(rmse, endpoint)),
        predictions=tuple(predictions),
    )
    golden.validate()
    return golden


def _golden_observations(
    task: EvaluationTask, candidate_id: str, analyses: Sequence[AnalysisSpec],
) -> Iterator[UnitObservation]:
    for unit_id, paths in zip(task.unit_ids, _PATHS.values(), strict=True):
        for model_id in ("base", candidate_id):
            for analysis in analyses:
                raw = (
                    {"response_path": paths[model_id], "truth_path": paths["truth"]}
                    if analysis.estimator == "mean_path_rmse"
                    else {"values": {"value": paths[model_id][-1]}}
                )
                yield UnitObservation(
                    unit_id=unit_id, scenario_id=task.scenario_id, train_repeat_id="repeat-1",
                    arm=ArmKind.CLAMPED, window=analysis.window, readout=analysis.readout,
                    unit=analysis.unit, candidate_id=model_id, **raw,
                )


def _eval_golden(
    task: EvaluationTask, parent_fit: FitRecord, candidate_fit: FitRecord, frozen: FrozenManifest,
) -> PairedEvalResult:
    from stochopia.harness.paired_stats import evaluate_pairs

    return evaluate_pairs(
        task, frozen.experiment.scenarios[0], parent_fit, candidate_fit,
        tuple(_golden_observations(task, candidate_fit.candidate_id, frozen.experiment.analysis)),
        frozen.experiment.analysis, cost={"train_seconds": 0.25, "eval_seconds": 0.5},
    )


def _development_eval(frozen: FrozenManifest, selection: SelectionRecord) -> PairedEvalResult:
    fits = {fit.fit_id: fit for fit in frozen.fits}
    task = next(row for row in frozen.tasks if row.candidate_fit_id == f"fit-{selection.candidate_id}")
    task = replace(
        task, split="dev", task_id=f"dev-{selection.selection_id}",
        unit_ids=frozen.experiment.splits["dev"].unit_ids, random_inputs_ref="shared-dev-rollout",
    )
    return _eval_golden(task, fits[task.parent_fit_id], fits[task.candidate_fit_id], frozen)


def _unit(
    analysis: AnalysisSpec, day: str, repeat: str, candidate: str, value: float, **overrides: Any,
) -> UnitObservation:
    return replace(UnitObservation(
        unit_id=day, scenario_id=analysis.scenario_id, train_repeat_id=repeat,
        arm=ArmKind.CLAMPED, window=analysis.window, readout=analysis.readout,
        unit=analysis.unit, candidate_id=candidate, values={"value": value},
    ), **overrides)


# --- Diagnosis and statistical contract ---


def test_f1_diagnose_source_and_retain_weak_signal() -> None:
    from stochopia.harness.runner import classify_diagnosis

    truth = (2.0, 2.0)
    options = {"min_effect": 0.2, "dominance_margin": 0.1}
    flow = classify_diagnosis((3.0, 3.0), (2.0, 2.0), truth, **options)
    price = classify_diagnosis((3.0, 3.0), (3.0, 3.0), truth, **options)
    weak = classify_diagnosis((2.02, 2.02), (2.01, 2.01), truth, **options)

    assert (flow["status"], flow["dominant_channel"]) == (ReportStatus.SUPPORTED, "flow")
    assert (flow["flow_rmse"], flow["price_rmse"]) == pytest.approx((1.0, 0.0))
    assert (price["status"], price["dominant_channel"]) == (ReportStatus.SUPPORTED, "price")
    assert (price["flow_rmse"], price["price_rmse"]) == pytest.approx((0.0, 1.0))
    assert (weak["status"], weak["dominant_channel"]) == (ReportStatus.INCONCLUSIVE, None)
    assert (weak["flow_rmse"], weak["price_rmse"]) == pytest.approx((0.01, 0.01))
    assert weak["reason"]


def test_f2_enforce_registered_groups_and_splits(
    openhyra: dict[str, ModuleType], tmp_path: Path, golden: FrozenManifest,
) -> None:
    entry = openhyra["entry"].Entry
    store = openhyra["store"].DevelopmentStore(tmp_path / "runs", golden.exp_id)
    experiment = golden.experiment
    groups = list(experiment.groups)
    groups[1] = replace(groups[1], llm_budget={"max_calls": 9})
    scene = experiment.scenarios[0]
    train = experiment.splits["train"]
    dev = experiment.splits["dev"]
    dev_seeds = dict(dev.unit_seeds)
    dev_seeds[dev.unit_ids[0]] = next(iter(train.unit_seeds.values()))
    invalid_manifests = (
        (replace(experiment, groups=tuple(groups)), "budget"),
        (replace(experiment, groups=experiment.groups[:-1]), "group"),
        (replace(experiment, scenarios=(replace(scene, candidate_ids=("base", "price")),)), "menu"),
        (replace(experiment, splits={**experiment.splits, "dev": replace(dev, unit_seeds=dev_seeds)}),
         "overlap"),
    )
    for draft, reason in invalid_manifests:
        with pytest.raises(ValueError, match=reason):
            entry(store, draft)
    assert not (store.path / "manifest.json").exists()

    verifier = openhyra["verifier"].Verifier(tmp_path / "verification", evaluate=_eval_golden)
    test_seeds = dict(golden.test_split.unit_seeds)
    test_seeds[golden.test_split.unit_ids[0]] = next(iter(dev.unit_seeds.values()))
    aliased = replace(
        golden.test_split, unit_seeds=test_seeds,
        source={**golden.test_split.source, "path": "different-location"},
    )
    with pytest.raises(ValueError, match="overlap"):
        verifier.register(replace(golden, test_split=aliased))


def test_f4_align_and_audit_day_pairs(golden: FrozenManifest) -> None:
    from stochopia.harness.paired_stats import align_observations, estimate_delta

    analysis = golden.experiment.analysis[1]
    rows = (
        _unit(analysis, "B", "r1", "price", 4),
        _unit(analysis, "A", "r1", "base", 1),
        _unit(analysis, "D", "r1", "price", 9),
        _unit(analysis, "A", "r1", "price", 2),
        _unit(analysis, "C", "r1", "base", 100),
        _unit(analysis, "D", "r1", "price", 9),
        _unit(analysis, "B", "r1", "base", 1),
        _unit(analysis, "D", "r1", "base", 1),
        _unit(analysis, "E", "r1", "base", 1),
        _unit(analysis, "E", "r1", "price", 9, window="misaligned"),
    )
    pairing = dict(parent_id="base", candidate_id="price", required_unit_ids=("A", "B", "C", "D", "E"))
    aligned = align_observations(analysis, rows, **pairing)
    assert aligned.days == ("A", "B")
    exclusions = " ".join(map(str, aligned.excluded)).lower()
    assert "duplicate" in exclusions and "missing" in exclusions
    assert all(day in str(aligned.excluded) for day in ("C", "D", "E"))

    delta = estimate_delta(analysis, rows, **pairing)
    assert (delta.n_pairs, delta.estimate) == (2, 2.0)
    assert delta.interval == pytest.approx((1.075, 2.925), abs=1e-12)
    assert delta.reason


def test_f4_report_day_and_training_uncertainty_separately(golden: FrozenManifest) -> None:
    from stochopia.harness.paired_stats import estimate_delta
    from stochopia.harness.report import judge_prediction

    analysis = replace(golden.experiment.analysis[1], min_train_repeats=2, pairing="day", n_resamples=80)
    forecast = next(row for row in golden.predictions if row.group_id == "evidence" and row.candidate_id == "price")
    rows = tuple(
        _unit(analysis, day, repeat, candidate, value)
        for repeat, deltas in (("r1", (1, 3)), ("r2", (4, 2)))
        for day, delta in zip(("A", "B"), deltas, strict=True)
        for candidate, value in (("base", 0), ("price", delta))
    )
    pair = dict(parent_id="base", candidate_id="price")
    for subset, expected_estimate, expected_days in ((rows[:2], 1, 1), (rows[:4], 2, 2)):
        short = estimate_delta(analysis, subset, **pair)
        assert (short.n_pairs, short.estimate, short.interval) == (expected_days, expected_estimate, None)
        assert short.reason
        verdict = judge_prediction(forecast, analysis, short, result_id="underpowered")
        assert verdict.status is ReportStatus.INCONCLUSIVE

    delta = estimate_delta(analysis, rows, **pair)
    assert (delta.n_pairs, delta.estimate) == (2, 2.5)
    assert delta.train_repeat_estimates == {"r1": 2.0, "r2": 3.0}
    # Both repeats move in opposite directions across A and B. Shared resamples cancel.
    assert delta.interval == (2.5, 2.5)
    assert estimate_delta(analysis, tuple(reversed(rows)), **pair) == delta


Selector = Callable[[dict[str, Any], ScenarioSpec], tuple[SelectionRecord, tuple[StructuredPrediction, ...]]]


# --- Development registration and frozen replay ---


def _selectors(frozen: FrozenManifest, round_index: int) -> dict[str, Selector]:
    selectors = {}
    for row in frozen.selections:
        if row.round_index != round_index:
            continue

        def select(
            snapshot: dict[str, Any], scene: ScenarioSpec, row: SelectionRecord = row,
        ) -> tuple[SelectionRecord, tuple[StructuredPrediction, ...]]:
            assert scene.scenario_id == row.scenario_id
            selected = replace(row, dev_snapshot_id=snapshot["dev_snapshot_id"])
            forecasts = tuple(
                replace(forecast, dev_snapshot_id=selected.dev_snapshot_id)
                for forecast in frozen.predictions
                if forecast.group_id == row.group_id and forecast.round_index == round_index
            )
            return selected, forecasts

        selectors[row.group_id] = select
    return selectors


def _entry(
    openhyra: dict[str, ModuleType], tmp_path: Path, experiment: ExperimentManifest,
) -> tuple[Entry, DevelopmentStore]:
    store = openhyra["store"].DevelopmentStore(tmp_path / "runs", experiment.exp_id)
    return openhyra["entry"].Entry(store, experiment), store


def test_f3_preregister_and_blind_group_feedback(
    openhyra: dict[str, ModuleType], tmp_path: Path, golden: FrozenManifest,
) -> None:
    entry, store = _entry(openhyra, tmp_path, golden.experiment)
    assert store.path == tmp_path / "runs" / "stochopia_tier0" / golden.exp_id
    round1 = store.path / "rounds" / "1"
    executed = []

    def evaluate(selection: SelectionRecord | CheckRequest) -> PairedEvalResult:
        assert (round1 / "preregistered.json").is_file()
        sealed = (round1 / "preregistered.json").read_text()
        assert all(row.selection_id in sealed for row in golden.selections if row.round_index == 1)
        assert all(row.prediction_id in sealed for row in golden.predictions)
        assert not (round1 / "feedback.json").exists()
        executed.append(selection.group_id)
        if not isinstance(selection, SelectionRecord):
            assert selection.candidate_id == "flow"
            selection = replace(golden.selections[0], selection_id="hidden-check", candidate_id="flow")
        return _development_eval(golden, selection)

    initial_evidence = {"day-level-flow": {"fault": "asymmetry"}, "unregistered": 11}
    hidden = openhyra["entry"].CheckRequest(
        group_id="evidence", scenario_id=golden.experiment.scenarios[0].scenario_id,
        candidate_id="flow", purpose="common_prediction", visible_to_group=False,
    )
    entry.run_round(
        1, selectors=_selectors(golden, 1), evaluate=evaluate,
        baseline_metrics={"start": 0.3}, evidence=initial_evidence, checks=(hidden,),
    )
    assert set(executed) == {"evidence", "metrics", "fixed"}
    assert len(executed) == 4
    assert (round1 / "feedback.json").is_file()

    snapshots = {}
    for group in ("evidence", "metrics", "fixed"):
        snapshots[group] = entry.snapshot(
            group, 2, baseline_metrics={"start": 0.3}, evidence=initial_evidence,
        )
        feedback = snapshots[group]["feedback"]
        assert len(feedback) == 1 and feedback[0]["trial"]["group_id"] == group
    assert snapshots["evidence"]["evidence"] == {"day-level-flow": {"fault": "asymmetry"}}
    assert snapshots["metrics"]["evidence"] == {}
    assert snapshots["fixed"]["evidence"] == {}
    assert len({row["dev_snapshot_id"] for row in snapshots.values()}) == 1
    evidence_trial = snapshots["evidence"]["feedback"][0]["trial"]
    assert evidence_trial["candidate_id"] == "price" and evidence_trial["budget_units"] == 1
    assert len(list((round1 / "measurements").glob("*.json"))) == 4

    # Metrics saw flow in round 1 and cannot submit a new ex ante forecast for it.
    choice, _ = _selectors(golden, 2)["metrics"](snapshots["metrics"], golden.experiment.scenarios[0])
    old = next(row for row in golden.predictions if row.group_id == "metrics" and row.candidate_id == "flow")
    late = replace(old, round_index=2, dev_snapshot_id=choice.dev_snapshot_id, prediction_id="late-flow")
    with pytest.raises(ValueError, match="late|observed|exposed|see|prior"):
        entry.preregister_round(
            2, selections=(choice,), predictions=(late,), baseline_metrics={"start": 0.3},
        )
    assert not (store.path / "rounds" / "2" / "preregistered.json").exists()


def test_f5_isolate_test_data_and_seal_frozen_inputs(
    openhyra: dict[str, ModuleType], tmp_path: Path, golden: FrozenManifest,
) -> None:
    verifier_module = openhyra["verifier"]
    verifier_root = tmp_path / "verification"
    config = verifier_module.VerifierConfig(
        exp_id=golden.exp_id, batch_id=golden.batch_id, test_split=golden.test_split, seeds=golden.seeds,
    )
    verifier_module.Verifier(verifier_root).register_config(config)
    entry, store = _entry(openhyra, tmp_path, golden.experiment)
    for index in (1, 2):
        entry.run_round(
            index, selectors=_selectors(golden, index), baseline_metrics={"start": 0.3},
            evaluate=lambda selection: _development_eval(golden, selection),
        )

    exported = entry.export(golden.final_choices, fits=golden.fits)
    with pytest.raises(ValueError, match="separate"):
        verifier_module.freeze(exported, config, store.path / "verification", tasks=golden.tasks)
    verifier_module.freeze(exported, config, verifier_root, tasks=golden.tasks)
    verifier = verifier_module.Verifier(verifier_root, evaluate=_eval_golden)
    sealed = verifier.frozen()
    assert len(sealed.selections) == 5 and len(sealed.predictions) == 4
    assert sealed.tasks == golden.tasks and sealed.test_split == golden.test_split
    for path in store.path.rglob("*.json"):
        payload = path.read_text()
        assert all(unit not in payload for unit in config.test_split.unit_ids)
        assert str(verifier_root) not in payload
    before = entry.snapshot("metrics", 3, baseline_metrics={})

    dev_log = store.path / "rounds" / "1" / "preregistered.json"
    dev_log.write_text(json.dumps({"modified_after_freeze": True}), encoding="utf-8")
    report = verifier.run()
    assert verifier.frozen() == sealed
    assert report["batch_id"] == golden.batch_id

    after = entry.snapshot("metrics", 3, baseline_metrics={})
    assert before == after
    assert golden.tasks[1].task_id not in json.dumps(after)


def test_f5_exclude_concurrent_runs_and_resume_uncommitted_tasks(
    openhyra: dict[str, ModuleType], tmp_path: Path, golden: FrozenManifest,
) -> None:
    module = openhyra["verifier"]
    entered = Event()
    release = Event()
    invocations = []

    def interrupted(
        task: EvaluationTask, parent: FitRecord, candidate: FitRecord, frozen: FrozenManifest,
    ) -> PairedEvalResult:
        invocations.append(task.task_id)
        if task == golden.tasks[1]:
            entered.set()
            if not release.wait(timeout=10):
                raise TimeoutError("test did not release the first verifier")
            raise RuntimeError("interrupted before committing second task")
        return _eval_golden(task, parent, candidate, frozen)

    verifier = module.Verifier(tmp_path / "verification", evaluate=interrupted)
    verifier.register(golden)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(verifier.run)
        try:
            assert entered.wait(timeout=10)
            with pytest.raises(module.VerifierBusyError):
                module.Verifier(tmp_path / "verification", evaluate=interrupted).run()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="interrupted"):
            future.result(timeout=10)

    assert verifier.result(golden.tasks[0].task_id).task_id == golden.tasks[0].task_id
    assert verifier.result(golden.tasks[1].task_id) is None

    def recover(
        task: EvaluationTask, parent: FitRecord, candidate: FitRecord, frozen: FrozenManifest,
    ) -> PairedEvalResult:
        invocations.append(task.task_id)
        return _eval_golden(task, parent, candidate, frozen)

    recovered = module.Verifier(tmp_path / "verification", evaluate=recover)
    first = recovered.run()
    assert first == recovered.run()
    assert invocations == ["test-flow", "test-price", "test-price"]
    reports = list((tmp_path / "verification").rglob("report.json"))
    assert len(reports) == 1 and json.loads(reports[0].read_text()) == first


def test_f5_recompute_golden_report_from_frozen_records(
    openhyra: dict[str, ModuleType], tmp_path: Path, golden: FrozenManifest,
) -> None:
    from stochopia.harness.report import build_report

    verifier = openhyra["verifier"].Verifier(tmp_path / "verification", evaluate=_eval_golden)
    verifier.register(golden)
    report = verifier.run()
    measurements = [verifier.result(task.task_id) for task in golden.tasks]
    assert report == build_report(verifier.frozen(), tuple(reversed(measurements)))

    price_delta = measurements[1].paired_diffs["clamped-rmse"]
    assert price_delta.estimate == pytest.approx(-0.8465654651614463, abs=1e-12)
    assert price_delta.interval == pytest.approx((-1.2172340824494587, -0.4466399550822213), abs=1e-12)
    assert measurements[0].paired_diffs["clamped-rmse"].interval == (0, 0)
    assert measurements[1].paired_diffs["end-response"].interval == (1, 1)

    choices = {row["group_id"]: row for row in report["choices"]}
    assert set(choices) == {"evidence", "metrics", "fixed"}
    assert all(row["final_candidate"] == "price" for row in choices.values())
    assert all(row["final_model_gain"] == pytest.approx(0.8465654651614463) for row in choices.values())
    assert [(choices[group]["first_frozen_qualified"], choices[group]["first_development_pass_cost"])
            for group in ("evidence", "metrics", "fixed")] == [(True, 1), (False, 2), (False, 2)]

    forecasts = {(row["group_id"], row["candidate_id"]): row for row in report["predictions"]}
    assert set(forecasts) == {("evidence", "flow"), ("evidence", "price"), ("metrics", "flow"), ("metrics", "price")}
    for group in ("evidence", "metrics"):
        assert forecasts[group, "flow"]["equivalence_status"] == ReportStatus.SUPPORTED.value
    assert forecasts["evidence", "price"]["status"] == ReportStatus.SUPPORTED.value
    refuted = forecasts["metrics", "price"]
    assert refuted["direction_status"] == refuted["magnitude_status"] == refuted["status"] == ReportStatus.REFUTED.value
    assert refuted["interval_width"] == 0 and refuted["model_score_delta"] > 0
    assert refuted["result_id"] == measurements[1].result_id
    assert len(report["refuted_improvements"]) == 1
    trace = report["refuted_improvements"][0]
    assert trace["prediction_id"] == refuted["prediction_id"]
    assert trace["final_selection_id"] == choices["metrics"]["final_selection_id"]
    assert trace["result_ids"] == [measurements[1].result_id]
    assert "fixed" not in report["x_to_delta_y"]


# --- Real runner and matching continuity ---


def _tiny_experiment(golden: FrozenManifest, backbone: str) -> ExperimentManifest:
    from stochopia.stats import derive_seed

    experiment = golden.experiment
    scenario = replace(
        experiment.scenarios[0],
        environment={
            "world": "plain",
            "world_overrides": {"n_trades": 96, "burn_in": 24},
        },
        action_plan={"children": 2, "spacing": 1, "relax": 4, "context": 8, "windows_per_day": 2},
    )
    splits = {}
    for name, data in experiment.splits.items():
        namespace = f"harness-continuity-{name}"
        seeds = {
            unit: derive_seed("stochopia.sim.world", "tier0-plain", namespace, index)
            for index, unit in enumerate(data.unit_ids)
        }
        splits[name] = replace(
            data, source={"kind": "simulated", "world": "tier0-plain", "namespace": namespace},
            unit_seeds=seeds,
        )
    recipes = {"base": "seed", "flow": "separate_flow", "price": "separate_flow_own_price"}
    candidates = []
    for candidate in experiment.candidates:
        channels = ("flow", "price") if candidate.candidate_id == "price" else candidate.channels
        candidates.append(replace(
            candidate, backbone=backbone, recipe=recipes[candidate.candidate_id],
            implementation={}, channels=channels,
        ))
    return replace(
        experiment, exp_id=f"continuity-{backbone}", scenarios=(scenario,), splits=splits,
        candidates=tuple(candidates),
        train_seeds={"repeat-1": 11},
        train_plan={"hidden": 8, "layers": 1, "steps": 2, "batch": 2, "window": 8, "device": "cpu"},
    )


def _verify_trained_gru(
    verifier_module: ModuleType, exported: DevelopmentExport, golden: FrozenManifest, root: Path,
) -> None:
    from stochopia.stats import derive_seed

    namespace = "harness-continuity-test"
    units = golden.test_split.unit_ids
    test_split = replace(
        golden.test_split,
        source={"kind": "simulated", "world": "tier0-plain", "namespace": namespace},
        unit_seeds={
            unit: derive_seed("stochopia.sim.world", "tier0-plain", namespace, index)
            for index, unit in enumerate(units)
        },
    )
    config = verifier_module.VerifierConfig(
        exp_id=exported.experiment.exp_id, batch_id=golden.batch_id, test_split=test_split, seeds=golden.seeds,
    )
    fits = {fit.candidate_id: fit for fit in exported.fits}
    assert set(fits) == {"base", "flow", "price"}
    tasks = []
    for task, candidate in zip(golden.tasks, ("flow", "price"), strict=True):
        tasks.append(replace(
            task, parent_fit_id=fits["base"].fit_id, candidate_fit_id=fits[candidate].fit_id,
        ))
    timestamps = {Path(fit.model_ref): Path(fit.model_ref).stat().st_mtime_ns for fit in fits.values()}
    verifier_module.freeze(exported, config, root, tasks=tasks)
    verifier = verifier_module.Verifier(root)
    report = verifier.run()
    assert all(row["status"] == "completed" for row in report["measurement"])
    for task in tasks:
        measured = verifier.result(task.task_id)
        assert {observation.unit_id for observation in measured.observations} == set(units)
    assert {path: path.stat().st_mtime_ns for path in timestamps} == timestamps


def test_continuity_switch_backbone_and_resume_development(
    openhyra: dict[str, ModuleType], tmp_path: Path, golden: FrozenManifest,
) -> None:
    from stochopia.harness.runner import Runner

    for backbone in ("gru", "ssm", "lru"):
        experiment = _tiny_experiment(golden, backbone)
        entry, store = _entry(openhyra, tmp_path, experiment)
        choices = replace(
            golden,
            selections=tuple(replace(row, exp_id=experiment.exp_id) for row in golden.selections),
            predictions=tuple(replace(row, exp_id=experiment.exp_id) for row in golden.predictions),
        )
        first_selectors = _selectors(choices, 1)
        del first_selectors["fixed"]
        entry.run_round(1, selectors=first_selectors, baseline_metrics={})

        reopened = openhyra["entry"].Entry(
            openhyra["store"].DevelopmentStore(tmp_path / "runs", experiment.exp_id), experiment,
        )
        snapshots = {
            group: reopened.snapshot(group, 2, baseline_metrics={})["feedback"]
            for group in ("evidence", "metrics", "fixed")
        }
        assert snapshots["metrics"][0]["trial"]["cost"]["train_seconds"] == (
            snapshots["fixed"][0]["trial"]["cost"]["train_seconds"]
        )
        for group, history in snapshots.items():
            assert len(history) == 1 and history[0]["trial"]["group_id"] == group
            item = history[0]
            trial_id = item["trial"]["trial_id"]
            recorded = store.path / "rounds" / "1" / "measurements" / f"{trial_id}.json"
            measurements = json.loads(recorded.read_text())["results"]
            assert len(measurements) == 1
            result = measurements[0]
            assert item["results"] == [{
                "result_id": result["result_id"], "status": "completed", "failure_reason": None,
            }]
            assert result["status"] == "completed"
            assert set(result["paired_diffs"]) == {"clamped-rmse", "end-response"}
            assert {row["unit_id"] for row in result["observations"]} == set(experiment.splits["dev"].unit_ids)

        repeat = "repeat-1"
        runner = Runner(experiment)
        parent = runner.fit_candidate("base", repeat, tmp_path / "diagnose" / backbone)
        diagnosis = runner.diagnose(experiment.scenarios[0].scenario_id, parent)
        assert diagnosis["status"] in (ReportStatus.SUPPORTED, ReportStatus.INCONCLUSIVE)
        assert diagnosis["flow_rmse"] >= 0 and diagnosis["price_rmse"] >= 0

        if backbone == "gru":
            next_selectors = _selectors(choices, 2)
            del next_selectors["fixed"]
            reopened.run_round(2, selectors=next_selectors, baseline_metrics={})
            later = reopened.snapshot("metrics", 3, baseline_metrics={})["feedback"]
            assert [row["trial"]["candidate_id"] for row in later] == ["flow", "price"]
            fixed = reopened.snapshot("fixed", 3, baseline_metrics={})["feedback"]
            assert [row["trial"]["candidate_id"] for row in fixed] == ["flow", "price"]
            final_choices = tuple(
                replace(choice, selection_id=fixed[-1]["trial"]["selection_id"])
                if choice.group_id == "fixed" else choice
                for choice in choices.final_choices
            )
            _verify_trained_gru(
                openhyra["verifier"], reopened.export(final_choices), golden, tmp_path / "verification",
            )
        assert store.path.is_dir()


def _short_lob_frozen(golden: FrozenManifest) -> FrozenManifest:
    experiment = golden.experiment
    scene = replace(
        experiment.scenarios[0], environment_kind="lob", environment={"depth": 2},
        action_plan={"short_fixture": True}, matching_version="jaxlob==0.19",
    )
    depth = replace(
        experiment.analysis[0], readout="best_ask_depth", unit="shares", estimator="mean",
        acceptance={"max_delta": -40},
    )
    size = replace(experiment.analysis[1], readout="filled_size", unit="shares")
    forecasts = tuple(replace(row, readout=size.readout, unit=size.unit) for row in golden.predictions)
    splits = {}
    for name, source in experiment.splits.items():
        unit_id = source.unit_ids[0]
        splits[name] = replace(source, unit_ids=(unit_id,), unit_seeds={unit_id: source.unit_seeds[unit_id]})
    test_id = golden.test_split.unit_ids[0]
    test_split = replace(
        golden.test_split, unit_ids=(test_id,), unit_seeds={test_id: golden.test_split.unit_seeds[test_id]},
    )
    experiment = replace(
        experiment, scenarios=(scene,), analysis=(depth, size), splits=splits, train_seeds={"repeat-1": 11},
    )
    tasks = tuple(replace(task, unit_ids=(test_id,)) for task in golden.tasks)
    return replace(golden, experiment=experiment, test_split=test_split, tasks=tasks, predictions=forecasts)


def test_continuity_runs_actual_matching_and_exports_post_event_books(
    openhyra: dict[str, ModuleType], tmp_path: Path, golden: FrozenManifest,
) -> None:
    pytest.importorskip("jaxlob", reason="actual matching requires the harness-lob extra on Python 3.10–3.12")
    from stochopia.harness.lob_adapter import export_lob_bench, run_short_lob

    lob = _short_lob_frozen(golden)
    entry, store = _entry(openhyra, tmp_path, lob.experiment)
    feedback = entry.run_round(1, selectors=_selectors(lob, 1), baseline_metrics={})
    for items in feedback.values():
        assert len(items) == 1
        trial = items[0]
        assert trial["results"][0]["status"] == "completed"
        assert trial["paired_diffs"]["clamped-rmse"]["estimate"] == -40
        fill = trial["paired_diffs"]["end-response"]
        assert fill["estimate"] == 40 and fill["interval"] is None and fill["reason"]
    assert store.path.is_dir()

    result = run_short_lob(depth=2)
    assert result.status is ReportStatus.SUPPORTED, result.reason
    assert result.matching_version == "jaxlob==0.19"
    assert result.with_order is not None and result.without_order is not None

    messages, books = tmp_path / "message.csv", tmp_path / "orderbook.csv"
    export_lob_bench(result.with_order, messages, books)
    with messages.open(newline="") as stream:
        message_rows = list(csv.reader(stream))
    with books.open(newline="") as stream:
        book_rows = list(csv.reader(stream))

    assert len(message_rows) == len(book_rows) == len(result.with_order.message_rows)
    execution = [index for index, row in enumerate(message_rows) if row[1] == "4"]
    assert execution == [4]
    index = execution[0]
    assert message_rows[index][1:] == ["4", "101", "40", "100100", "-1"]
    assert book_rows[index - 1][:4] == ["100100", "100", "100000", "80"]
    assert book_rows[index][:4] == ["100100", "60", "100000", "80"]
    assert book_rows[-1][:4] == ["100100", "45", "100000", "80"]
    assert tuple(result.without_order.orderbook_rows[-1][:4]) == (100100, 85, 100000, 80)
    assert len(result.fills) == 1


def test_continuity_reports_missing_matching_engine_separately(
    openhyra: dict[str, ModuleType], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, golden: FrozenManifest,
) -> None:
    from stochopia.harness import runner
    from stochopia.harness.lob_adapter import LobComparison
    from stochopia.harness.lob_adapter import run_short_lob

    def missing(name: str) -> ModuleType:
        if name.startswith("jaxlob"):
            raise ModuleNotFoundError(f"No module named {name}", name=name)
        return importlib.import_module(name)

    result = run_short_lob(import_module=missing)
    assert result.status is ReportStatus.UNAVAILABLE and result.reason
    assert not result.fills
    assert result.with_order is result.without_order is result.matching_version is None

    def replay_without_dependency(*, depth: int) -> LobComparison:
        return run_short_lob(depth=depth, import_module=missing)

    monkeypatch.setattr(runner, "run_short_lob", replay_without_dependency)
    lob = _short_lob_frozen(golden)
    dev_runner = runner.Runner(lob.experiment)
    fits = {
        spec.candidate_id: dev_runner.fit_candidate(spec.candidate_id, "repeat-1", tmp_path / "fits")
        for spec in lob.experiment.candidates
    }
    tasks = tuple(
        replace(task, parent_fit_id=fits["base"].fit_id, candidate_fit_id=fits[candidate].fit_id)
        for task, candidate in zip(lob.tasks, ("flow", "price"), strict=True)
    )
    verifier = openhyra["verifier"].Verifier(tmp_path / "verification")
    verifier.register(replace(lob, fits=tuple(fits.values()), tasks=tasks))
    report = verifier.run()
    assert len(report["predictions"]) == 4
    assert all(row["status"] == ReportStatus.UNAVAILABLE.value for row in report["predictions"])
    assert all(row["status"] == "degraded" and row["failure_reason"] for row in report["measurement"])
