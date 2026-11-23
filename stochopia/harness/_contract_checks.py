"""Semantic constraints for field-validated P0 records."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .contract import (
        AnalysisSpec, CandidateRef, DataSplit, EvaluationTask, ExperimentManifest, FitRecord,
        FrozenManifest, GroupSpec, PairedDelta, PairedEvalResult, PredictionVerdict, ScenarioSpec,
        SearchTrialRecord, SelectionRecord, StructuredPrediction, UnitObservation,
    )

_RecordT = TypeVar("_RecordT")


def _unique(items: Collection[Hashable], name: str) -> None:
    if len(set(items)) != len(items):
        raise ValueError(f"{name}: expected unique values")


def _index(records: Collection[_RecordT], field_name: str) -> dict[str, _RecordT]:
    indexed = {getattr(record, field_name): record for record in records}
    if len(indexed) != len(records):
        raise ValueError(f"{field_name}: expected unique values")
    return indexed


def _nonnegative(values: Iterable[float], name: str) -> None:
    if any(value < 0 for value in values):
        raise ValueError(f"{name}: expected nonnegative values")


def _interval(bounds: tuple[float, float], name: str) -> None:
    if bounds[0] > bounds[1]:
        raise ValueError(f"{name}: lower bound exceeds upper bound")


def _schema(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"schema: expected {expected}")


def _disjoint_splits(splits: Mapping[str, DataSplit]) -> None:
    seen_ids = set()
    seen_seeds = set()
    for name, split in splits.items():
        if seen_ids.intersection(split.unit_ids):
            raise ValueError(f"splits.{name}: overlapping unit IDs")
        if seen_seeds.intersection(split.unit_seeds.values()):
            raise ValueError(f"splits.{name}: overlapping simulated seeds")
        seen_ids.update(split.unit_ids)
        seen_seeds.update(split.unit_seeds.values())


def _require_simulated_seeds(scenarios: tuple[ScenarioSpec, ...], splits: Mapping[str, DataSplit]) -> None:
    if any(scene.environment_kind == "tier0_sim" for scene in scenarios):
        for name, split in splits.items():
            if split.unit_seeds.keys() != set(split.unit_ids):
                raise ValueError(f"splits.{name}: simulated days require resolved seeds")


# --- Scenario and development manifest checks ---


def check_data_split(split: DataSplit) -> None:
    if not split.source or not split.unit_ids:
        raise ValueError("data split requires a source and at least one unit")
    _unique(split.unit_ids, "unit_ids")
    if not split.unit_seeds.keys() <= set(split.unit_ids):
        raise ValueError("unit_seeds: unknown unit ID")
    seeds = tuple(split.unit_seeds.values())
    _nonnegative(seeds, "unit_seeds")
    _unique(seeds, "unit_seeds")


def check_scenario(scene: ScenarioSpec) -> None:
    _unique(scene.candidate_ids, "candidate_ids")
    if scene.parent_id not in scene.candidate_ids:
        raise ValueError("parent_id: baseline must appear in the candidate menu")
    if not scene.environment or not scene.action_plan:
        raise ValueError("scenario requires environment and action_plan")
    if scene.environment_kind == "lob" and scene.matching_version is None:
        raise ValueError("lob scenario requires matching_version")


def check_candidate(candidate: CandidateRef) -> None:
    _unique(candidate.channels, "channels")
    if candidate.parent_id == candidate.candidate_id:
        raise ValueError("parent_id cannot reference the candidate itself")
    if candidate.channels and candidate.parent_id is None:
        raise ValueError("a candidate with changes requires parent_id")


def check_group(group: GroupSpec) -> None:
    _unique(group.allowed_evidence, "allowed_evidence")
    _nonnegative(group.llm_budget.values(), "llm_budget")
    model_settings = (group.model, group.model_version, group.prompt_template_id)
    if group.kind == "no_llm":
        if group.selection_rule is None or any(model_settings) or group.llm_budget:
            raise ValueError("no_llm requires a selection rule and no LLM settings")
    elif not all(model_settings) or not group.llm_budget or group.selection_rule is not None:
        raise ValueError("LLM groups require model, version, prompt template and budget")
    if bool(group.allowed_evidence) != (group.kind == "evidence_llm"):
        raise ValueError("only evidence_llm receives additional allowed evidence")


def check_analysis(analysis: AnalysisSpec) -> None:
    if not analysis.acceptance:
        raise ValueError("acceptance must register the decision thresholds")
    if analysis.min_effect <= 0:
        raise ValueError("min_effect must be positive")
    low, high = analysis.equivalence_band
    if not low < 0 < high:
        raise ValueError("equivalence_band must straddle zero")
    if analysis.min_pairs < 2 or analysis.min_train_repeats < 1 or analysis.n_resamples < 1:
        raise ValueError("analysis requires at least two pairs and positive repeat counts")
    if not 0 < analysis.alpha < 1 or analysis.seed < 0:
        raise ValueError("analysis requires 0 < alpha < 1 and a nonnegative seed")


def check_experiment(manifest: ExperimentManifest) -> None:
    from .contract import MANIFEST_SCHEMA

    _schema(manifest.schema, MANIFEST_SCHEMA)
    if manifest.splits.keys() != {"train", "dev"}:
        raise ValueError("development splits must contain exactly train and dev")
    _disjoint_splits(manifest.splits)
    if not manifest.train_seeds or not manifest.train_plan:
        raise ValueError("training requires repeat seeds and a plan")
    _nonnegative(manifest.train_seeds.values(), "train_seeds")
    _unique(tuple(manifest.train_seeds.values()), "train_seeds")
    if manifest.seeds.keys() != {"rollout", "delta_ci"}:
        raise ValueError("development seeds must contain rollout and delta_ci")
    _nonnegative(manifest.seeds.values(), "seeds")
    if manifest.budget.keys() != {"search", "final_check"}:
        raise ValueError("budget must separate search and final_check")
    for name, budget in manifest.budget.items():
        if not budget:
            raise ValueError(f"budget.{name}: expected registered cost units")
        _nonnegative(budget.values(), f"budget.{name}")

    groups = _index(manifest.groups, "group_id")
    llm_groups = [group for group in groups.values() if group.kind != "no_llm"]
    if not llm_groups or len(groups) - len(llm_groups) != 1:
        raise ValueError("expected at least one LLM group and exactly one no_llm group")
    policies = {
        (group.model, group.model_version, group.prompt_template_id, tuple(sorted(group.llm_budget.items())))
        for group in llm_groups
    }
    if len(policies) != 1:
        raise ValueError("LLM groups must match model, model version, prompt template and budget")

    scenarios = _index(manifest.scenarios, "scenario_id")
    candidates = _index(manifest.candidates, "candidate_id")
    analysis = _index(manifest.analysis, "analysis_id")
    if not scenarios or manifest.prediction_candidate_set.keys() != scenarios.keys():
        raise ValueError("common prediction sets must match the registered scenarios")
    _require_simulated_seeds(manifest.scenarios, manifest.splits)
    if {item.scenario_id for item in analysis.values()} != set(scenarios):
        raise ValueError("analysis must cover every scenario")
    for candidate in manifest.candidates:
        if candidate.parent_id is not None and candidate.parent_id not in candidates:
            raise ValueError(f"{candidate.candidate_id}: unknown parent_id")
    for scenario in manifest.scenarios:
        menu = set(scenario.candidate_ids)
        if not menu <= candidates.keys():
            raise ValueError(f"{scenario.scenario_id}: unknown candidate in menu")
        changes = menu - {scenario.parent_id}
        channels = {channel for key in changes for channel in candidates[key].channels}
        if channels != {"flow", "price"}:
            raise ValueError(f"{scenario.scenario_id}: menu must cover flow and price")
        common = manifest.prediction_candidate_set[scenario.scenario_id]
        _unique(common, "prediction_candidate_set")
        if not common or not set(common) <= changes:
            raise ValueError("common prediction set must contain non-baseline menu candidates")


# --- Selection, forecasts and search costs ---


def check_selection(selection: SelectionRecord) -> None:
    _nonnegative((selection.round_index, selection.sequence), "round and sequence")
    if selection.action == "select":
        if selection.candidate_id in (None, selection.parent_id):
            raise ValueError("select requires a non-baseline candidate_id")
    elif selection.candidate_id is not None:
        raise ValueError("only select takes a candidate_id")


def check_prediction(prediction: StructuredPrediction) -> None:
    from .contract import Direction, PREDICTION_SCHEMA

    _schema(prediction.schema, PREDICTION_SCHEMA)
    _nonnegative((prediction.round_index,), "round_index")
    if prediction.candidate_id == prediction.parent_id:
        raise ValueError("prediction requires a non-baseline candidate_id")
    if prediction.abstained:
        if prediction.abstain_reason is None:
            raise ValueError("abstain_reason is required for an abstention")
        if prediction.direction is not None or prediction.effect_interval is not None:
            raise ValueError("abstention cannot include a directional forecast")
        return
    if prediction.abstain_reason is not None:
        raise ValueError("abstain_reason requires abstained=True")
    for name in ("readout", "unit", "window", "direction", "conditions", "falsifier"):
        if getattr(prediction, name) is None:
            raise ValueError(f"{name} is required for a prediction")
    if prediction.direction is Direction.UNCHANGED:
        if prediction.effect_interval is not None:
            raise ValueError("unchanged predictions use the analysis equivalence band")
        return
    if prediction.effect_interval is None:
        raise ValueError("directional predictions require an effect_interval")
    _interval(prediction.effect_interval, "effect_interval")
    low, high = prediction.effect_interval
    if (prediction.direction is Direction.INCREASE and low <= 0
            or prediction.direction is Direction.DECREASE and high >= 0):
        raise ValueError("effect_interval must agree with the prediction direction")


def check_trial(trial: SearchTrialRecord) -> None:
    from .contract import ReportStatus

    _nonnegative((trial.sequence, trial.budget_units), "trial sequence and budget")
    _nonnegative(trial.cost.values(), "cost")
    if trial.purpose == "search" and (trial.selection_id is None or not trial.charged_to_search):
        raise ValueError("search trial requires a selection_id and must be charged to search")
    if trial.status == "completed" and trial.development_status not in (
        ReportStatus.SUPPORTED, ReportStatus.REFUTED, ReportStatus.INCONCLUSIVE
    ):
        raise ValueError("completed trial requires a development assessment")
    if trial.status in ("pending", "failed", "unavailable") and trial.development_status not in (
        None, ReportStatus.UNAVAILABLE
    ):
        raise ValueError("trial without a completed measurement cannot claim an assessment")


# --- Fit and paired measurements ---


def check_fit(fit: FitRecord) -> None:
    _nonnegative((fit.seed,), "seed")
    if not fit.config:
        raise ValueError("config is required to replay a fit")
    if fit.parent_id == fit.candidate_id:
        raise ValueError("parent_id cannot reference the candidate itself")


def check_observation(observation: UnitObservation) -> None:
    from .contract import ArmKind

    if observation.arm is not ArmKind.TRUTH and observation.candidate_id is None:
        raise ValueError("model-side observation requires candidate_id")
    if not (observation.response_path or observation.truth_path or observation.values):
        raise ValueError("observation requires a path or raw values")
    if observation.response_path and observation.truth_path and len(observation.response_path) != len(observation.truth_path):
        raise ValueError("response_path and truth_path must share the event window")


def check_delta(delta: PairedDelta) -> None:
    _nonnegative((delta.n_pairs, delta.seed), "n_pairs and seed")
    if not 0 < delta.confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")
    if (delta.estimate is None) != (delta.n_pairs == 0):
        raise ValueError("a point estimate requires at least one pair")
    if delta.interval is None:
        if delta.reason is None:
            raise ValueError("missing interval requires a reason")
    else:
        _interval(delta.interval, "interval")
        if delta.n_pairs < 2:
            raise ValueError("interval requires at least two pairs")


def check_paired_result(result: PairedEvalResult) -> None:
    from .contract import ArmKind, PAIRED_EVAL_SCHEMA

    _schema(result.schema, PAIRED_EVAL_SCHEMA)
    if result.candidate_id == result.parent_id or result.parent_fit_id == result.candidate_fit_id:
        raise ValueError("paired result requires distinct candidate and baseline fits")
    _unique(result.arms, "arms")
    _nonnegative(result.cost.values(), "cost")
    if result.observations and result.observations_ref is not None:
        raise ValueError("provide inline observations or an observation reference")
    if result.status == "completed":
        if not result.arms or not result.paired_diffs or not (result.observations or result.observations_ref):
            raise ValueError("completed result requires arms, day observations and paired deltas")
        if not {"train_seconds", "eval_seconds"} <= result.cost.keys():
            raise ValueError("completed cost must include training and evaluation time")
        if result.failure_reason is not None:
            raise ValueError("completed result cannot carry a failure_reason")
    elif result.failure_reason is None:
        raise ValueError("failed or degraded result requires a failure_reason")
    keys = []
    model_keys = {result.candidate_id: set(), result.parent_id: set()}
    for observation in result.observations:
        if (observation.scenario_id != result.scenario_id or observation.arm not in result.arms
                or observation.candidate_id not in (None, result.candidate_id, result.parent_id)):
            raise ValueError("observation does not belong to the result's scene, arms or models")
        key = (observation.unit_id, observation.train_repeat_id, observation.arm,
               observation.window, observation.readout, observation.unit)
        keys.append((observation.candidate_id, key))
        if observation.candidate_id is not None and observation.arm is not ArmKind.TRUTH:
            model_keys[observation.candidate_id].add(key)
    _unique(keys, "day observation keys")
    if result.observations and any(delta.n_pairs for delta in result.paired_diffs.values()):
        if not model_keys[result.candidate_id] & model_keys[result.parent_id]:
            raise ValueError("reported paired deltas require aligned parent and candidate observations")


def check_verdict(verdict: PredictionVerdict) -> None:
    from .contract import ReportStatus

    if verdict.measured is not None and verdict.result_id is None:
        raise ValueError("measured prediction requires result_id")
    if verdict.status in (ReportStatus.SUPPORTED, ReportStatus.REFUTED):
        if verdict.measured is None or verdict.measured.interval is None:
            raise ValueError("a decided prediction requires a measured interval")
    if verdict.interval_width is not None:
        _nonnegative((verdict.interval_width,), "interval_width")
        if verdict.measured is None or verdict.measured.interval is None:
            raise ValueError("interval_width requires a measured interval")
    if verdict.equivalence_status is not None and (
        verdict.direction_status is not None or verdict.magnitude_status is not None
    ):
        raise ValueError("equivalence and directional assessments are mutually exclusive")
    parts = (verdict.direction_status, verdict.magnitude_status, verdict.equivalence_status)
    if verdict.status in (ReportStatus.ABSTAINED, ReportStatus.UNAVAILABLE):
        if any(part is not None for part in parts):
            raise ValueError("an unavailable or abstained prediction has no scored components")
        if verdict.status is ReportStatus.ABSTAINED and verdict.measured is not None:
            raise ValueError("an abstained prediction has no measured effect")
        return
    if verdict.measured is None:
        raise ValueError("assessed prediction requires a measured effect")
    components = ((verdict.equivalence_status,) if verdict.equivalence_status is not None
                  else (verdict.direction_status, verdict.magnitude_status))
    allowed = (ReportStatus.SUPPORTED, ReportStatus.REFUTED, ReportStatus.INCONCLUSIVE)
    if any(component not in allowed for component in components):
        raise ValueError("assessment requires equivalence or both direction and magnitude statuses")
    if verdict.measured.interval is None and any(
        component is not ReportStatus.INCONCLUSIVE for component in components
    ):
        raise ValueError("without an interval, assessment components must be inconclusive")
    if ReportStatus.REFUTED in components:
        expected = ReportStatus.REFUTED
    elif ReportStatus.INCONCLUSIVE in components:
        expected = ReportStatus.INCONCLUSIVE
    else:
        expected = ReportStatus.SUPPORTED
    if verdict.status is not expected:
        raise ValueError("overall verdict disagrees with its scored components")


# --- Sealed verification checks ---


def check_task(task: EvaluationTask) -> None:
    if task.parent_fit_id == task.candidate_fit_id or not task.unit_ids:
        raise ValueError("evaluation task requires two fits and at least one unit")
    _unique(task.unit_ids, "unit_ids")


def check_frozen(frozen: FrozenManifest) -> None:
    from .contract import FROZEN_MANIFEST_SCHEMA

    _schema(frozen.schema, FROZEN_MANIFEST_SCHEMA)
    if frozen.exp_id != frozen.experiment.exp_id:
        raise ValueError("exp_id does not match the sealed development manifest")
    splits = {**frozen.experiment.splits, "test": frozen.test_split}
    _disjoint_splits(splits)
    _require_simulated_seeds(frozen.experiment.scenarios, splits)
    if frozen.seeds.keys() != {"rollout", "delta_ci"}:
        raise ValueError("test seeds must contain rollout and delta_ci")
    _nonnegative(frozen.seeds.values(), "seeds")
    refs = _FrozenReferences(frozen)
    first_choices = _check_frozen_choices(frozen, refs)
    _check_frozen_predictions(frozen, refs, first_choices)
    _check_frozen_trials(frozen, refs)
    _check_frozen_tasks(frozen, refs, first_choices)


class _FrozenReferences:
    """Index ids once for referential checks across the sealed payload."""

    def __init__(self, frozen: FrozenManifest) -> None:
        experiment = frozen.experiment
        self.scenarios = {item.scenario_id: item for item in experiment.scenarios}
        self.candidates = {item.candidate_id: item for item in experiment.candidates}
        self.groups = {item.group_id: item for item in experiment.groups}
        self.analysis = {item.analysis_id: item for item in experiment.analysis}
        self.selections = _index(frozen.selections, "selection_id")
        self.fits = _index(frozen.fits, "fit_id")

    def subject(
        self, record: SelectionRecord | StructuredPrediction | SearchTrialRecord, exp_id: str,
    ) -> ScenarioSpec:
        if record.exp_id != exp_id:
            raise ValueError("record exp_id does not match frozen experiment")
        if record.group_id not in self.groups or record.scenario_id not in self.scenarios:
            raise ValueError("record references an unknown group or scenario")
        return self.scenarios[record.scenario_id]

    def selection(self, selection_id: str, group_id: str, scenario_id: str) -> SelectionRecord:
        selection = self.selections.get(selection_id)
        if selection is None or (group_id, scenario_id) != (selection.group_id, selection.scenario_id):
            raise ValueError("selection reference does not match the group and scenario")
        return selection


def _check_frozen_choices(frozen: FrozenManifest, refs: _FrozenReferences) -> dict[tuple[str, str], SelectionRecord]:
    sequence_keys = []
    for record in frozen.selections:
        scene = refs.subject(record, frozen.exp_id)
        if record.parent_id != scene.parent_id or (
            record.candidate_id is not None and record.candidate_id not in scene.candidate_ids
        ):
            raise ValueError("selection baseline or candidate does not match the scenario")
        sequence_keys.append((record.group_id, record.scenario_id, record.sequence))
    _unique(sequence_keys, "selection sequence")
    expected = {(group, scene) for group in refs.groups for scene in refs.scenarios}
    choices = [(choice.group_id, choice.scenario_id) for choice in frozen.final_choices]
    _unique(choices, "final choices")
    if set(choices) != expected:
        raise ValueError("final choices must cover every group and scenario")
    for choice in frozen.final_choices:
        selection = refs.selection(choice.selection_id, choice.group_id, choice.scenario_id)
        if choice.candidate_id != (selection.candidate_id or selection.parent_id):
            raise ValueError("final choice must follow the recorded selection")
    first_choices = {}
    for row in sorted(frozen.selections, key=lambda selection: selection.sequence):
        first_choices.setdefault((row.group_id, row.scenario_id), row)
    scene_snapshots = {}
    for row in first_choices.values():
        basis = (row.round_index, row.dev_snapshot_id)
        if scene_snapshots.setdefault(row.scenario_id, basis) != basis:
            raise ValueError("initial choices must share a development snapshot within a scenario")
    return first_choices


def _check_frozen_predictions(
    frozen: FrozenManifest, refs: _FrozenReferences, first_choices: Mapping[tuple[str, str], SelectionRecord],
) -> None:
    _index(frozen.predictions, "prediction_id")
    snapshots: dict[tuple[str, int], str] = {}
    for row in refs.selections.values():
        key = (row.group_id, row.round_index)
        if snapshots.setdefault(key, row.dev_snapshot_id) != row.dev_snapshot_id:
            raise ValueError("selections in the same group and round must share a development snapshot")
    covered: dict[tuple[str, str, str], set[str]] = {}
    for prediction in frozen.predictions:
        scene = refs.subject(prediction, frozen.exp_id)
        group = refs.groups[prediction.group_id]
        if group.kind == "no_llm":
            raise ValueError("no_llm cannot submit predictions")
        analysis = refs.analysis.get(prediction.analysis_id)
        if analysis is None or analysis.scenario_id != scene.scenario_id:
            raise ValueError("prediction references unknown or unrelated analysis_id")
        if prediction.parent_id != scene.parent_id or prediction.candidate_id not in scene.candidate_ids:
            raise ValueError("prediction baseline or candidate does not match the scenario")
        first_choice = first_choices[(prediction.group_id, prediction.scenario_id)]
        snapshot = snapshots.get((prediction.group_id, prediction.round_index))
        if prediction.round_index < first_choice.round_index or prediction.dev_snapshot_id != snapshot:
            raise ValueError("prediction must match the registered group and round development snapshot")
        if not prediction.abstained and (prediction.readout, prediction.window, prediction.unit) != (
            analysis.readout, analysis.window, analysis.unit
        ):
            raise ValueError("prediction readout, window and unit must match the analysis")
        if prediction.round_index == first_choice.round_index:
            key = (prediction.group_id, prediction.scenario_id, prediction.candidate_id)
            covered.setdefault(key, set()).add(prediction.analysis_id)

    llm_groups = [group.group_id for group in refs.groups.values() if group.kind != "no_llm"]
    for scene, common in frozen.experiment.prediction_candidate_set.items():
        for candidate in common:
            expected = covered.get((llm_groups[0], scene, candidate))
            if not expected or any(covered.get((group, scene, candidate)) != expected for group in llm_groups):
                raise ValueError("LLM groups must cover each common candidate on the same shared analysis set")


def _check_frozen_trials(frozen: FrozenManifest, refs: _FrozenReferences) -> None:
    _index(frozen.search_trials, "trial_id")
    sequence_keys = []
    for trial in frozen.search_trials:
        scene = refs.subject(trial, frozen.exp_id)
        if trial.status == "pending" or trial.candidate_id not in scene.candidate_ids:
            raise ValueError("frozen trial must be terminal and refer to a menu candidate")
        if trial.selection_id is not None:
            selection = refs.selection(trial.selection_id, trial.group_id, trial.scenario_id)
            if trial.purpose == "search" and selection.candidate_id != trial.candidate_id:
                raise ValueError("search trial did not run its selected candidate")
        sequence_keys.append((trial.group_id, trial.scenario_id, trial.sequence))
    _unique(sequence_keys, "trial sequence")


def _check_frozen_tasks(
    frozen: FrozenManifest, refs: _FrozenReferences, first_choices: Mapping[tuple[str, str], SelectionRecord],
) -> None:
    for fit in frozen.fits:
        candidate = refs.candidates.get(fit.candidate_id)
        if candidate is None or candidate.parent_id != fit.parent_id:
            raise ValueError("fit candidate or lineage does not match the manifest")
        if frozen.experiment.train_seeds.get(fit.train_repeat_id) != fit.seed:
            raise ValueError("fit seed does not match the registered training repeat")
    _index(frozen.tasks, "task_id")
    test_units = set(frozen.test_split.unit_ids)
    covered = set()
    for task in frozen.tasks:
        scene = refs.scenarios.get(task.scenario_id)
        parent = refs.fits.get(task.parent_fit_id)
        candidate = refs.fits.get(task.candidate_fit_id)
        if scene is None or parent is None or candidate is None:
            raise ValueError("task references an unknown scenario or fit")
        if task.split != "test" or not set(task.unit_ids) <= test_units:
            raise ValueError("frozen task may only use registered test units")
        if (parent.candidate_id != scene.parent_id or candidate.candidate_id == scene.parent_id
                or candidate.candidate_id not in scene.candidate_ids):
            raise ValueError("task fits do not match the baseline and candidate menu")
        if parent.train_repeat_id != candidate.train_repeat_id:
            raise ValueError("task fits must share a training repeat")
        if parent.train_data_ref != candidate.train_data_ref:
            raise ValueError("task fits must share the training data reference")
        covered.add((scene.scenario_id, candidate.candidate_id))

    required = {(row.scenario_id, row.candidate_id) for row in frozen.predictions if not row.abstained}
    required.update((row.scenario_id, row.candidate_id) for row in first_choices.values()
                    if row.candidate_id is not None)
    required.update(
        (row.scenario_id, row.candidate_id) for row in frozen.final_choices
        if row.candidate_id != refs.scenarios[row.scenario_id].parent_id
    )
    if not required <= covered:
        raise ValueError("frozen tasks must cover first choices, final choices and common predictions")
