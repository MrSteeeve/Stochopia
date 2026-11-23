"""Recompute the selection, prediction, and model outcomes from a frozen batch."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from math import inf, isfinite
from numbers import Real
from statistics import mean
from typing import Any

from .contract import (
    AnalysisSpec, Direction, EvaluationTask, FitRecord, FrozenManifest, PairedDelta, PairedEvalResult,
    PredictionVerdict, ReportStatus, SelectionRecord, StructuredPrediction,
)
from .paired_stats import align_observations, estimate_delta, has_readout, is_unavailable_readout

_ReportRow = dict[str, Any]
_ScenarioCandidate = tuple[str, str]


# --- Forecast and selection decisions ---


def _interval_status(measured: tuple[float, float], target: tuple[float, float]) -> ReportStatus:
    """Compare two closed intervals using inclusive boundaries.

    If target.min <= measured.min <= measured.max <= target.max, support the
    claim. If the intervals are disjoint, refute it. An overlap at a shared
    endpoint counts as an overlap and is inconclusive unless fully contained.
    """
    low, high = measured
    target_low, target_high = target
    if target_low <= low and high <= target_high:
        return ReportStatus.SUPPORTED
    if high < target_low or target_high < low:
        return ReportStatus.REFUTED
    return ReportStatus.INCONCLUSIVE


def judge_prediction(
    prediction: StructuredPrediction,
    analysis: AnalysisSpec,
    measured: PairedDelta | None,
    *,
    result_id: str | None = None,
    model_score_delta: float | None = None,
) -> PredictionVerdict:
    """Score the measured CI against the registered forecast regions.

    Increase means [min_effect, infinity) and decrease means
    (-infinity, -min_effect]. Magnitude uses the forecast's signed effect
    interval. Either refuted component refutes the claim, and both must be
    supported to support it. An unchanged forecast uses the frozen symmetric
    or asymmetric equivalence interval, with the same containment rule.
    A point estimate alone is insufficient for any of those comparisons.
    """
    if (prediction.scenario_id, prediction.analysis_id) != (analysis.scenario_id, analysis.analysis_id):
        raise ValueError("prediction does not match the registered analysis")
    common = {"prediction_id": prediction.prediction_id, "model_score_delta": model_score_delta}
    if prediction.abstained:
        return PredictionVerdict(
            **common, status=ReportStatus.ABSTAINED, reason=prediction.abstain_reason,
        )
    if (prediction.readout, prediction.window, prediction.unit) != (analysis.readout, analysis.window, analysis.unit):
        raise ValueError("prediction readout, window or unit does not match the analysis")
    if measured is None or is_unavailable_readout(measured):
        reason = measured.reason if measured is not None else None
        return PredictionVerdict(
            **common, status=ReportStatus.UNAVAILABLE,
            reason=reason or "the frozen evaluation did not provide this readout", result_id=result_id,
        )
    if result_id is None:
        raise ValueError("measured verdict must reference the paired result")
    if (measured.readout, measured.estimator, measured.unit) != (analysis.readout, analysis.estimator, analysis.unit):
        raise ValueError("paired delta does not match the registered readout, estimator or unit")
    if measured.n_pairs < analysis.min_pairs or len(measured.train_repeat_estimates) < analysis.min_train_repeats:
        if measured.interval is not None:
            measured = replace(measured, interval=None, reason="fewer paired days or training repeats than registered")
    interval = measured.interval
    common.update(result_id=result_id, measured=measured,
                  interval_width=interval[1] - interval[0] if interval is not None else None)

    def assess(bounds: tuple[float, float]) -> ReportStatus:
        return _interval_status(interval, bounds) if interval is not None else ReportStatus.INCONCLUSIVE

    if prediction.direction is Direction.UNCHANGED:
        verdict = assess(analysis.equivalence_band)
        return PredictionVerdict(
            **common, status=verdict, equivalence_status=verdict,
            reason=measured.reason if interval is None else f"equivalence is {verdict.value}",
        )
    direction_bounds = (
        (analysis.min_effect, inf) if prediction.direction is Direction.INCREASE else (-inf, -analysis.min_effect)
    )
    direction = assess(direction_bounds)
    magnitude = assess(prediction.effect_interval)
    if ReportStatus.REFUTED in (direction, magnitude):
        status = ReportStatus.REFUTED
    elif direction is magnitude is ReportStatus.SUPPORTED:
        status = ReportStatus.SUPPORTED
    else:
        status = ReportStatus.INCONCLUSIVE
    return PredictionVerdict(
        **common, status=status, direction_status=direction, magnitude_status=magnitude,
        reason=measured.reason if interval is None else f"direction is {direction.value}; magnitude is {magnitude.value}",
    )


def _gate(spec: AnalysisSpec) -> tuple[float, float] | None:
    acceptance = spec.acceptance
    if "min_delta" not in acceptance and "max_delta" not in acceptance:
        return None
    for key in ("min_delta", "max_delta"):
        if key in acceptance:
            value = acceptance[key]
            if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
                raise ValueError(f"{spec.analysis_id}: {key} requires a finite limit")
    low, high = acceptance.get("min_delta", -inf), acceptance.get("max_delta", inf)
    if low > high:
        raise ValueError(f"{spec.analysis_id}: reversed acceptance bounds")
    return (low, high)


def _combine_gates(statuses: list[ReportStatus]) -> ReportStatus:
    if not statuses:
        return ReportStatus.UNAVAILABLE
    for status in (ReportStatus.REFUTED, ReportStatus.UNAVAILABLE, ReportStatus.INCONCLUSIVE):
        if status in statuses:
            return status
    return ReportStatus.SUPPORTED


def assess_candidate(analyses: Iterable[AnalysisSpec], paired_diffs: Mapping[str, PairedDelta]) -> ReportStatus:
    """Apply the frozen minimum and maximum delta gates to a candidate."""
    statuses = []
    for spec in analyses:
        bounds = _gate(spec)
        if bounds is None:
            continue
        delta = paired_diffs.get(spec.analysis_id)
        if delta is None or is_unavailable_readout(delta):
            statuses.append(ReportStatus.UNAVAILABLE)
        else:
            _check_delta_readout(delta, spec)
            if (delta.interval is None or delta.n_pairs < spec.min_pairs
                    or len(delta.train_repeat_estimates) < spec.min_train_repeats):
                statuses.append(ReportStatus.INCONCLUSIVE)
            else:
                statuses.append(_interval_status(delta.interval, bounds))
    return _combine_gates(statuses)


def _check_delta_readout(delta: PairedDelta, spec: AnalysisSpec) -> None:
    if (delta.readout, delta.estimator, delta.unit) != (spec.readout, spec.estimator, spec.unit):
        raise ValueError(f"{spec.analysis_id}: result reported a different readout, estimator, or unit")


# --- Recompute frozen measurements ---


def _registered_delta(result: PairedEvalResult, spec: AnalysisSpec) -> PairedDelta | None:
    delta = result.paired_diffs.get(spec.analysis_id, result.paired_diffs.get(spec.readout))
    if delta is None:
        return None
    _check_delta_readout(delta, spec)
    return delta


def summarize_candidate(
    analyses: Iterable[AnalysisSpec],
    results: Iterable[PairedEvalResult],
    *,
    required_unit_ids: Iterable[str] | None = None,
) -> dict[str, PairedDelta]:
    """Recompute repeat-weighted estimates from raw days, with no task-CI averaging."""
    usable = tuple(result for result in results if result.status != "failed"
                   and (result.observations or result.observations_ref or result.paired_diffs))
    if not usable:
        return {}
    parent_id, candidate_id, scene = usable[0].parent_id, usable[0].candidate_id, usable[0].scenario_id
    if any((row.parent_id, row.candidate_id, row.scenario_id) != (parent_id, candidate_id, scene) for row in usable):
        raise ValueError("candidate aggregation received multiple models or scenarios")
    references = [result.observations_ref for result in usable if result.observations_ref is not None]
    if references and len(usable) > 1:
        raise ValueError("repeat aggregation requires inline paired observations for every task")
    observations = []
    for result in usable:
        observations.extend(result.observations)
    required = tuple(required_unit_ids) if required_unit_ids is not None else None
    deltas = {}
    for spec in analyses:
        if spec.scenario_id != scene:
            continue
        if has_readout(spec, observations):
            deltas[spec.analysis_id] = estimate_delta(
                spec, observations, parent_id=parent_id, candidate_id=candidate_id, required_unit_ids=required,
            )
        elif len(usable) == 1 and references:
            delta = _registered_delta(usable[0], spec)
            if delta is not None:
                deltas[spec.analysis_id] = delta
        else:
            # A task may have excluded every duplicate row before storing raw observations.
            reasons = set()
            unavailable = None
            for result in usable:
                delta = _registered_delta(result, spec)
                if delta is None:
                    continue
                if is_unavailable_readout(delta):
                    unavailable = delta
                elif delta.n_pairs == 0 and delta.reason is not None:
                    reasons.add(delta.reason)
            if reasons:
                empty = estimate_delta(spec, (), parent_id=parent_id, candidate_id=candidate_id)
                deltas[spec.analysis_id] = replace(empty, reason="; ".join(sorted(reasons)))
            elif unavailable is not None:
                deltas[spec.analysis_id] = unavailable
    return deltas


def _verify_result(result: PairedEvalResult, task: EvaluationTask, fits: Mapping[str, FitRecord]) -> None:
    parent = fits[task.parent_fit_id]
    candidate = fits[task.candidate_fit_id]
    expected = (
        task.split, task.scenario_id, task.parent_fit_id, task.candidate_fit_id,
        task.action_ref, task.random_inputs_ref, parent.candidate_id, candidate.candidate_id,
    )
    actual = (
        result.split, result.scenario_id, result.parent_fit_id, result.candidate_fit_id,
        result.action_ref, result.random_inputs_ref, result.parent_id, result.candidate_id,
    )
    if actual != expected:
        raise ValueError(f"{task.task_id}: result does not match the frozen task")
    if any(row.unit_id not in task.unit_ids or row.train_repeat_id != parent.train_repeat_id
           for row in result.observations):
        raise ValueError(f"{task.task_id}: result includes observations outside the frozen task")


def primary_analysis(analyses: Iterable[AnalysisSpec]) -> AnalysisSpec | None:
    """Return the declared primary readout, or the first acceptance gate."""
    analyses = tuple(analyses)
    declared = [spec for spec in analyses if spec.acceptance.get("role") == "primary"]
    if len(declared) > 1:
        raise ValueError("a scenario can register only one primary selection readout")
    if declared:
        return declared[0]
    return next((spec for spec in analyses if _gate(spec) is not None), None)


def _better_sign(spec: AnalysisSpec | None) -> int | None:
    if spec is None:
        return None
    if spec.acceptance.get("score_direction") in ("increase", "decrease"):
        return 1 if spec.acceptance["score_direction"] == "increase" else -1
    bounds = _gate(spec)
    if bounds is None or all(isfinite(bound) for bound in bounds):
        return None
    return -1 if isfinite(bounds[1]) else 1


def model_score_gain(spec: AnalysisSpec | None, delta: PairedDelta | None) -> float | None:
    """Orient a measured delta so positive gain means better performance."""
    sign = _better_sign(spec)
    return None if sign is None or delta is None or delta.estimate is None else sign * delta.estimate


# Compatibility for OpenHyra scoring bridges from before the public API.
_primary = primary_analysis
_gain = model_score_gain


@dataclass(frozen=True)
class _CandidateMeasurement:
    parent_id: str
    measurement_id: str
    result_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    qualification: ReportStatus
    primary_analysis_id: str | None
    model_gain: float | None
    paired_diffs: Mapping[str, PairedDelta]
    comparable: tuple[tuple[str, ...], tuple[str, ...]] | None


@dataclass(frozen=True)
class _FrozenMeasurements:
    analyses: Mapping[str, list[AnalysisSpec]]
    candidates: Mapping[_ScenarioCandidate, _CandidateMeasurement]


def _performance_for(frozen: FrozenManifest, results: tuple[PairedEvalResult, ...]) -> _FrozenMeasurements:
    tasks = {task.task_id: task for task in frozen.tasks}
    fits = {fit.fit_id: fit for fit in frozen.fits}
    by_candidate = defaultdict(list)
    scenarios = defaultdict(list)
    known_results = set()
    for spec in frozen.experiment.analysis:
        scenarios[spec.scenario_id].append(spec)
    for result in results:
        if result.task_id not in tasks or result.result_id in known_results:
            raise ValueError("unexpected frozen task or repeated result ID")
        known_results.add(result.result_id)
        task = tasks.pop(result.task_id)
        _verify_result(result, task, fits)
        by_candidate[result.scenario_id, result.candidate_id].append((task, result))
    if tasks:
        raise ValueError(f"missing frozen task results: {', '.join(sorted(tasks))}")
    performance = {}
    for (scene, candidate), entries in sorted(by_candidate.items()):
        entries.sort(key=lambda item: item[0].task_id)
        compared = tuple(result for _, result in entries)
        required: set[str] = set()
        raw = []
        for task, result in entries:
            required.update(task.unit_ids)
            raw.extend(result.observations)
        analyses = scenarios[scene]
        deltas = summarize_candidate(analyses, compared, required_unit_ids=required)
        primary = primary_analysis(analyses)
        primary_delta = deltas.get(primary.analysis_id) if primary else None
        comparability = None
        if primary is not None and primary_delta is not None and primary_delta.estimate is not None and raw:
            alignment = align_observations(
                primary, raw, parent_id=compared[0].parent_id, candidate_id=candidate,
                required_unit_ids=required,
            )
            comparability = (alignment.days, tuple(alignment.by_repeat))
        measurement_id = compared[0].result_id
        if len(compared) > 1:
            measurement_id = f"aggregate:{scene}:{candidate}"
        performance[scene, candidate] = _CandidateMeasurement(
            parent_id=compared[0].parent_id, measurement_id=measurement_id,
            result_ids=tuple(result.result_id for result in compared), task_ids=tuple(result.task_id for result in compared),
            qualification=assess_candidate(analyses, deltas),
            primary_analysis_id=primary.analysis_id if primary else None,
            model_gain=model_score_gain(primary, primary_delta), paired_diffs=deltas, comparable=comparability,
        )
    return _FrozenMeasurements(scenarios, performance)


# --- Report rows and group summaries ---


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    counts = Counter(values)
    return {status.value: counts[status.value] for status in ReportStatus}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None


def _costs(costs: Iterable[Mapping[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for cost in costs:
        for field, value in cost.items():
            totals[field] += value
    return dict(sorted(totals.items()))


def _common_menu_best(
    frozen: FrozenManifest,
    measurements: _FrozenMeasurements,
    selection: SelectionRecord,
    final: _CandidateMeasurement | None,
) -> float | None:
    scene = selection.scenario_id
    common = []
    for candidate in frozen.experiment.prediction_candidate_set[scene]:
        measured = measurements.candidates.get((scene, candidate))
        if measured is not None and measured.comparable is not None and measured.model_gain is not None:
            common.append(measured)
    if selection.action == "keep_parent":
        supports = {row.comparable for row in common}
        if len(supports) == 1:
            return max(row.model_gain for row in common)
        return None
    if final is None or final.comparable is None:
        return None
    return max((row.model_gain for row in common if row.comparable == final.comparable), default=None)


def _choice_rows(frozen: FrozenManifest, measurements: _FrozenMeasurements) -> Iterator[_ReportRow]:
    scenarios = measurements.analyses
    performance = measurements.candidates
    choices = {(row.group_id, row.scenario_id): row for row in frozen.final_choices}
    by_selection = {row.selection_id: row for row in frozen.selections}
    by_group = defaultdict(list)
    trials = defaultdict(list)
    for row in frozen.selections:
        by_group[row.group_id, row.scenario_id].append(row)
    for row in frozen.search_trials:
        trials[row.group_id, row.scenario_id].append(row)

    def candidate_status(selection: SelectionRecord, scene: str) -> ReportStatus:
        if selection.action == "abstain":
            return ReportStatus.ABSTAINED
        if selection.action == "keep_parent":
            tested = []
            for spec in scenarios[scene]:
                gate = _gate(spec)
                if gate is not None:
                    tested.append(_interval_status((0.0, 0.0), gate))
            return _combine_gates(tested)
        measurement = performance.get((scene, selection.candidate_id))
        return measurement.qualification if measurement is not None else ReportStatus.UNAVAILABLE

    for (group, scene), final in sorted(choices.items()):
        first = min(by_group[group, scene], key=lambda row: row.sequence)
        final_selection = by_selection[final.selection_id]
        primary = primary_analysis(scenarios[scene])
        final_perf = performance.get((scene, final.candidate_id))
        final_gain = final_perf.model_gain if final_perf is not None else None
        if final_selection.action == "keep_parent" and _better_sign(primary) is not None:
            final_gain = 0.0
        first_status = candidate_status(first, scene)
        used = 0.0
        first_pass_cost = None
        charged = []
        for trial in sorted(trials[group, scene], key=lambda row: row.sequence):
            if trial.charged_to_search:
                used += trial.budget_units
                charged.append(trial.cost)
            if (first_pass_cost is None and trial.purpose == "search"
                    and trial.status == "completed" and trial.development_status is ReportStatus.SUPPORTED):
                first_pass_cost = used
        best = _common_menu_best(frozen, measurements, final_selection, final_perf)
        # These gains already use the primary analysis's better-direction sign.
        # A choice outside the shared menu can beat its best member and has zero shortfall.
        gap = max(0.0, best - final_gain) if best is not None and final_gain is not None else None
        first_id = first.candidate_id or (first.parent_id if first.action == "keep_parent" else None)
        yield {
            "group_id": group, "scenario_id": scene, "first_selection_id": first.selection_id,
            "first_candidate": first_id, "first_status": first_status.value,
            "first_frozen_qualified": first_status is ReportStatus.SUPPORTED,
            "first_development_pass_cost": first_pass_cost,
            "search_budget_used": used, "search_cost": _costs(charged),
            "search_status": (ReportStatus.SUPPORTED if first_pass_cost is not None else ReportStatus.REFUTED).value,
            "final_selection_id": final.selection_id, "final_candidate": final.candidate_id,
            "final_status": candidate_status(final_selection, scene).value, "final_model_gain": final_gain,
            "final_result_ids": list(final_perf.result_ids) if final_perf is not None else [],
            "stopping_reason": final.stopping_reason,
            "common_menu_best_gain": best,
            "common_menu_effect_gap": gap,
        }


def _prediction_rows(frozen: FrozenManifest, measurements: _FrozenMeasurements) -> Iterator[_ReportRow]:
    analysis = {spec.analysis_id: spec for spec in frozen.experiment.analysis}
    common = frozen.experiment.prediction_candidate_set
    llm_groups = {group.group_id for group in frozen.experiment.groups if group.kind != "no_llm"}
    first_round_by_scene: dict[str, int] = {}
    for selection in frozen.selections:
        scene = selection.scenario_id
        first_round_by_scene[scene] = min(first_round_by_scene.get(scene, selection.round_index), selection.round_index)
    keys = defaultdict(dict)
    for forecast in sorted(frozen.predictions, key=lambda row: row.prediction_id):
        scene = forecast.scenario_id
        if forecast.round_index == first_round_by_scene[scene] and forecast.candidate_id in common[scene]:
            key = (scene, forecast.candidate_id, forecast.analysis_id)
            keys[key].setdefault(forecast.group_id, forecast.prediction_id)
    comparison_ids: set[str] = set()
    for group_rows in keys.values():
        if group_rows.keys() == llm_groups:
            comparison_ids.update(group_rows.values())
    for forecast in sorted(frozen.predictions, key=lambda row: row.prediction_id):
        spec = analysis[forecast.analysis_id]
        evaluation = measurements.candidates.get((forecast.scenario_id, forecast.candidate_id))
        delta = evaluation.paired_diffs.get(spec.analysis_id) if evaluation is not None else None
        verdict = judge_prediction(
            forecast, spec, delta, result_id=evaluation.measurement_id if evaluation is not None else None,
            model_score_delta=evaluation.model_gain if evaluation is not None else None,
        )
        if forecast.abstained:
            forecast_interval = None
        elif forecast.direction is Direction.UNCHANGED:
            forecast_interval = spec.equivalence_band
        else:
            forecast_interval = forecast.effect_interval
        yield {
            **verdict.to_dict(), "group_id": forecast.group_id, "scenario_id": forecast.scenario_id,
            "candidate_id": forecast.candidate_id, "analysis_id": forecast.analysis_id,
            "round_index": forecast.round_index, "dev_snapshot_id": forecast.dev_snapshot_id,
            "common_comparison": forecast.prediction_id in comparison_ids,
            "predicted_interval_width": None if forecast_interval is None else forecast_interval[1] - forecast_interval[0],
            "result_ids": list(evaluation.result_ids) if evaluation is not None else [],
        }


def _selection_summary(choices: list[_ReportRow], groups: Iterable[str]) -> dict[str, _ReportRow]:
    summary = {}
    for group_id in sorted(groups):
        rows = [row for row in choices if row["group_id"] == group_id]
        first = _counts(row["first_status"] for row in rows)
        final = _counts(row["final_status"] for row in rows)
        found_on_development = sum(row["first_development_pass_cost"] is not None for row in rows)
        total = len(rows)
        summary[group_id] = {
            "n_scenarios": total, "first_status_counts": first,
            "first_qualification_rate": _rate(first[ReportStatus.SUPPORTED.value], total),
            "first_abstention_rate": _rate(first[ReportStatus.ABSTAINED.value], total),
            "final_status_counts": final,
            "final_qualification_rate": _rate(final[ReportStatus.SUPPORTED.value], total),
            "development_pass_rate": _rate(found_on_development, total),
            "search_budget_used": sum(row["search_budget_used"] for row in rows),
            "mean_first_development_pass_cost": _mean(row["first_development_pass_cost"] for row in rows),
            "search_cost": _costs(row["search_cost"] for row in rows),
            "mean_common_menu_effect_gap": _mean(row["common_menu_effect_gap"] for row in rows),
        }
    return summary


def _prediction_summary(predictions: list[_ReportRow], groups: Iterable[str]) -> dict[str, _ReportRow]:
    summary = {}
    for group_id in sorted(groups):
        rows = [row for row in predictions if row["group_id"] == group_id and row["common_comparison"]]
        outcomes = _counts(row["status"] for row in rows)
        coverage = len(rows) - outcomes[ReportStatus.ABSTAINED.value]
        parts = {}
        for name in ("direction", "magnitude", "equivalence"):
            counts = _counts(row[f"{name}_status"] for row in rows)
            hits = counts[ReportStatus.SUPPORTED.value]
            misses = counts[ReportStatus.REFUTED.value]
            decided = hits + misses
            measured = decided + counts[ReportStatus.INCONCLUSIVE.value]
            parts[name] = {
                "counts": counts, "n_decided": decided, "n_measured": measured,
                "hit_rate": _rate(hits, decided), "decidable_rate": _rate(decided, measured),
            }
        summary[group_id] = {
            "n_common_forecasts": len(rows), "prediction_status_counts": outcomes,
            "n_submitted": coverage, "coverage_rate": _rate(coverage, len(rows)),
            "mean_interval_width": _mean(row["interval_width"] for row in rows),
            "mean_predicted_interval_width": _mean(row["predicted_interval_width"] for row in rows),
            "components": parts,
        }
    return summary


def _model_row(scene: str, candidate: str, measurement: _CandidateMeasurement) -> _ReportRow:
    deltas = {}
    training_range = {}
    for key, delta in measurement.paired_diffs.items():
        deltas[key] = delta.to_dict()
        repeats = delta.train_repeat_estimates.values()
        training_range[key] = max(repeats) - min(repeats) if len(repeats) > 1 else None
    return {
        "scenario_id": scene, "candidate_id": candidate, "parent_id": measurement.parent_id,
        "measurement_id": measurement.measurement_id, "result_ids": list(measurement.result_ids),
        "task_ids": list(measurement.task_ids), "qualification": measurement.qualification.value,
        "primary_analysis_id": measurement.primary_analysis_id, "model_gain": measurement.model_gain,
        "paired_diffs": deltas, "training_repeat_range": training_range,
    }


def _task_row(result: PairedEvalResult) -> _ReportRow:
    return {
        "task_id": result.task_id, "result_id": result.result_id, "status": result.status,
        "failure_reason": result.failure_reason, "cost": dict(result.cost),
        "analysis_notes": {key: delta.reason for key, delta in result.paired_diffs.items() if delta.reason},
    }


def build_report(frozen: FrozenManifest, results: Iterable[PairedEvalResult]) -> dict[str, Any]:
    """Publish a JSON-ready report based on the sealed manifest and task results."""
    results = tuple(sorted(results, key=lambda row: row.task_id))
    measurements = _performance_for(frozen, results)
    choices = list(_choice_rows(frozen, measurements))
    predictions = list(_prediction_rows(frozen, measurements))
    final_by_group = {(row["group_id"], row["scenario_id"]): row for row in choices}
    refuted = []
    for row in predictions:
        if row["status"] != ReportStatus.REFUTED.value or (row["model_score_delta"] or 0) <= 0:
            continue
        choice = final_by_group[row["group_id"], row["scenario_id"]]
        refuted.append({
            "group_id": row["group_id"], "scenario_id": row["scenario_id"], "candidate_id": row["candidate_id"],
            "prediction_id": row["prediction_id"], "model_gain": row["model_score_delta"],
            "result_ids": row["result_ids"], "measurement_id": row["result_id"],
            "final_selection_id": choice["final_selection_id"] if choice["final_candidate"] == row["candidate_id"] else None,
        })
    model_rows = []
    for (scene, candidate), measurement in measurements.candidates.items():
        model_rows.append(_model_row(scene, candidate, measurement))
    groups = {group.group_id: group.kind for group in frozen.experiment.groups}
    return {
        "exp_id": frozen.exp_id, "batch_id": frozen.batch_id, "choices": choices,
        "predictions": predictions, "model_performance": model_rows, "refuted_improvements": refuted,
        "measurement": [_task_row(row) for row in results],
        "budgets": {
            "registered": {scope: dict(limits) for scope, limits in frozen.experiment.budget.items()},
            "development_checks": _costs(row.cost for row in frozen.search_trials if not row.charged_to_search),
            "frozen_measurement": _costs(row.cost for row in results),
        },
        "z_to_x": _selection_summary(choices, groups),
        "x_to_delta_y": _prediction_summary(predictions, {key for key, kind in groups.items() if kind != "no_llm"}),
    }
