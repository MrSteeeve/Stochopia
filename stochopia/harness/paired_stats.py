"""Pair by day and resample the same day indices on both sides of each comparison."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from math import isfinite, sqrt
from random import Random
from statistics import mean

from .contract import (
    AnalysisSpec, ArmKind, EvaluationTask, FitRecord, PairedDelta, PairedEvalResult, ScenarioSpec, UnitObservation,
)

_MEAN_ESTIMATORS = {"mean", "daily_mean"}
_CURVE_ESTIMATOR = "mean_path_rmse"
_RATIO_ESTIMATOR = "ratio_of_sums"

# --- Observation alignment ---


@dataclass(frozen=True)
class _DayPair:
    parent: UnitObservation
    candidate: UnitObservation
    truth: tuple[float, ...] = ()


@dataclass(frozen=True)
class PairAlignment:
    """Per-repeat records and the complete day IDs shared by all usable repeats."""

    days: tuple[str, ...]
    by_repeat: Mapping[str, Mapping[str, _DayPair]]
    excluded: tuple[str, ...]


def _observation_key(row: UnitObservation) -> tuple[str, str, str | None, ArmKind, str, str, str, str]:
    return (row.unit_id, row.train_repeat_id, row.candidate_id, row.arm,
            row.scenario_id, row.readout, row.window, row.unit)


def _is_scalar(value: object) -> bool:
    return type(value) in (int, float) and isfinite(value)


def _make_pair(
    analysis: AnalysisSpec,
    parent: UnitObservation,
    candidate: UnitObservation,
    reference: UnitObservation | None,
) -> _DayPair | None:
    offsets = parent.values.get("event_offsets")
    if offsets != candidate.values.get("event_offsets"):
        return None
    if analysis.estimator in _MEAN_ESTIMATORS:
        if all(_is_scalar(row.values.get("value")) for row in (parent, candidate)):
            return _DayPair(parent, candidate)
    elif analysis.estimator == _RATIO_ESTIMATOR:
        if all(_is_scalar(row.values.get("numerator")) and
               _is_scalar(row.values.get("denominator")) and row.values["denominator"] > 0
               for row in (parent, candidate)):
            return _DayPair(parent, candidate)
    elif analysis.estimator == _CURVE_ESTIMATOR:
        references = {row.truth_path for row in (parent, candidate) if row.truth_path}
        if reference is not None:
            if reference.truth_path or reference.response_path:
                references.add(reference.truth_path or reference.response_path)
            if reference.values.get("event_offsets", offsets) != offsets:
                return None
        if len(references) == 1:
            truth = next(iter(references))
            if offsets is not None and (
                not isinstance(offsets, tuple) or len(offsets) != len(truth)
                or not all(_is_scalar(item) or isinstance(item, str) for item in offsets)
            ):
                return None
            if "event_offsets" in analysis.acceptance and offsets != analysis.acceptance["event_offsets"]:
                return None
            if len(parent.response_path) == len(candidate.response_path) == len(truth):
                return _DayPair(parent, candidate, truth)
    return None


def _exclude_mixed_curve_windows(by_repeat: dict[str, dict[str, _DayPair]], exclusions: list[str]) -> None:
    signatures = set()
    for pairs in by_repeat.values():
        for pair in pairs.values():
            signatures.add((len(pair.truth), pair.parent.values.get("event_offsets")))
    if len(signatures) < 2:
        return
    for repeat_id, pairs in by_repeat.items():
        for day in pairs:
            exclusions.append(f"{repeat_id}/{day}: misaligned curve grid across days")
        pairs.clear()


def has_readout(analysis: AnalysisSpec, rows: Iterable[UnitObservation]) -> bool:
    """Whether at least one model-side row exposes this registered readout."""
    arm_filter = analysis.acceptance.get("arm")
    return any(
        row.scenario_id == analysis.scenario_id and row.readout == analysis.readout
        and row.window == analysis.window and row.unit == analysis.unit and row.arm is not ArmKind.TRUTH
        and (arm_filter is None or row.arm.value == arm_filter)
        for row in rows
    )


def align_observations(
    analysis: AnalysisSpec,
    observations: Iterable[UnitObservation],
    *,
    parent_id: str,
    candidate_id: str,
    required_unit_ids: Iterable[str] | None = None,
) -> PairAlignment:
    """Exclude unpaired, duplicate, or incompatible rows with a day-level explanation."""
    analysis.validate()
    if parent_id == candidate_id:
        raise ValueError("pairing requires distinct parent and candidate IDs")
    if analysis.estimator not in _MEAN_ESTIMATORS | {_CURVE_ESTIMATOR, _RATIO_ESTIMATOR}:
        raise ValueError(f"unsupported paired estimator: {analysis.estimator}")
    required = set(required_unit_ids) if required_unit_ids is not None else None
    arm_filter = analysis.acceptance.get("arm")
    registered_arm = ArmKind(arm_filter) if arm_filter is not None else None
    if registered_arm is ArmKind.TRUTH:
        raise ValueError("registered analysis arm must be a model arm")
    indexed: dict[tuple[str, str], list[UnitObservation]] = defaultdict(list)
    other_context: set[tuple[str, str]] = set()
    exclusions: list[str] = []
    for row in observations:
        if row.scenario_id != analysis.scenario_id or row.readout != analysis.readout:
            continue
        if row.candidate_id not in (None, parent_id, candidate_id):
            continue
        row.validate()
        location = f"{row.train_repeat_id}/{row.unit_id}"
        if (row.window, row.unit) != (analysis.window, analysis.unit):
            other_context.add((row.train_repeat_id, row.unit_id))
            continue
        if required is not None and row.unit_id not in required:
            exclusions.append(f"{location}: outside registered unit IDs")
            continue
        if registered_arm is not None and row.arm not in (registered_arm, ArmKind.TRUTH):
            continue
        indexed[row.train_repeat_id, row.unit_id].append(row)

    contexts = set(indexed) | other_context
    repeats = sorted({repeat for repeat, _ in contexts})
    unit_ids = required if required is not None else {day for _, day in contexts}
    by_repeat = {repeat: {} for repeat in repeats}
    for repeat_id in repeats:
        for day in sorted(unit_ids):
            location = f"{repeat_id}/{day}"
            rows = indexed[repeat_id, day]
            keys = Counter(_observation_key(row) for row in rows)
            duplicate = {key for key, count in keys.items() if count > 1}
            if duplicate:
                exclusions.append(f"{location}: duplicate observation keys")
            clean = [row for row in rows if _observation_key(row) not in duplicate]
            parent = [row for row in clean if row.candidate_id == parent_id and row.arm != ArmKind.TRUTH]
            candidate = [row for row in clean if row.candidate_id == candidate_id and row.arm != ArmKind.TRUTH]
            truth = [row for row in clean if row.arm is ArmKind.TRUTH]
            if not parent or not candidate:
                exclusions.append(f"{location}: missing parent or candidate observation")
                if (repeat_id, day) in other_context:
                    exclusions.append(f"{location}: misaligned window or unit")
                continue
            if duplicate or len(parent) != 1 or len(candidate) != 1 or len(truth) > 1:
                exclusions.append(f"{location}: ambiguous model arm or truth observation")
                continue
            if parent[0].arm != candidate[0].arm:
                exclusions.append(f"{location}: misaligned model arms")
                continue
            paired = _make_pair(analysis, parent[0], candidate[0], truth[0] if truth else None)
            if paired is None:
                exclusions.append(f"{location}: missing or misaligned measurements")
            else:
                by_repeat[repeat_id][day] = paired

    if analysis.estimator == _CURVE_ESTIMATOR:
        _exclude_mixed_curve_windows(by_repeat, exclusions)
    by_repeat = {repeat: pairs for repeat, pairs in by_repeat.items() if pairs}
    common_days = set(unit_ids)
    for pairs in by_repeat.values():
        common_days.intersection_update(pairs)
    days = tuple(sorted(common_days)) if by_repeat else ()
    for repeat, pairs in by_repeat.items():
        for day in sorted(set(pairs).difference(days)):
            exclusions.append(f"{repeat}/{day}: not a complete day across training repeats")
    return PairAlignment(days=days, by_repeat=by_repeat, excluded=tuple(sorted(set(exclusions))))


# --- Estimators and resampling ---


def _rmse(response_paths: list[tuple[float, ...]], truth_paths: list[tuple[float, ...]]) -> float:
    """Compute sqrt(mean_t((mean_d response[d,t] - mean_d truth[d,t]) ** 2)).

    Both arrays have shape (drawn days, event times), including duplicate days
    from a bootstrap draw. Pool across days before applying the nonlinear RMSE.
    """
    response_by_time = zip(*response_paths, strict=True)
    truth_by_time = zip(*truth_paths, strict=True)
    squared_mean_residuals = []
    for responses, truths in zip(response_by_time, truth_by_time, strict=True):
        squared_mean_residuals.append((mean(responses) - mean(truths)) ** 2)
    return sqrt(mean(squared_mean_residuals))


def _ratio(rows: Iterable[UnitObservation]) -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        numerator += row.values["numerator"]
        denominator += row.values["denominator"]
    return numerator / denominator


def _difference(estimator: str, pairs: Iterable[_DayPair]) -> float:
    """Apply the selected estimator to both sides and return candidate minus parent."""
    pairs = tuple(pairs)
    if estimator == _CURVE_ESTIMATOR:
        truths = [pair.truth for pair in pairs]
        parent = _rmse([pair.parent.response_path for pair in pairs], truths)
        candidate = _rmse([pair.candidate.response_path for pair in pairs], truths)
        return candidate - parent
    if estimator == _RATIO_ESTIMATOR:
        return _ratio(pair.candidate for pair in pairs) - _ratio(pair.parent for pair in pairs)
    return mean(pair.candidate.values["value"] - pair.parent.values["value"] for pair in pairs)


def _quantile(sorted_values: list[float], probability: float) -> float:
    """Interpolate between order statistics at zero-based rank (B - 1) * p."""
    rank = (len(sorted_values) - 1) * probability
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (rank - lower) * (sorted_values[upper] - sorted_values[lower])


def _weights(analysis: AnalysisSpec, repeat_ids: Iterable[str]) -> dict[str, float]:
    registered = analysis.acceptance.get("repeat_weights")
    if registered is None:
        return {repeat: 1.0 for repeat in repeat_ids}
    if not isinstance(registered, Mapping) or any(
        repeat not in registered or not _is_scalar(registered[repeat]) or registered[repeat] <= 0
        for repeat in repeat_ids
    ):
        raise ValueError("repeat_weights must give every available repeat a positive registered weight")
    return {repeat: registered[repeat] for repeat in repeat_ids}


def _draws(
    analysis: AnalysisSpec,
    days: tuple[str, ...],
    resamples: Iterable[Iterable[str]] | None,
) -> Iterable[Iterable[str]]:
    """Supply ordered draws of N complete day IDs, with replacement.

    Explicit draws let the caller check a mathematical fixture; the estimator
    validates that each draw contains exactly N IDs from the paired days.
    Exhaustive pairing enumerates all N ** N ordered draws and requires a
    registered draw budget at least that large. Ordinary pairing draws exactly
    n_resamples times from the independently registered analysis seed.
    """
    if resamples is not None:
        return resamples
    if analysis.pairing == "day_exhaustive":
        if analysis.n_resamples < len(days) ** len(days):
            raise ValueError("day_exhaustive requires n_resamples sufficient for all ordered day draws")
        return product(days, repeat=len(days))
    rng = Random(analysis.seed)
    return (rng.choices(days, k=len(days)) for _ in range(analysis.n_resamples))


def estimate_delta(
    analysis: AnalysisSpec,
    observations: Iterable[UnitObservation],
    *,
    parent_id: str,
    candidate_id: str,
    required_unit_ids: Iterable[str] | None = None,
    resamples: Iterable[Iterable[str]] | None = None,
) -> PairedDelta:
    """Estimate the paired delta and percentile interval over independent days.

    Every draw has N complete day IDs with replacement, shared by the parent,
    candidate, truth and all training repeats. For each draw, compute each
    repeat's candidate-minus-parent delta and take its weighted repeat mean.
    The CI endpoints are the alpha/2 and (1 - alpha/2) quantiles of those means.
    Training repeats contribute no independent days. Below either minimum,
    return the point estimate where available and omit the interval.
    """
    if analysis.pairing not in ("day", "day_exhaustive"):
        raise ValueError(f"unsupported pairing declaration: {analysis.pairing}")
    paired = align_observations(
        analysis, observations, parent_id=parent_id, candidate_id=candidate_id, required_unit_ids=required_unit_ids,
    )
    by_repeat = paired.by_repeat
    days = paired.days
    n_pairs = len(days)
    weights = _weights(analysis, by_repeat)

    def differences(indices: tuple[str, ...]) -> dict[str, float]:
        estimates = {}
        for repeat_id, repeat in by_repeat.items():
            estimates[repeat_id] = _difference(analysis.estimator, (repeat[day] for day in indices))
        return estimates

    def aggregate(estimates: Mapping[str, float]) -> float:
        return sum(value * weights[repeat] for repeat, value in estimates.items()) / sum(weights.values())

    repeat_estimates = differences(days) if days else {
        repeat_id: _difference(analysis.estimator, tuple(repeat.values()))
        for repeat_id, repeat in by_repeat.items()
    }
    estimate = aggregate(repeat_estimates) if days else None
    interval = None
    reasons = []
    if n_pairs < analysis.min_pairs:
        reasons.append(f"requires {analysis.min_pairs} paired days, found {n_pairs}")
    if len(by_repeat) < analysis.min_train_repeats:
        reasons.append(f"requires {analysis.min_train_repeats} training repeats, found {len(by_repeat)}")
    if not reasons:
        allowed_days = set(days)
        bootstrap = []
        for row in _draws(analysis, days, resamples):
            draw = tuple(row)
            if len(draw) != n_pairs or not set(draw) <= allowed_days:
                raise ValueError("resamples must draw only complete registered days with replacement")
            bootstrap.append(aggregate(differences(draw)))
        if not bootstrap:
            raise ValueError("at least one bootstrap draw is required")
        bootstrap.sort()
        interval = (_quantile(bootstrap, analysis.alpha / 2), _quantile(bootstrap, 1 - analysis.alpha / 2))
    return PairedDelta(
        readout=analysis.readout, estimator=analysis.estimator, unit=analysis.unit,
        estimate=estimate, interval=interval, n_pairs=n_pairs,
        method="exhaustive_percentile_bootstrap" if analysis.pairing == "day_exhaustive" else "percentile_bootstrap",
        seed=analysis.seed, confidence_level=1 - analysis.alpha, train_repeat_estimates=repeat_estimates,
        reason="; ".join([*reasons, *paired.excluded]) or None,
    )


# --- Per-task results ---


def evaluate_pairs(
    task: EvaluationTask,
    scenario: ScenarioSpec,
    parent_fit: FitRecord,
    candidate_fit: FitRecord,
    observations: Iterable[UnitObservation],
    analyses: Iterable[AnalysisSpec],
    *,
    cost: Mapping[str, float] | None = None,
) -> PairedEvalResult:
    """Build a result for one task, preserving valid raw rows for repeat aggregation."""
    if (task.scenario_id, task.parent_fit_id, task.candidate_fit_id) != (
        scenario.scenario_id, parent_fit.fit_id, candidate_fit.fit_id
    ) or parent_fit.train_repeat_id != candidate_fit.train_repeat_id:
        raise ValueError("task does not match its scenario and paired fits")
    parent_id, candidate_id = parent_fit.candidate_id, candidate_fit.candidate_id
    if parent_id != scenario.parent_id or candidate_id not in scenario.candidate_ids:
        raise ValueError("task models do not match the scenario menu")
    rows = tuple(observations)
    units = set(task.unit_ids)
    for row in rows:
        if (row.scenario_id, row.train_repeat_id) != (scenario.scenario_id, parent_fit.train_repeat_id):
            raise ValueError("task received observations from another scenario or training repeat")
        if row.candidate_id not in (None, parent_id, candidate_id):
            raise ValueError("task received observations from an unrelated model")
    counts = Counter(_observation_key(row) for row in rows)
    contaminated = {
        (row.unit_id, row.arm, row.readout, row.window, row.unit)
        for row in rows if counts[_observation_key(row)] > 1
    }
    clean = tuple(
        row for row in rows if row.unit_id in units
        and (row.unit_id, row.arm, row.readout, row.window, row.unit) not in contaminated
    )
    paired_diffs = {}
    for analysis in analyses:
        if analysis.scenario_id != scenario.scenario_id or not has_readout(analysis, rows):
            continue
        paired_diffs[analysis.analysis_id] = estimate_delta(
            analysis, rows, parent_id=parent_id, candidate_id=candidate_id, required_unit_ids=task.unit_ids,
        )
    completed = bool(clean and paired_diffs)
    result = PairedEvalResult(
        result_id=f"{task.task_id}-result", task_id=task.task_id, split=task.split, scenario_id=task.scenario_id,
        candidate_id=candidate_id, parent_id=parent_id, parent_fit_id=parent_fit.fit_id,
        candidate_fit_id=candidate_fit.fit_id, action_ref=task.action_ref, random_inputs_ref=task.random_inputs_ref,
        arms=tuple(sorted({row.arm for row in clean}, key=lambda arm: arm.value)),
        status="completed" if completed else "degraded",
        cost={"train_seconds": 0.0, "eval_seconds": 0.0} if cost is None else cost,
        observations=clean, paired_diffs=paired_diffs,
        failure_reason=None if completed else "no eligible observations or registered paired readouts",
    )
    result.validate()
    return result
