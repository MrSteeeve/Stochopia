"""Serializable P0 records shared by the world-model and research-loop sides.

Validation covers payloads and references. Data access, simulation, statistics
and the verification lifecycle are handled by their respective executors.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import cache
from math import isfinite
from types import MappingProxyType, UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


class ReportStatus(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    ABSTAINED = "abstained"
    UNAVAILABLE = "unavailable"


class Direction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    UNCHANGED = "unchanged"


class ArmKind(str, Enum):
    FREE = "free"
    CLAMPED = "clamped"
    TRUTH = "truth"
    VISIBLE = "visible"
    BLIND = "blind"
    NO_INJECTION = "no_injection"


@cache
def _record_hints(record_type):
    return get_type_hints(record_type)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_value(value, path):
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path}: JSON object keys must be strings")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and isfinite(value):
        return value
    raise ValueError(f"{path}: expected a finite JSON value")


def _number(value):
    try:
        return type(value) in (int, float) and isfinite(value)
    except OverflowError:
        return False


def _field_value(value, annotation, path, *, wire):
    """Validate a typed field and decode its JSON representation when needed."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is Any:
        _json_value(value, path)
        return value
    if origin in (Union, UnionType):
        # The only unions in the contract are optional fields.
        if value is None:
            return None
        return _field_value(value, args[0], path, wire=wire)
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{path}: expected one of {args!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple) if wire else tuple):
            raise ValueError(f"{path}: expected an array")
        if args[-1] is Ellipsis:
            item_types = (args[0],) * len(value)
        elif len(value) == len(args):
            item_types = args
        else:
            raise ValueError(f"{path}: expected {len(args)} items")
        return tuple(
            _field_value(item, item_type, f"{path}[{i}]", wire=wire)
            for i, (item, item_type) in enumerate(zip(value, item_types))
        )
    if origin is Mapping:
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: expected an object")
        return {
            _field_value(key, args[0], f"{path}.key", wire=wire):
            _field_value(item, args[1], f"{path}.{key}", wire=wire)
            for key, item in value.items()
        }
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if wire and isinstance(value, str):
            try:
                return annotation(value)
            except ValueError:
                pass
        if isinstance(value, annotation):
            return value
        raise ValueError(f"{path}: expected {annotation.__name__}")
    if isinstance(annotation, type) and issubclass(annotation, _Record):
        if wire:
            return annotation.from_dict(value)
        if not isinstance(value, annotation):
            raise ValueError(f"{path}: expected {annotation.__name__}")
        value.validate()
        return value
    if annotation is str:
        valid = type(value) is str and bool(value.strip())
    elif annotation is float:
        valid = _number(value)
    else:
        valid = type(value) is annotation
    if not valid:
        raise ValueError(f"{path}: expected {annotation.__name__}")
    return value


def _wire(value):
    if isinstance(value, _Record):
        return {item.name: _wire(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    return value


class _Record:
    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (Mapping, list, tuple)):
                object.__setattr__(self, item.name, _freeze(value))

    def validate(self) -> None:
        hints = _record_hints(type(self))
        for item in fields(self):
            _field_value(getattr(self, item.name), hints[item.name], item.name, wire=False)
        self._check()

    def _check(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _wire(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{cls.__name__}: expected an object")
        names = {item.name for item in fields(cls)}
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown fields {sorted(map(str, unknown))}")
        hints = _record_hints(cls)
        decoded = {
            name: _field_value(value, hints[name], name, wire=True)
            for name, value in raw.items()
        }
        try:
            record = cls(**decoded)
        except TypeError as exc:
            raise ValueError(f"{cls.__name__}: {exc}") from exc
        record.validate()
        return record


def _unique(items, name):
    if len(set(items)) != len(items):
        raise ValueError(f"{name}: expected unique values")


def _index(records, field_name):
    _unique([getattr(record, field_name) for record in records], field_name)
    return {getattr(record, field_name): record for record in records}


def _nonnegative(values, name):
    if any(value < 0 for value in values):
        raise ValueError(f"{name}: expected nonnegative values")


def _interval(bounds, name):
    if bounds[0] > bounds[1]:
        raise ValueError(f"{name}: lower bound exceeds upper bound")


def _schema(actual, expected):
    if actual != expected:
        raise ValueError(f"schema: expected {expected}")


@dataclass(frozen=True, kw_only=True)
class DataSplit(_Record):
    """Resolved canonical day IDs and the derived seeds of simulated days."""

    source: Mapping[str, Any]
    unit_ids: tuple[str, ...]
    unit_seeds: Mapping[str, int] = field(default_factory=dict)

    def _check(self):
        if not self.source or not self.unit_ids:
            raise ValueError("data split requires a source and at least one unit")
        _unique(self.unit_ids, "unit_ids")
        if not self.unit_seeds.keys() <= set(self.unit_ids):
            raise ValueError("unit_seeds: unknown unit ID")
        seeds = tuple(self.unit_seeds.values())
        _nonnegative(seeds, "unit_seeds")
        _unique(seeds, "unit_seeds")


def _disjoint_splits(splits):
    seen_ids = set()
    seen_seeds = set()
    for name, split in splits.items():
        if seen_ids.intersection(split.unit_ids):
            raise ValueError(f"splits.{name}: overlapping unit IDs")
        if seen_seeds.intersection(split.unit_seeds.values()):
            raise ValueError(f"splits.{name}: overlapping simulated seeds")
        seen_ids.update(split.unit_ids)
        seen_seeds.update(split.unit_seeds.values())


def _require_simulated_seeds(scenarios, splits):
    if all(scene.environment_kind == "tier0_sim" for scene in scenarios):
        for name, split in splits.items():
            if split.unit_seeds.keys() != set(split.unit_ids):
                raise ValueError(f"splits.{name}: simulated days require resolved seeds")


@dataclass(frozen=True, kw_only=True)
class ScenarioSpec(_Record):
    scenario_id: str
    diagnostic_question: str
    environment_kind: Literal["tier0_sim", "lob"]
    environment: Mapping[str, Any]
    action_plan: Mapping[str, Any]
    window: str
    parent_id: str
    candidate_ids: tuple[str, ...]
    matching_version: str | None = None

    def _check(self):
        _unique(self.candidate_ids, "candidate_ids")
        if self.parent_id not in self.candidate_ids:
            raise ValueError("parent_id: baseline must appear in the candidate menu")
        if not self.environment or not self.action_plan:
            raise ValueError("scenario requires environment and action_plan")
        if self.environment_kind == "lob" and self.matching_version is None:
            raise ValueError("lob scenario requires matching_version")


@dataclass(frozen=True, kw_only=True)
class CandidateRef(_Record):
    candidate_id: str
    backbone: str
    recipe: str
    implementation: Mapping[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    channels: tuple[Literal["flow", "price"], ...] = ()

    def _check(self):
        _unique(self.channels, "channels")
        if self.parent_id == self.candidate_id:
            raise ValueError("parent_id cannot reference the candidate itself")
        if self.channels and self.parent_id is None:
            raise ValueError("a candidate with changes requires parent_id")


@dataclass(frozen=True, kw_only=True)
class GroupSpec(_Record):
    group_id: str
    kind: Literal["evidence_llm", "metrics_llm", "no_llm"]
    model: str | None = None
    model_version: str | None = None
    prompt_template_id: str | None = None
    llm_budget: Mapping[str, float] = field(default_factory=dict)
    allowed_evidence: tuple[str, ...] = ()
    selection_rule: str | None = None

    def _check(self):
        _unique(self.allowed_evidence, "allowed_evidence")
        _nonnegative(self.llm_budget.values(), "llm_budget")
        model_settings = (self.model, self.model_version, self.prompt_template_id)
        if self.kind == "no_llm":
            if self.selection_rule is None or any(model_settings) or self.llm_budget:
                raise ValueError("no_llm requires a selection rule and no LLM settings")
        elif not all(model_settings) or not self.llm_budget or self.selection_rule is not None:
            raise ValueError("LLM groups require model, version, prompt template and budget")
        if bool(self.allowed_evidence) != (self.kind == "evidence_llm"):
            raise ValueError("only evidence_llm receives additional allowed evidence")


@dataclass(frozen=True, kw_only=True)
class AnalysisSpec(_Record):
    analysis_id: str
    scenario_id: str
    readout: str
    window: str
    unit: str
    estimator: str
    acceptance: Mapping[str, Any]
    min_effect: float
    equivalence_band: tuple[float, float]
    min_pairs: int
    min_train_repeats: int
    pairing: str
    alpha: float
    n_resamples: int
    seed: int

    def _check(self):
        if not self.acceptance:
            raise ValueError("acceptance must register the decision thresholds")
        if self.min_effect <= 0:
            raise ValueError("min_effect must be positive")
        low, high = self.equivalence_band
        if not low < 0 < high:
            raise ValueError("equivalence_band must straddle zero")
        if self.min_pairs < 2 or self.min_train_repeats < 1 or self.n_resamples < 1:
            raise ValueError("analysis requires at least two pairs and positive repeat counts")
        if not 0 < self.alpha < 1 or self.seed < 0:
            raise ValueError("analysis requires 0 < alpha < 1 and a nonnegative seed")


MANIFEST_SCHEMA = "openhyra-stochopia-manifest.v1"
FROZEN_MANIFEST_SCHEMA = "openhyra-stochopia-frozen-manifest.v1"
PREDICTION_SCHEMA = "openhyra-stochopia-prediction.v1"
PAIRED_EVAL_SCHEMA = "openhyra-stochopia-paired-eval.v1"


@dataclass(frozen=True, kw_only=True)
class ExperimentManifest(_Record):
    """Development view. No test source, test unit IDs or test seeds."""

    exp_id: str
    scenarios: tuple[ScenarioSpec, ...]
    candidates: tuple[CandidateRef, ...]
    groups: tuple[GroupSpec, ...]
    splits: Mapping[str, DataSplit]
    train_seeds: Mapping[str, int]
    train_plan: Mapping[str, Any]
    measurement_version: str
    fit_version: str
    seeds: Mapping[str, int]
    budget: Mapping[str, Mapping[str, float]]
    analysis: tuple[AnalysisSpec, ...]
    prediction_candidate_set: Mapping[str, tuple[str, ...]]
    unselected_sample_rule: str
    schema: str = MANIFEST_SCHEMA

    def _check(self):
        _schema(self.schema, MANIFEST_SCHEMA)
        if self.splits.keys() != {"train", "dev"}:
            raise ValueError("development splits must contain exactly train and dev")
        _disjoint_splits(self.splits)
        if not self.train_seeds or not self.train_plan:
            raise ValueError("training requires repeat seeds and a plan")
        _nonnegative(self.train_seeds.values(), "train_seeds")
        _unique(tuple(self.train_seeds.values()), "train_seeds")
        if self.seeds.keys() != {"rollout", "delta_ci"}:
            raise ValueError("development seeds must contain rollout and delta_ci")
        _nonnegative(self.seeds.values(), "seeds")
        if self.budget.keys() != {"search", "final_check"}:
            raise ValueError("budget must separate search and final_check")
        for name, budget in self.budget.items():
            if not budget:
                raise ValueError(f"budget.{name}: expected registered cost units")
            _nonnegative(budget.values(), f"budget.{name}")

        groups = _index(self.groups, "group_id")
        by_kind = _index(tuple(groups.values()), "kind")
        if by_kind.keys() != {"evidence_llm", "metrics_llm", "no_llm"}:
            raise ValueError("exactly one group of each P0 kind is required")
        evidence, metrics = by_kind["evidence_llm"], by_kind["metrics_llm"]
        for name in ("model", "model_version", "prompt_template_id", "llm_budget"):
            if getattr(evidence, name) != getattr(metrics, name):
                raise ValueError(f"LLM groups must match {name}")

        scenarios = _index(self.scenarios, "scenario_id")
        candidates = _index(self.candidates, "candidate_id")
        analysis = _index(self.analysis, "analysis_id")
        if not scenarios or self.prediction_candidate_set.keys() != scenarios.keys():
            raise ValueError("common prediction sets must match the registered scenarios")
        _require_simulated_seeds(self.scenarios, self.splits)
        if {item.scenario_id for item in analysis.values()} != set(scenarios):
            raise ValueError("analysis must cover every scenario")
        for candidate in self.candidates:
            if candidate.parent_id is not None and candidate.parent_id not in candidates:
                raise ValueError(f"{candidate.candidate_id}: unknown parent_id")
        for scenario in self.scenarios:
            menu = set(scenario.candidate_ids)
            if not menu <= candidates.keys():
                raise ValueError(f"{scenario.scenario_id}: unknown candidate in menu")
            changes = menu - {scenario.parent_id}
            channels = {channel for key in changes for channel in candidates[key].channels}
            if channels != {"flow", "price"}:
                raise ValueError(f"{scenario.scenario_id}: menu must cover flow and price")
            common = self.prediction_candidate_set[scenario.scenario_id]
            _unique(common, "prediction_candidate_set")
            if not common or not set(common) <= changes:
                raise ValueError("common prediction set must contain non-baseline menu candidates")

@dataclass(frozen=True, kw_only=True)
class SelectionRecord(_Record):
    selection_id: str
    exp_id: str
    round_index: int
    group_id: str
    scenario_id: str
    sequence: int
    parent_id: str
    dev_snapshot_id: str
    menu_id: str
    action: Literal["select", "keep_parent", "abstain"]
    reason: str
    candidate_id: str | None = None

    def _check(self):
        _nonnegative((self.round_index, self.sequence), "round and sequence")
        if self.action == "select":
            if self.candidate_id in (None, self.parent_id):
                raise ValueError("select requires a non-baseline candidate_id")
        elif self.candidate_id is not None:
            raise ValueError("only select takes a candidate_id")


@dataclass(frozen=True, kw_only=True)
class StructuredPrediction(_Record):
    prediction_id: str
    exp_id: str
    round_index: int
    group_id: str
    scenario_id: str
    explanation_id: str
    candidate_id: str
    parent_id: str
    analysis_id: str
    dev_snapshot_id: str
    readout: str | None = None
    unit: str | None = None
    window: str | None = None
    direction: Direction | None = None
    effect_interval: tuple[float, float] | None = None
    conditions: str | None = None
    falsifier: str | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    schema: str = PREDICTION_SCHEMA

    def _check(self):
        _schema(self.schema, PREDICTION_SCHEMA)
        _nonnegative((self.round_index,), "round_index")
        if self.candidate_id == self.parent_id:
            raise ValueError("prediction requires a non-baseline candidate_id")
        if self.abstained:
            if self.abstain_reason is None:
                raise ValueError("abstain_reason is required for an abstention")
            if self.direction is not None or self.effect_interval is not None:
                raise ValueError("abstention cannot include a directional forecast")
            return
        if self.abstain_reason is not None:
            raise ValueError("abstain_reason requires abstained=True")
        for name in ("readout", "unit", "window", "direction", "conditions", "falsifier"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required for a prediction")
        if self.direction is Direction.UNCHANGED:
            if self.effect_interval is not None:
                raise ValueError("unchanged predictions use the analysis equivalence band")
            return
        if self.effect_interval is None:
            raise ValueError("directional predictions require an effect_interval")
        _interval(self.effect_interval, "effect_interval")
        low, high = self.effect_interval
        if (self.direction is Direction.INCREASE and low <= 0
                or self.direction is Direction.DECREASE and high >= 0):
            raise ValueError("effect_interval must agree with the prediction direction")


@dataclass(frozen=True, kw_only=True)
class SearchTrialRecord(_Record):
    trial_id: str
    exp_id: str
    group_id: str
    scenario_id: str
    sequence: int
    candidate_id: str
    purpose: Literal["search", "common_prediction", "unselected_check"]
    charged_to_search: bool
    budget_units: float
    cost: Mapping[str, float]
    status: Literal["pending", "completed", "failed", "unavailable"]
    selection_id: str | None = None
    development_status: ReportStatus | None = None

    def _check(self):
        _nonnegative((self.sequence, self.budget_units), "trial sequence and budget")
        _nonnegative(self.cost.values(), "cost")
        if self.purpose == "search" and (self.selection_id is None or not self.charged_to_search):
            raise ValueError("search trial requires a selection_id and must be charged to search")
        if self.status == "completed" and self.development_status not in (
            ReportStatus.SUPPORTED, ReportStatus.REFUTED, ReportStatus.INCONCLUSIVE
        ):
            raise ValueError("completed trial requires a development assessment")
        if self.status in ("pending", "failed", "unavailable") and self.development_status not in (
            None, ReportStatus.UNAVAILABLE
        ):
            raise ValueError("trial without a completed measurement cannot claim an assessment")


@dataclass(frozen=True, kw_only=True)
class FitRecord(_Record):
    fit_id: str
    candidate_id: str
    train_repeat_id: str
    train_data_ref: str
    config: Mapping[str, Any]
    seed: int
    model_ref: str
    parent_id: str | None = None
    price_weights_ref: str | None = None
    background_weights_ref: str | None = None

    def _check(self):
        _nonnegative((self.seed,), "seed")
        if not self.config:
            raise ValueError("config is required to replay a fit")
        if self.parent_id == self.candidate_id:
            raise ValueError("parent_id cannot reference the candidate itself")


@dataclass(frozen=True, kw_only=True)
class UnitObservation(_Record):
    unit_id: str
    scenario_id: str
    train_repeat_id: str
    arm: ArmKind
    window: str
    readout: str
    unit: str
    candidate_id: str | None = None
    response_path: tuple[float, ...] = ()
    truth_path: tuple[float, ...] = ()
    values: Mapping[str, Any] = field(default_factory=dict)

    def _check(self):
        if self.arm is not ArmKind.TRUTH and self.candidate_id is None:
            raise ValueError("model-side observation requires candidate_id")
        if not (self.response_path or self.truth_path or self.values):
            raise ValueError("observation requires a path or raw values")
        if self.response_path and self.truth_path and len(self.response_path) != len(self.truth_path):
            raise ValueError("response_path and truth_path must share the event window")


@dataclass(frozen=True, kw_only=True)
class PairedDelta(_Record):
    readout: str
    estimator: str
    unit: str
    estimate: float | None
    interval: tuple[float, float] | None
    n_pairs: int
    method: str
    seed: int
    confidence_level: float
    train_repeat_estimates: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None

    def _check(self):
        _nonnegative((self.n_pairs, self.seed), "n_pairs and seed")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if (self.estimate is None) != (self.n_pairs == 0):
            raise ValueError("a point estimate requires at least one pair")
        if self.interval is None:
            if self.reason is None:
                raise ValueError("missing interval requires a reason")
        else:
            _interval(self.interval, "interval")
            if self.n_pairs < 2:
                raise ValueError("interval requires at least two pairs")


@dataclass(frozen=True, kw_only=True)
class PairedEvalResult(_Record):
    result_id: str
    task_id: str
    split: Literal["dev", "test"]
    scenario_id: str
    candidate_id: str
    parent_id: str
    parent_fit_id: str
    candidate_fit_id: str
    action_ref: str
    random_inputs_ref: str
    arms: tuple[ArmKind, ...]
    status: Literal["completed", "failed", "degraded"]
    cost: Mapping[str, float]
    observations: tuple[UnitObservation, ...] = ()
    observations_ref: str | None = None
    paired_diffs: Mapping[str, PairedDelta] = field(default_factory=dict)
    failure_reason: str | None = None
    schema: str = PAIRED_EVAL_SCHEMA

    def _check(self):
        _schema(self.schema, PAIRED_EVAL_SCHEMA)
        if self.candidate_id == self.parent_id or self.parent_fit_id == self.candidate_fit_id:
            raise ValueError("paired result requires distinct candidate and baseline fits")
        _unique(self.arms, "arms")
        _nonnegative(self.cost.values(), "cost")
        if self.observations and self.observations_ref is not None:
            raise ValueError("provide inline observations or an observation reference")
        if self.status == "completed":
            if not self.arms or not self.paired_diffs or not (self.observations or self.observations_ref):
                raise ValueError("completed result requires arms, day observations and paired deltas")
            if not {"train_seconds", "eval_seconds"} <= self.cost.keys():
                raise ValueError("completed cost must include training and evaluation time")
            if self.failure_reason is not None:
                raise ValueError("completed result cannot carry a failure_reason")
        elif self.failure_reason is None:
            raise ValueError("failed or degraded result requires a failure_reason")
        keys = []
        model_keys = {self.candidate_id: set(), self.parent_id: set()}
        for observation in self.observations:
            if (observation.scenario_id != self.scenario_id or observation.arm not in self.arms
                    or observation.candidate_id not in (None, self.candidate_id, self.parent_id)):
                raise ValueError("observation does not belong to the result's scene, arms or models")
            key = (observation.unit_id, observation.train_repeat_id, observation.arm,
                   observation.window, observation.readout, observation.unit)
            keys.append((observation.candidate_id, key))
            if observation.candidate_id is not None and observation.arm is not ArmKind.TRUTH:
                model_keys[observation.candidate_id].add(key)
        _unique(keys, "day observation keys")
        if self.observations and any(delta.n_pairs for delta in self.paired_diffs.values()):
            if not model_keys[self.candidate_id] & model_keys[self.parent_id]:
                raise ValueError("reported paired deltas require aligned parent and candidate observations")


@dataclass(frozen=True, kw_only=True)
class PredictionVerdict(_Record):
    prediction_id: str
    status: ReportStatus
    reason: str
    result_id: str | None = None
    measured: PairedDelta | None = None
    direction_status: ReportStatus | None = None
    magnitude_status: ReportStatus | None = None
    equivalence_status: ReportStatus | None = None
    interval_width: float | None = None
    model_score_delta: float | None = None

    def _check(self):
        if self.measured is not None and self.result_id is None:
            raise ValueError("measured prediction requires result_id")
        if self.status in (ReportStatus.SUPPORTED, ReportStatus.REFUTED):
            if self.measured is None or self.measured.interval is None:
                raise ValueError("a decided prediction requires a measured interval")
        if self.interval_width is not None:
            _nonnegative((self.interval_width,), "interval_width")
            if self.measured is None or self.measured.interval is None:
                raise ValueError("interval_width requires a measured interval")
        if self.equivalence_status is not None and (
            self.direction_status is not None or self.magnitude_status is not None
        ):
            raise ValueError("equivalence and directional assessments are mutually exclusive")
        parts = (self.direction_status, self.magnitude_status, self.equivalence_status)
        if self.status in (ReportStatus.ABSTAINED, ReportStatus.UNAVAILABLE):
            if any(part is not None for part in parts):
                raise ValueError("an unavailable or abstained prediction has no scored components")
            if self.status is ReportStatus.ABSTAINED and self.measured is not None:
                raise ValueError("an abstained prediction has no measured effect")
            return
        if self.measured is None:
            raise ValueError("assessed prediction requires a measured effect")
        components = ((self.equivalence_status,) if self.equivalence_status is not None
                      else (self.direction_status, self.magnitude_status))
        allowed = (ReportStatus.SUPPORTED, ReportStatus.REFUTED, ReportStatus.INCONCLUSIVE)
        if any(component not in allowed for component in components):
            raise ValueError("assessment requires equivalence or both direction and magnitude statuses")
        if self.measured.interval is None and any(
            component is not ReportStatus.INCONCLUSIVE for component in components
        ):
            raise ValueError("without an interval, assessment components must be inconclusive")
        if ReportStatus.REFUTED in components:
            expected = ReportStatus.REFUTED
        elif ReportStatus.INCONCLUSIVE in components:
            expected = ReportStatus.INCONCLUSIVE
        else:
            expected = ReportStatus.SUPPORTED
        if self.status is not expected:
            raise ValueError("overall verdict disagrees with its scored components")


@dataclass(frozen=True, kw_only=True)
class EvaluationTask(_Record):
    task_id: str
    split: Literal["dev", "test"]
    scenario_id: str
    parent_fit_id: str
    candidate_fit_id: str
    unit_ids: tuple[str, ...]
    action_ref: str
    random_inputs_ref: str

    def _check(self):
        if self.parent_fit_id == self.candidate_fit_id or not self.unit_ids:
            raise ValueError("evaluation task requires two fits and at least one unit")
        _unique(self.unit_ids, "unit_ids")


@dataclass(frozen=True, kw_only=True)
class FinalChoice(_Record):
    group_id: str
    scenario_id: str
    candidate_id: str
    selection_id: str
    stopping_reason: str


class _FrozenReferences:
    """Index ids once for referential checks across the sealed payload."""

    def __init__(self, frozen):
        self.scenarios = _index(frozen.experiment.scenarios, "scenario_id")
        self.candidates = _index(frozen.experiment.candidates, "candidate_id")
        self.groups = _index(frozen.experiment.groups, "group_id")
        self.analysis = _index(frozen.experiment.analysis, "analysis_id")
        self.selections = _index(frozen.selections, "selection_id")
        self.fits = _index(frozen.fits, "fit_id")

    def subject(self, record, exp_id):
        if record.exp_id != exp_id:
            raise ValueError("record exp_id does not match frozen experiment")
        if record.group_id not in self.groups or record.scenario_id not in self.scenarios:
            raise ValueError("record references an unknown group or scenario")
        return self.scenarios[record.scenario_id]

    def selection(self, selection_id, group_id, scenario_id):
        selection = self.selections.get(selection_id)
        if selection is None or (group_id, scenario_id) != (selection.group_id, selection.scenario_id):
            raise ValueError("selection reference does not match the group and scenario")
        return selection


@dataclass(frozen=True, kw_only=True)
class FrozenManifest(_Record):
    """Complete sealed input for the Verifier; never returned to the dev loop."""

    exp_id: str
    batch_id: str
    experiment: ExperimentManifest
    test_split: DataSplit
    seeds: Mapping[str, int]
    tasks: tuple[EvaluationTask, ...]
    selections: tuple[SelectionRecord, ...]
    predictions: tuple[StructuredPrediction, ...]
    search_trials: tuple[SearchTrialRecord, ...]
    final_choices: tuple[FinalChoice, ...]
    fits: tuple[FitRecord, ...]
    schema: str = FROZEN_MANIFEST_SCHEMA

    def _check(self):
        _schema(self.schema, FROZEN_MANIFEST_SCHEMA)
        if self.exp_id != self.experiment.exp_id:
            raise ValueError("exp_id does not match the sealed development manifest")
        splits = {**self.experiment.splits, "test": self.test_split}
        _disjoint_splits(splits)
        _require_simulated_seeds(self.experiment.scenarios, splits)
        if self.seeds.keys() != {"rollout", "delta_ci"}:
            raise ValueError("test seeds must contain rollout and delta_ci")
        _nonnegative(self.seeds.values(), "seeds")
        refs = _FrozenReferences(self)
        first_choices = self._check_choices(refs)
        self._check_predictions(refs, first_choices)
        self._check_trials(refs)
        self._check_tasks(refs, first_choices)

    def _check_choices(self, refs):
        sequence_keys = []
        for record in self.selections:
            scene = refs.subject(record, self.exp_id)
            if record.parent_id != scene.parent_id or (
                record.candidate_id is not None and record.candidate_id not in scene.candidate_ids
            ):
                raise ValueError("selection baseline or candidate does not match the scenario")
            sequence_keys.append((record.group_id, record.scenario_id, record.sequence))
        _unique(sequence_keys, "selection sequence")
        expected = {(group, scene) for group in refs.groups for scene in refs.scenarios}
        choices = [(choice.group_id, choice.scenario_id) for choice in self.final_choices]
        _unique(choices, "final choices")
        if set(choices) != expected:
            raise ValueError("final choices must cover every group and scenario")
        for choice in self.final_choices:
            selection = refs.selection(choice.selection_id, choice.group_id, choice.scenario_id)
            if choice.candidate_id != (selection.candidate_id or selection.parent_id):
                raise ValueError("final choice must follow the recorded selection")
        first_choices = {}
        for row in sorted(self.selections, key=lambda selection: selection.sequence):
            first_choices.setdefault((row.group_id, row.scenario_id), row)
        scene_snapshots = {}
        for row in first_choices.values():
            basis = (row.round_index, row.dev_snapshot_id)
            if scene_snapshots.setdefault(row.scenario_id, basis) != basis:
                raise ValueError("initial choices must share a development snapshot within a scenario")
        return first_choices

    def _check_predictions(self, refs, first_choices):
        _index(self.predictions, "prediction_id")
        covered = {}
        for prediction in self.predictions:
            scene = refs.subject(prediction, self.exp_id)
            group = refs.groups[prediction.group_id]
            if group.kind == "no_llm":
                raise ValueError("no_llm cannot submit predictions")
            analysis = refs.analysis.get(prediction.analysis_id)
            if analysis is None or analysis.scenario_id != scene.scenario_id:
                raise ValueError("prediction references unknown or unrelated analysis_id")
            if prediction.parent_id != scene.parent_id or prediction.candidate_id not in scene.candidate_ids:
                raise ValueError("prediction baseline or candidate does not match the scenario")
            first_choice = first_choices[(prediction.group_id, prediction.scenario_id)]
            if prediction.round_index < first_choice.round_index or (
                prediction.round_index == first_choice.round_index
                and prediction.dev_snapshot_id != first_choice.dev_snapshot_id
            ):
                raise ValueError("first-round prediction must match the registered development snapshot")
            if not prediction.abstained and (prediction.readout, prediction.window, prediction.unit) != (
                analysis.readout, analysis.window, analysis.unit
            ):
                raise ValueError("prediction readout, window and unit must match the analysis")
            key = (prediction.group_id, prediction.scenario_id, prediction.candidate_id)
            covered.setdefault(key, set()).add(prediction.analysis_id)

        llm_groups = [group.group_id for group in refs.groups.values() if group.kind != "no_llm"]
        for scene, common in self.experiment.prediction_candidate_set.items():
            for candidate in common:
                left, right = (covered.get((group, scene, candidate), set()) for group in llm_groups)
                if not left & right:
                    raise ValueError("LLM groups must cover the common candidates on a shared analysis")

    def _check_trials(self, refs):
        _index(self.search_trials, "trial_id")
        sequence_keys = []
        for trial in self.search_trials:
            scene = refs.subject(trial, self.exp_id)
            if trial.status == "pending" or trial.candidate_id not in scene.candidate_ids:
                raise ValueError("frozen trial must be terminal and refer to a menu candidate")
            if trial.selection_id is not None:
                selection = refs.selection(trial.selection_id, trial.group_id, trial.scenario_id)
                if trial.purpose == "search" and selection.candidate_id != trial.candidate_id:
                    raise ValueError("search trial did not run its selected candidate")
            sequence_keys.append((trial.group_id, trial.scenario_id, trial.sequence))
        _unique(sequence_keys, "trial sequence")

    def _check_tasks(self, refs, first_choices):
        for fit in self.fits:
            candidate = refs.candidates.get(fit.candidate_id)
            if candidate is None or candidate.parent_id != fit.parent_id:
                raise ValueError("fit candidate or lineage does not match the manifest")
            if self.experiment.train_seeds.get(fit.train_repeat_id) != fit.seed:
                raise ValueError("fit seed does not match the registered training repeat")
        _index(self.tasks, "task_id")
        test_units = set(self.test_split.unit_ids)
        covered = set()
        for task in self.tasks:
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

        required = {(row.scenario_id, row.candidate_id) for row in self.predictions if not row.abstained}
        required.update((row.scenario_id, row.candidate_id) for row in first_choices.values()
                        if row.candidate_id is not None)
        required.update(
            (row.scenario_id, row.candidate_id) for row in self.final_choices
            if row.candidate_id != refs.scenarios[row.scenario_id].parent_id
        )
        if not required <= covered:
            raise ValueError("frozen tasks must cover first choices, final choices and common predictions")
