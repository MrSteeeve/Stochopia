"""Serializable P0 records for the world-model and research-loop sides.

Fields without an explicit rule are required nonblank strings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from . import _contract_checks as checks
from ._serde import (
    _Record, boolean, enum, integer, json_value, literal, mapping, optional, real, record, sequence, text,
)


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


MANIFEST_SCHEMA = "openhyra-stochopia-manifest.v1"
FROZEN_MANIFEST_SCHEMA = "openhyra-stochopia-frozen-manifest.v1"
PREDICTION_SCHEMA = "openhyra-stochopia-prediction.v1"
PAIRED_EVAL_SCHEMA = "openhyra-stochopia-paired-eval.v1"

_optional_text = optional(text)
_optional_status = optional(enum(ReportStatus))
_texts = sequence(text)
_integers = mapping(integer)
_numbers = mapping(real)
_path = sequence(real)
_bounds = sequence(real, length=2)
_payload = mapping(json_value)
_split_name = literal("dev", "test")


# --- Scenario and development manifest declarations ---


@dataclass(frozen=True, kw_only=True)
class DataSplit(_Record):
    """Resolved canonical day IDs and the derived seeds of simulated days."""

    source: Mapping[str, Any]
    unit_ids: tuple[str, ...]
    unit_seeds: Mapping[str, int] = field(default_factory=dict)

    _field_rules = {
        "source": _payload,
        "unit_ids": _texts,
        "unit_seeds": _integers,
    }
    _check = checks.check_data_split


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

    _field_rules = {
        "environment_kind": literal("tier0_sim", "lob"),
        "environment": _payload,
        "action_plan": _payload,
        "candidate_ids": _texts,
        "matching_version": _optional_text,
    }
    _check = checks.check_scenario


@dataclass(frozen=True, kw_only=True)
class CandidateRef(_Record):
    candidate_id: str
    backbone: str
    recipe: str
    implementation: Mapping[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    channels: tuple[Literal["flow", "price"], ...] = ()

    _field_rules = {
        "implementation": _payload,
        "parent_id": _optional_text,
        "channels": sequence(literal("flow", "price")),
    }
    _check = checks.check_candidate


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

    _field_rules = {
        "kind": literal("evidence_llm", "metrics_llm", "no_llm"),
        "model": _optional_text,
        "model_version": _optional_text,
        "prompt_template_id": _optional_text,
        "llm_budget": _numbers,
        "allowed_evidence": _texts,
        "selection_rule": _optional_text,
    }
    _check = checks.check_group


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

    _field_rules = {
        "acceptance": _payload,
        "min_effect": real,
        "equivalence_band": _bounds,
        "min_pairs": integer,
        "min_train_repeats": integer,
        "alpha": real,
        "n_resamples": integer,
        "seed": integer,
    }
    _check = checks.check_analysis


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

    _field_rules = {
        "scenarios": sequence(record(ScenarioSpec)),
        "candidates": sequence(record(CandidateRef)),
        "groups": sequence(record(GroupSpec)),
        "splits": mapping(record(DataSplit)),
        "train_seeds": _integers,
        "train_plan": _payload,
        "seeds": _integers,
        "budget": mapping(_numbers),
        "analysis": sequence(record(AnalysisSpec)),
        "prediction_candidate_set": mapping(_texts),
    }
    _check = checks.check_experiment


# --- Selection, forecasts and search costs ---


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

    _field_rules = {
        "round_index": integer,
        "sequence": integer,
        "action": literal("select", "keep_parent", "abstain"),
        "candidate_id": _optional_text,
    }
    _check = checks.check_selection


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

    _field_rules = {
        "round_index": integer,
        "readout": _optional_text,
        "unit": _optional_text,
        "window": _optional_text,
        "direction": optional(enum(Direction)),
        "effect_interval": optional(_bounds),
        "conditions": _optional_text,
        "falsifier": _optional_text,
        "abstained": boolean,
        "abstain_reason": _optional_text,
    }
    _check = checks.check_prediction


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

    _field_rules = {
        "sequence": integer,
        "purpose": literal("search", "common_prediction", "unselected_check"),
        "charged_to_search": boolean,
        "budget_units": real,
        "cost": _numbers,
        "status": literal("pending", "completed", "failed", "unavailable"),
        "selection_id": _optional_text,
        "development_status": _optional_status,
    }
    _check = checks.check_trial


# --- Fit and paired measurements ---


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

    _field_rules = {
        "config": _payload,
        "seed": integer,
        "parent_id": _optional_text,
        "price_weights_ref": _optional_text,
        "background_weights_ref": _optional_text,
    }
    _check = checks.check_fit


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

    _field_rules = {
        "arm": enum(ArmKind),
        "candidate_id": _optional_text,
        "response_path": _path,
        "truth_path": _path,
        "values": _payload,
    }
    _check = checks.check_observation


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

    _field_rules = {
        "estimate": optional(real),
        "interval": optional(_bounds),
        "n_pairs": integer,
        "seed": integer,
        "confidence_level": real,
        "train_repeat_estimates": _numbers,
        "reason": _optional_text,
    }
    _check = checks.check_delta


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

    _field_rules = {
        "split": _split_name,
        "arms": sequence(enum(ArmKind)),
        "status": literal("completed", "failed", "degraded"),
        "cost": _numbers,
        "observations": sequence(record(UnitObservation)),
        "observations_ref": _optional_text,
        "paired_diffs": mapping(record(PairedDelta)),
        "failure_reason": _optional_text,
    }
    _check = checks.check_paired_result


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

    _field_rules = {
        "status": enum(ReportStatus),
        "result_id": _optional_text,
        "measured": optional(record(PairedDelta)),
        "direction_status": _optional_status,
        "magnitude_status": _optional_status,
        "equivalence_status": _optional_status,
        "interval_width": optional(real),
        "model_score_delta": optional(real),
    }
    _check = checks.check_verdict


# --- Sealed verification requests ---


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

    _field_rules = {
        "split": _split_name,
        "unit_ids": _texts,
    }
    _check = checks.check_task


@dataclass(frozen=True, kw_only=True)
class FinalChoice(_Record):
    group_id: str
    scenario_id: str
    candidate_id: str
    selection_id: str
    stopping_reason: str


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

    _field_rules = {
        "experiment": record(ExperimentManifest),
        "test_split": record(DataSplit),
        "seeds": _integers,
        "tasks": sequence(record(EvaluationTask)),
        "selections": sequence(record(SelectionRecord)),
        "predictions": sequence(record(StructuredPrediction)),
        "search_trials": sequence(record(SearchTrialRecord)),
        "final_choices": sequence(record(FinalChoice)),
        "fits": sequence(record(FitRecord)),
    }
    _check = checks.check_frozen
