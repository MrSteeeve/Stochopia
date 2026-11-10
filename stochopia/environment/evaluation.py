"""Versioned v3 task suites, economic replay, and benchmark aggregation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..common import atomic_write_json, canonical_json, sha256_json
from ..products import ProductSpec
from ..stats import cluster_bootstrap_ci, derive_seed
from .agent_runner import parse_environment_action
from .core import StochopiaStructurerEnv
from .trajectory import TRAJECTORY_FORMAT, to_jsonable
from .types import (
    EpisodeTask,
    InvalidAction,
    RequestQuote,
    Skip,
    SubmitDesign,
    SubmitProduct,
)


TASK_SUITE_FORMAT = "stochopia.task-suite.v1"
EVALUATION_FORMAT = "stochopia.trajectory-evaluation.v1"
AGGREGATE_FORMAT = "stochopia.evaluation-aggregate.v1"
# Every DSL environment version; the exact one is checked on --replay.
_DSL_VERSION_PREFIX = "v3-spine-dsl-experimental."


def _canonical_json(value: Any) -> str:
    return canonical_json(to_jsonable(value))


def _digest(value: Any) -> str:
    return sha256_json(to_jsonable(value))


def _atomic_write(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    return atomic_write_json(destination, payload)


@dataclass(frozen=True)
class TaskSuite:
    """A self-contained, split-labelled collection of frozen v3 tasks."""

    name: str
    version: str
    split: str
    tasks: tuple[EpisodeTask, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    format: str = TASK_SUITE_FORMAT

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        if self.format != TASK_SUITE_FORMAT:
            raise ValueError(f"unsupported task-suite format: {self.format!r}")
        for name in ("name", "version", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"TaskSuite.{name} must be non-empty")
        if self.split not in {"train", "dev", "test", "private_test"}:
            raise ValueError(
                "TaskSuite.split must be train, dev, test, or private_test"
            )
        if not self.tasks:
            raise ValueError("TaskSuite.tasks must not be empty")
        hashes = [task.task_hash for task in self.tasks]
        if len(set(hashes)) != len(hashes):
            raise ValueError("TaskSuite contains duplicate task manifests")
        # Fail immediately if metadata is not portable JSON.
        _canonical_json(dict(self.metadata))

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "name": self.name,
            "version": self.version,
            "split": self.split,
            "metadata": to_jsonable(dict(self.metadata)),
            "task_count": len(self.tasks),
            "tasks": [task.manifest for task in self.tasks],
            "task_hashes": [task.task_hash for task in self.tasks],
        }

    @property
    def suite_hash(self) -> str:
        return _digest(self._payload_without_hash())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["suite_hash"] = self.suite_hash
        return payload

    def save(self, path: str | Path) -> Path:
        return _atomic_write(path, self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskSuite":
        raw = dict(payload)
        raw.pop("suite_hash", None)
        tasks_payload = raw.get("tasks")
        if not isinstance(tasks_payload, list):
            raise ValueError("task suite tasks must be a list")
        suite = cls(
            name=raw.get("name", ""),
            version=raw.get("version", ""),
            split=raw.get("split", ""),
            tasks=tuple(EpisodeTask.from_manifest(item) for item in tasks_payload),
            metadata=raw.get("metadata") or {},
            format=raw.get("format", ""),
        )
        return suite

    @classmethod
    def load(cls, path: str | Path) -> "TaskSuite":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load task suite: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("task suite root must be an object")
        return cls.from_dict(payload)


def _action_from_transition(payload: Mapping[str, Any]):
    action_name = payload.get("action")
    if action_name == "request_dsl_quote":
        from ..adapters.structurer_dsl import DSLContract, DSLProductSpec
        from .dsl import RequestDSLQuote

        contract = DSLContract(**dict(payload["product"]["contract"]))
        return RequestDSLQuote(DSLProductSpec.from_contract(contract))
    if action_name == "invalid":
        return InvalidAction(
            reason=str(payload.get("reason", "invalid action")),
            raw=payload.get("raw"),
        )
    if action_name == "request_quote":
        return RequestQuote(ProductSpec(**dict(payload["product"])))
    if action_name == "submit_product":
        return SubmitProduct(
            ProductSpec(**dict(payload["product"])),
            str(payload.get("explanation", "")),
        )
    if action_name == "submit_design":
        return SubmitDesign(
            str(payload["quote_id"]),
            str(payload.get("explanation", "")),
        )
    if action_name == "skip":
        return Skip(str(payload.get("reason", "")))
    return parse_environment_action(payload)


def _numeric_total(summary: Mapping[str, Any], name: str) -> float:
    normalized = summary.get("normalized_totals")
    if isinstance(normalized, Mapping):
        value = normalized.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0


def _metrics_from_transitions(
    transitions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float | None], Mapping[str, Any], Mapping[str, Any]]:
    if not transitions:
        return {}, {}, {}
    final_info = transitions[-1].get("info") or {}
    episode_summary = final_info.get("episode_summary") or {}
    reward_summary = episode_summary.get("reward") or {}
    constraint_summary = episode_summary.get("constraints") or {}

    submissions = 0
    hard_passes = 0
    contract_eligible = 0
    contract_passes = 0
    lifecycle_closures = 0
    for transition in transitions:
        constraints = transition.get("constraint_signals") or {}
        is_submission = constraints.get("accepted") is not None
        if is_submission:
            submissions += 1
            if constraints.get("hard_pass") is True:
                hard_passes += 1
                if constraints.get("client_contract_pass") is not None:
                    contract_eligible += 1
                if constraints.get("client_contract_pass") is True:
                    contract_passes += 1
        tool_result = transition.get("tool_result") or {}
        lifecycle_closures += sum(
            1
            for event in tool_result.get("lifecycle_events") or ()
            if isinstance(event, Mapping)
        )

    steps = int(constraint_summary.get("steps", len(transitions)))
    invalid = int(constraint_summary.get("invalid_actions", 0))
    accepted = int(constraint_summary.get("accepted_submissions", 0))
    final_transition = transitions[-1]
    metrics = {
        "completion_rate": float(bool(final_transition.get("terminated"))),
        "truncation_rate": float(bool(final_transition.get("truncated"))),
        "rollout_end_rate": float(
            bool(
                final_transition.get("terminated")
                or final_transition.get("truncated")
            )
        ),
        "steps": float(steps),
        "invalid_action_rate": invalid / steps if steps else 0.0,
        "submission_rate": submissions / steps if steps else 0.0,
        # Conditional rates are undefined (None) on an empty denominator;
        # aggregation skips non-numeric values instead of pooling a fake 0.
        "hard_execution_rate_given_submission": (
            hard_passes / submissions if submissions else None
        ),
        "contract_acceptance_rate_given_hard_pass": (
            contract_passes / contract_eligible if contract_eligible else None
        ),
        "settlement_acceptance_rate": (
            accepted / submissions if submissions else None
        ),
        "accepted_submissions": float(accepted),
        "lifecycle_closures": float(lifecycle_closures),
        "client_flow_adjusted_return": _numeric_total(
            reward_summary, "client_utility"
        ),
        "dealer_ex_ante_margin_per_face_sum": _numeric_total(
            reward_summary, "dealer_economics"
        ),
        "dealer_lifecycle_pnl_to_risk_capital": _numeric_total(
            reward_summary, "terminal_lifecycle_pnl"
        ),
        "capital_time_ratio": -_numeric_total(
            reward_summary, "capital_efficiency"
        ),
        "stress_time_ratio": -_numeric_total(
            reward_summary, "risk_change"
        ),
        "query_cost": _numeric_total(reward_summary, "query_cost"),
        "quote_cost": _numeric_total(reward_summary, "quote_cost"),
        "operational_cost": _numeric_total(
            reward_summary, "operational_cost"
        ),
    }
    return metrics, reward_summary, constraint_summary


def _replay_transitions(
    payload: Mapping[str, Any], task: EpisodeTask
) -> list[Mapping[str, Any]]:
    """Re-run every recorded action and require exact economic equality."""

    metadata = payload["metadata"]
    configuration = dict(payload["environment_configuration"])
    if str(metadata["environment_version"]).startswith(_DSL_VERSION_PREFIX):
        from .dsl import StochopiaDSLEnv

        environment = StochopiaDSLEnv(task, **configuration)
    else:
        environment = StochopiaStructurerEnv(task, **configuration)
    if metadata["environment_version"] != environment.environment_version:
        raise ValueError(
            "trajectory environment version differs from installed evaluator"
        )
    if metadata["pricing_version"] != environment.pricing_version:
        raise ValueError(
            "trajectory pricing version differs from installed evaluator"
        )
    entries = payload.get("entries") or []
    environment.reset(
        seed=int(payload["run_id_seed"]),
        options=payload.get("reset_options") or {},
    )
    if to_jsonable(environment.action_schema) != payload["public_action_schema"]:
        raise ValueError("trajectory public action schema cannot be reproduced")

    replayed: list[Mapping[str, Any]] = []
    for entry in entries:
        expected = entry["transition"]
        action_payload = expected.get("action")
        if not isinstance(action_payload, Mapping):
            raise ValueError("trajectory action must be an object")
        action = _action_from_transition(action_payload)
        actual = to_jsonable(environment.step(action))
        if actual != expected:
            mismatched = sorted(
                key
                for key in set(actual) | set(expected)
                if actual.get(key) != expected.get(key)
            )
            raise ValueError(
                "economic replay diverged at entry "
                f"{entry.get('index')}: {mismatched}"
            )
        replayed.append(actual)
    return replayed


def replay_and_evaluate_payload(
    payload: Mapping[str, Any],
    *,
    suite_hash: str | None = None,
    replay: bool = False,
) -> dict[str, Any]:
    """Score one trajectory from its recorded transitions.

    With ``replay=True`` every action is first re-run through a fresh
    environment and each transition must match exactly; this doubles the
    evaluation cost and is off by default.
    """

    if payload.get("format") != TRAJECTORY_FORMAT:
        raise ValueError(
            f"unsupported trajectory format {payload.get('format')!r}; expected {TRAJECTORY_FORMAT}"
        )
    metadata = payload["metadata"]
    task = EpisodeTask.from_manifest(payload["task_manifest"])
    if replay:
        transitions = _replay_transitions(payload, task)
    else:
        transitions = [entry["transition"] for entry in payload.get("entries") or []]
    metrics, reward_summary, constraint_summary = _metrics_from_transitions(transitions)
    metrics = dict(metrics)
    metrics.setdefault("completion_rate", 0.0)
    metrics.setdefault("truncation_rate", 0.0)
    metrics.setdefault("rollout_end_rate", 0.0)
    metrics.setdefault("steps", 0.0)
    run_outcome = payload.get("run_outcome")
    metrics["infrastructure_failure_rate"] = float(
        isinstance(run_outcome, Mapping)
        and run_outcome.get("status") == "infrastructure_error"
    )
    metrics["partial_run_rate"] = float(payload["status"] == "partial")
    is_dsl = str(metadata["environment_version"]).startswith(_DSL_VERSION_PREFIX)
    result = {
        "format": "stochopia.dsl-experiment-evaluation.v1" if is_dsl else EVALUATION_FORMAT,
        "replay_verified": bool(replay),
        "run_id_seed": payload["run_id_seed"],
        "task_hash": task.task_hash,
        "public_task_id": task.public_task_id,
        "suite_hash": suite_hash,
        "policy_kind": payload["policy_run_metadata"].get("policy_kind", ""),
        "policy_name": payload["policy_run_metadata"].get("policy_name", ""),
        "policy_config_hash": _digest(payload["policy_run_metadata"]),
        "environment_version": metadata["environment_version"],
        "pricing_version": metadata["pricing_version"],
        "status": payload["status"],
        "metrics": metrics,
        "reward_summary": reward_summary,
        "constraint_summary": constraint_summary,
    }
    return result


def replay_and_evaluate(
    trajectory_path: str | Path,
    *,
    output_path: str | Path | None = None,
    suite_hash: str | None = None,
    replay: bool = False,
) -> dict[str, Any]:
    try:
        payload = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load trajectory: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory root must be an object")
    result = replay_and_evaluate_payload(payload, suite_hash=suite_hash, replay=replay)
    if output_path is not None:
        _atomic_write(output_path, result)
    return result


def aggregate_evaluations(
    evaluations: Iterable[Mapping[str, Any]],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate v3 metrics by policy with task-clustered confidence intervals."""

    rows = [dict(item) for item in evaluations]
    if not rows:
        raise ValueError("at least one evaluation is required")
    for row in rows:
        if row.get("format") != EVALUATION_FORMAT:
            raise ValueError("all evaluations must be v3 Level-0 results")
    suite_hashes = {row.get("suite_hash") for row in rows}
    if len(suite_hashes) != 1:
        raise ValueError("cannot aggregate evaluations from mixed suites")
    if suite_hashes == {None}:
        raise ValueError(
            "v3 aggregation requires evaluations bound to a frozen task suite"
        )
    for version_key in ("environment_version", "pricing_version"):
        versions = {row.get(version_key) for row in rows}
        if len(versions) != 1:
            raise ValueError(
                f"cannot aggregate evaluations with mixed {version_key}: "
                f"{sorted(map(str, versions))}"
            )

    metric_names = sorted(
        {
            name
            for row in rows
            for name, value in (row.get("metrics") or {}).items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        }
    )
    policies: dict[str, list[dict[str, Any]]] = {}
    policy_hashes: dict[str, set[str]] = {}
    for row in rows:
        policy = str(row.get("policy_name") or "unknown")
        policy_hash = row.get("policy_config_hash")
        if not isinstance(policy_hash, str) or not policy_hash:
            raise ValueError("evaluation is missing policy_config_hash")
        policy_hashes.setdefault(policy, set()).add(policy_hash)
        flat = {"task_hash": row["task_hash"]}
        flat.update(row.get("metrics") or {})
        policies.setdefault(policy, []).append(flat)
    ambiguous = {
        name: sorted(hashes)
        for name, hashes in policy_hashes.items()
        if len(hashes) != 1
    }
    if ambiguous:
        raise ValueError(
            "one policy_name maps to multiple policy configurations: "
            f"{ambiguous}"
        )

    summaries: dict[str, Any] = {}
    for policy, policy_rows in sorted(policies.items()):
        metric_summary: dict[str, Any] = {}
        for metric in metric_names:
            usable = [
                item
                for item in policy_rows
                if isinstance(item.get(metric), (int, float))
                and not isinstance(item.get(metric), bool)
            ]
            if not usable:
                continue
            by_task: dict[str, list[float]] = {}
            for item in usable:
                by_task.setdefault(str(item["task_hash"]), []).append(
                    float(item[metric])
                )
            task_means = [
                {
                    "task_hash": task_hash,
                    metric: sum(values) / len(values),
                }
                for task_hash, values in by_task.items()
            ]
            point, low, high = cluster_bootstrap_ci(
                task_means,
                metric,
                cluster_key="task_hash",
                n_resamples=n_resamples,
                seed=derive_seed(
                    "stochopia.seed.aggregate.v1",
                    policy,
                    metric,
                    seed,
                ),
            )
            metric_summary[metric] = {
                "mean": point,
                "ci_low": low,
                "ci_high": high,
                "n_runs": len(usable),
                "n_tasks": len(task_means),
            }
        summaries[policy] = {
            "runs": len(policy_rows),
            "tasks": len({item["task_hash"] for item in policy_rows}),
            "policy_config_hash": next(iter(policy_hashes[policy])),
            "metrics": metric_summary,
        }

    result = {
        "format": AGGREGATE_FORMAT,
        "suite_hash": next(iter(suite_hashes)),
        "environment_version": rows[0].get("environment_version"),
        "pricing_version": rows[0].get("pricing_version"),
        "evaluation_count": len(rows),
        "policy_count": len(summaries),
        "bootstrap_resamples": n_resamples,
        "seed": seed,
        "policies": summaries,
    }
    return result


def load_evaluations(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load evaluation {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"evaluation {path} root must be an object")
        results.append(payload)
    return results


def save_evaluation_aggregate(
    path: str | Path,
    aggregate: Mapping[str, Any],
) -> Path:
    return _atomic_write(path, dict(aggregate))


__all__ = [
    "AGGREGATE_FORMAT",
    "EVALUATION_FORMAT",
    "TASK_SUITE_FORMAT",
    "TaskSuite",
    "aggregate_evaluations",
    "load_evaluations",
    "replay_and_evaluate",
    "replay_and_evaluate_payload",
    "save_evaluation_aggregate",
]
