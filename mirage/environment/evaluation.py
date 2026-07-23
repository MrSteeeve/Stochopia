"""Versioned v3 task suites, economic replay, and benchmark aggregation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..products import ProductSpec
from ..stats import cluster_bootstrap_ci, derive_seed
from .agent_runner import parse_environment_action
from .core import MirageStructurerEnv
from .trajectory import TrajectoryRecorder, to_jsonable
from .types import (
    EpisodeTask,
    InvalidAction,
    RequestQuote,
    Skip,
    SubmitDesign,
    SubmitProduct,
)


TASK_SUITE_FORMAT = "mirage-task-suite.v3"
EVALUATION_FORMAT = "mirage-trajectory-evaluation.v3"
AGGREGATE_FORMAT = "mirage-evaluation-aggregate.v3"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_write(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


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
        expected_hash = raw.pop("suite_hash", None)
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
        if raw.get("task_count") != len(suite.tasks):
            raise ValueError("task suite task_count mismatch")
        if raw.get("task_hashes") != [item.task_hash for item in suite.tasks]:
            raise ValueError("task suite task_hashes mismatch")
        if expected_hash != suite.suite_hash:
            raise ValueError("task suite hash mismatch")
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
) -> tuple[dict[str, float], Mapping[str, Any], Mapping[str, Any]]:
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
        "hard_execution_rate_given_submission": (
            hard_passes / submissions if submissions else 0.0
        ),
        "contract_acceptance_rate_given_hard_pass": (
            contract_passes / contract_eligible if contract_eligible else 0.0
        ),
        "settlement_acceptance_rate": (
            accepted / submissions if submissions else 0.0
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


def replay_and_evaluate_payload(
    payload: Mapping[str, Any],
    *,
    suite_hash: str | None = None,
) -> dict[str, Any]:
    """Re-run every action and require exact economic transition equality."""

    if not TrajectoryRecorder.verify_payload(payload):
        raise ValueError("trajectory hash verification failed")
    metadata = payload["metadata"]
    task = EpisodeTask.from_manifest(payload["task_manifest"])
    if task.task_hash != metadata["task_hash"]:
        raise ValueError("trajectory task hash does not match reconstructed task")
    configuration = dict(payload["environment_configuration"])
    environment = MirageStructurerEnv(task, **configuration)
    if metadata["environment_version"] != environment.environment_version:
        raise ValueError(
            "trajectory environment version differs from installed evaluator"
        )
    if metadata["pricing_version"] != environment.pricing_version:
        raise ValueError(
            "trajectory pricing version differs from installed evaluator"
        )
    entries = payload.get("entries") or []
    seed = int(payload["run_id_seed"])
    _, reset_info = environment.reset(
        seed=seed,
        options=payload.get("reset_options") or {},
    )
    if reset_info["state_hash"] != payload["initial_state_hash"]:
        raise ValueError("trajectory initial state cannot be reproduced")
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

    metrics, reward_summary, constraint_summary = _metrics_from_transitions(
        replayed
    )
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
    result = {
        "format": EVALUATION_FORMAT,
        "replay_verified": True,
        "trajectory_root_hash": payload["root_hash"],
        "run_id_seed": payload["run_id_seed"],
        "task_hash": task.task_hash,
        "public_task_id": task.public_task_id,
        "suite_hash": suite_hash,
        "policy_kind": payload["policy_run_metadata"].get("policy_kind", ""),
        "policy_name": payload["policy_run_metadata"].get("policy_name", ""),
        "policy_config_hash": metadata["policy_run_metadata_hash"],
        "environment_version": metadata["environment_version"],
        "pricing_version": metadata["pricing_version"],
        "status": payload["status"],
        "metrics": metrics,
        "reward_summary": reward_summary,
        "constraint_summary": constraint_summary,
    }
    result["evaluation_hash"] = _digest(result)
    return result


def replay_and_evaluate(
    trajectory_path: str | Path,
    *,
    output_path: str | Path | None = None,
    suite_hash: str | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load trajectory: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("trajectory root must be an object")
    result = replay_and_evaluate_payload(payload, suite_hash=suite_hash)
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
        if row.get("format") != EVALUATION_FORMAT or row.get("replay_verified") is not True:
            raise ValueError("all evaluations must be replay-verified v3 results")
        expected_hash = row.get("evaluation_hash")
        check = dict(row)
        check.pop("evaluation_hash", None)
        if expected_hash != _digest(check):
            raise ValueError("evaluation hash mismatch")
    suite_hashes = {row.get("suite_hash") for row in rows}
    if len(suite_hashes) != 1:
        raise ValueError("cannot aggregate evaluations from mixed suites")
    if suite_hashes == {None}:
        raise ValueError(
            "v3 aggregation requires evaluations bound to a frozen task suite"
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
                    "mirage.v3.aggregate",
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
        "evaluation_count": len(rows),
        "policy_count": len(summaries),
        "bootstrap_resamples": n_resamples,
        "seed": seed,
        "policies": summaries,
    }
    result["aggregate_hash"] = _digest(result)
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
