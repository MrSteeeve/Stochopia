"""Lossless, versioned trajectory recording for the v3 environment spine."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Mapping

from .. import __version__ as source_package_version
from ..benchmark import ACTION_SCHEMA_VERSION
from ..common import atomic_write_json, canonical_json
from .core import ENVIRONMENT_VERSION, PRICING_VERSION
from .types import (
    REWARD_SCHEMA_VERSION,
    EpisodeTask,
    StepTransition,
)


TRAJECTORY_FORMAT = "stochopia.trajectory-json.v2"
CONSTRAINT_SCHEMA_VERSION = "stochopia.constraint-signals.v1"
_RUNTIME_DISTRIBUTIONS = (
    "httpx",
    "PyYAML",
    "rich",
    "python-dotenv",
    "tushare",
)
def to_jsonable(value: Any) -> Any:
    """Convert typed transitions (including dates/tuples) to JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [to_jsonable(item) for item in value]
        return sorted(converted, key=_canonical_json)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported trajectory value: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return canonical_json(to_jsonable(value))


def _json_snapshot(value: Any) -> Any:
    """Detach from caller objects through canonical JSON."""

    return json.loads(_canonical_json(value))


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        value = completed.stdout.strip()
        return value if value else "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _dependency_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = package_version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Versions required to interpret or reproduce a stored rollout."""

    schema_version: str
    environment_version: str
    pricing_version: str
    task_version: str
    level: str
    task_hash: str
    action_schema_version: str
    reward_schema_version: str
    constraint_schema_version: str
    package_version: str
    git_sha: str


@dataclass(frozen=True)
class TrajectoryEntry:
    """One recorded transition plus the policy metadata that produced it."""

    index: int
    policy_metadata: Mapping[str, Any]
    transition: Mapping[str, Any]


class TrajectoryRecorder:
    """Record complete transitions (including privileged ``info``) as canonical JSON snapshots."""

    def __init__(
        self,
        task_or_env: EpisodeTask | Any,
        *,
        environment_version: str | None = None,
        pricing_version: str | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(task_or_env, EpisodeTask):
            task = task_or_env
            env = None
        else:
            task = getattr(task_or_env, "task", None)
            env = task_or_env
        if not isinstance(task, EpisodeTask):
            raise TypeError("TrajectoryRecorder expects an EpisodeTask or StochopiaStructurerEnv")

        env_version = environment_version or getattr(
            env, "environment_version", ENVIRONMENT_VERSION
        )
        price_version = pricing_version or getattr(env, "pricing_version", PRICING_VERSION)
        environment_config = getattr(env, "configuration", {"unbound_environment": True})
        reset_options = (
            getattr(env, "reset_options", {})
            if env is not None and getattr(env, "is_reset", False)
            else {}
        )
        run_id_seed = (
            getattr(env, "run_id_seed", task.task_seed)
            if env is not None and getattr(env, "is_reset", False)
            else task.task_seed
        )
        dependencies = _dependency_versions()
        if getattr(env, "level", "Level0") == "DSLExperimental":
            for distribution in ("structurerdsl", "torch", "numpy"):
                dependencies[distribution] = package_version(distribution)
        policy_run_metadata = dict(run_metadata or {})
        public_action_schema = (
            getattr(env, "action_schema", {"unbound_environment": True})
            if env is not None and getattr(env, "is_reset", False)
            else {"unbound_environment": True}
        )
        self.metadata = TrajectoryMetadata(
            schema_version=task.schema,
            environment_version=str(env_version),
            pricing_version=str(price_version),
            task_version=task.version,
            level=str(getattr(env, "level", "Level0")),
            task_hash=task.task_hash,
            action_schema_version=str(public_action_schema.get("schema_version", ACTION_SCHEMA_VERSION)),
            reward_schema_version=REWARD_SCHEMA_VERSION,
            constraint_schema_version=CONSTRAINT_SCHEMA_VERSION,
            package_version=source_package_version,
            git_sha=_git_sha(),
        )
        self._task_manifest = _json_snapshot(task.manifest)
        self._environment_configuration = _json_snapshot(environment_config)
        self._reset_options = _json_snapshot(reset_options)
        self._run_id_seed = int(run_id_seed)
        self._dependency_versions = _json_snapshot(dependencies)
        self._policy_run_metadata = _json_snapshot(policy_run_metadata)
        self._public_action_schema = _json_snapshot(public_action_schema)
        self._run_outcome: Mapping[str, Any] | None = None
        self._entries: list[TrajectoryEntry] = []

    @property
    def entries(self) -> tuple[TrajectoryEntry, ...]:
        return tuple(self._entries)

    def record(
        self,
        transition: StepTransition,
        *,
        policy_metadata: Mapping[str, Any] | None = None,
    ) -> TrajectoryEntry:
        if not isinstance(transition, StepTransition):
            raise TypeError("record() expects a StepTransition")
        transition_snapshot = _json_snapshot(transition)
        if not isinstance(transition_snapshot, Mapping):
            raise TypeError("serialized transition must be an object")

        info = transition_snapshot.get("info")
        if not isinstance(info, Mapping):
            raise ValueError("transition.info must serialize to an object")
        if info.get("run_id_seed") != self._run_id_seed:
            raise ValueError("transition run_id_seed disagrees with trajectory")

        policy = _json_snapshot(dict(policy_metadata or {}))
        if not isinstance(policy, Mapping):
            raise TypeError("policy_metadata must serialize to an object")
        entry = TrajectoryEntry(
            index=len(self._entries),
            policy_metadata=policy,
            transition=transition_snapshot,
        )
        self._entries.append(entry)
        return entry

    def mark_run_outcome(
        self,
        status: str,
        details: Mapping[str, Any],
    ) -> None:
        """Commit a non-environment terminal runner outcome to the trajectory."""

        if status != "infrastructure_error":
            raise ValueError("unsupported partial run outcome")
        if self._entries and (
            self._entries[-1].transition.get("terminated")
            or self._entries[-1].transition.get("truncated")
        ):
            raise ValueError("cannot attach a partial outcome after environment end")
        self._run_outcome = _json_snapshot(
                {
                    "status": status,
                    "details": dict(details),
                }
            )

    def to_dict(self) -> dict[str, Any]:
        entries = [to_jsonable(entry) for entry in self._entries]
        terminal = bool(
            entries
            and (
                entries[-1]["transition"].get("terminated")
                or entries[-1]["transition"].get("truncated")
            )
        )
        return {
            "format": TRAJECTORY_FORMAT,
            "metadata": to_jsonable(self.metadata),
            "task_manifest": to_jsonable(self._task_manifest),
            "environment_configuration": to_jsonable(
                self._environment_configuration
            ),
            "reset_options": to_jsonable(self._reset_options),
            "dependency_versions": to_jsonable(self._dependency_versions),
            "policy_run_metadata": to_jsonable(self._policy_run_metadata),
            "public_action_schema": to_jsonable(self._public_action_schema),
            "run_outcome": to_jsonable(self._run_outcome),
            "run_id_seed": self._run_id_seed,
            "status": "complete" if terminal else "partial",
            "entry_count": len(entries),
            "entries": entries,
        }

    def save(self, path: str | Path) -> Path:
        """Save one portable JSON trajectory and return its resolved path."""

        return atomic_write_json(Path(path).expanduser().resolve(), self.to_dict())

__all__ = [
    "CONSTRAINT_SCHEMA_VERSION",
    "TRAJECTORY_FORMAT",
    "TrajectoryEntry",
    "TrajectoryMetadata",
    "TrajectoryRecorder",
    "to_jsonable",
]
