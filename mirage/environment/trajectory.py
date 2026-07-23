"""Lossless, versioned trajectory recording for the v3 environment spine."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .. import __version__ as source_package_version
from .core import ENVIRONMENT_VERSION, PRICING_VERSION
from .types import (
    REWARD_SCHEMA_VERSION,
    TASK_SCHEMA,
    EpisodeTask,
    StepTransition,
)


TRAJECTORY_FORMAT = "mirage-trajectory-json-v2"
ACTION_SCHEMA_VERSION = "mirage.environment.action.v3"
CONSTRAINT_SCHEMA_VERSION = "mirage.constraint-signals.v1"


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
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


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


def _installed_package_version() -> str:
    try:
        return package_version("mirage-bench")
    except PackageNotFoundError:
        return source_package_version


def _implementation_hash() -> str:
    pricing_path = Path(__file__).resolve().parents[1] / "pricing.py"
    try:
        return hashlib.sha256(pricing_path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


@dataclass(frozen=True)
class TrajectoryMetadata:
    """Versions required to interpret or reproduce a stored rollout."""

    schema_version: str
    environment_version: str
    pricing_version: str
    task_version: str
    level: str
    task_hash: str
    environment_config_hash: str
    action_schema_version: str
    reward_schema_version: str
    constraint_schema_version: str
    package_version: str
    git_sha: str
    pricing_implementation_hash: str


@dataclass(frozen=True)
class TrajectoryEntry:
    """A transition plus a cryptographic link to the preceding entry."""

    index: int
    previous_record_hash: str
    record_hash: str
    policy_metadata: Mapping[str, Any]
    transition: Mapping[str, Any]


class TrajectoryRecorder:
    """Record complete transitions and enforce state/hash continuity.

    The record hash commits to the version metadata, entry index, prior record
    hash, policy metadata, and the full serialised transition (including
    privileged ``info``).  In addition, adjacent transitions must satisfy
    ``previous.state_hash_after == next.state_hash_before``.
    """

    def __init__(
        self,
        task_or_env: EpisodeTask | Any,
        *,
        initial_state_hash: str | None = None,
        environment_version: str | None = None,
        pricing_version: str | None = None,
    ) -> None:
        if isinstance(task_or_env, EpisodeTask):
            task = task_or_env
            env = None
        else:
            task = getattr(task_or_env, "task", None)
            env = task_or_env
        if not isinstance(task, EpisodeTask):
            raise TypeError("TrajectoryRecorder expects an EpisodeTask or MirageStructurerEnv")

        env_version = environment_version or getattr(
            env, "environment_version", ENVIRONMENT_VERSION
        )
        price_version = pricing_version or getattr(env, "pricing_version", PRICING_VERSION)
        environment_config = getattr(env, "configuration", {"unbound_environment": True})
        self.metadata = TrajectoryMetadata(
            schema_version=task.schema,
            environment_version=str(env_version),
            pricing_version=str(price_version),
            task_version=task.version,
            level=str(getattr(env, "level", "Level0")),
            task_hash=task.task_hash,
            environment_config_hash=_digest(environment_config),
            action_schema_version=ACTION_SCHEMA_VERSION,
            reward_schema_version=REWARD_SCHEMA_VERSION,
            constraint_schema_version=CONSTRAINT_SCHEMA_VERSION,
            package_version=_installed_package_version(),
            git_sha=_git_sha(),
            pricing_implementation_hash=_implementation_hash(),
        )
        self._task_manifest = _freeze_json(_json_snapshot(task.manifest))
        if initial_state_hash is None and env is not None and getattr(env, "is_reset", False):
            initial_state_hash = str(env.state_hash)
        self._initial_state_hash = initial_state_hash
        self._entries: list[TrajectoryEntry] = []

    @property
    def initial_state_hash(self) -> str | None:
        return self._initial_state_hash

    @property
    def entries(self) -> tuple[TrajectoryEntry, ...]:
        return tuple(self._entries)

    @property
    def transitions(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(entry.transition for entry in self._entries)

    def record(
        self,
        transition: StepTransition,
        *,
        policy_metadata: Mapping[str, Any] | None = None,
    ) -> TrajectoryEntry:
        if not isinstance(transition, StepTransition):
            raise TypeError("record() expects a StepTransition")
        transition_snapshot = _freeze_json(_json_snapshot(transition))
        if not isinstance(transition_snapshot, Mapping):
            raise TypeError("serialized transition must be an object")

        if self._initial_state_hash is None:
            self._initial_state_hash = str(transition_snapshot["state_hash_before"])
        if not self._entries:
            if transition_snapshot["state_hash_before"] != self._initial_state_hash:
                raise ValueError("first transition does not start at initial_state_hash")
            previous_record_hash = self._genesis_hash()
        else:
            previous = self._entries[-1]
            if (
                transition_snapshot["state_hash_before"]
                != previous.transition["state_hash_after"]
            ):
                raise ValueError("transition state hashes are not continuous")
            previous_record_hash = previous.record_hash

        info = transition_snapshot.get("info")
        if not isinstance(info, Mapping):
            raise ValueError("transition.info must serialize to an object")
        info_hash = info.get("state_hash")
        if (
            info_hash is not None
            and str(info_hash) != transition_snapshot["state_hash_after"]
        ):
            raise ValueError("transition.info state_hash disagrees with state_hash_after")

        index = len(self._entries)
        policy = _freeze_json(_json_snapshot(dict(policy_metadata or {})))
        if not isinstance(policy, Mapping):
            raise TypeError("policy_metadata must serialize to an object")
        hash_payload = self._entry_hash_payload(
            index=index,
            previous_record_hash=previous_record_hash,
            policy_metadata=policy,
            transition=transition_snapshot,
        )
        entry = TrajectoryEntry(
            index=index,
            previous_record_hash=previous_record_hash,
            record_hash=_digest(hash_payload),
            policy_metadata=policy,
            transition=transition_snapshot,
        )
        self._entries.append(entry)
        return entry

    append = record

    def _genesis_hash(self) -> str:
        return _digest(
            {
                "format": TRAJECTORY_FORMAT,
                "metadata": self.metadata,
                "initial_state_hash": self._initial_state_hash,
            }
        )

    def _entry_hash_payload(
        self,
        *,
        index: int,
        previous_record_hash: str,
        policy_metadata: Mapping[str, Any],
        transition: StepTransition | Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "metadata": to_jsonable(self.metadata),
            "index": index,
            "previous_record_hash": previous_record_hash,
            "policy_metadata": to_jsonable(policy_metadata),
            "transition": to_jsonable(transition),
        }

    def to_dict(self) -> dict[str, Any]:
        entries = [to_jsonable(entry) for entry in self._entries]
        terminal = bool(
            entries
            and (
                entries[-1]["transition"].get("terminated")
                or entries[-1]["transition"].get("truncated")
            )
        )
        final_record_hash = (
            entries[-1]["record_hash"] if entries else self._genesis_hash()
        )
        final_state_hash = (
            entries[-1]["transition"]["state_hash_after"]
            if entries
            else self._initial_state_hash
        )
        payload = {
            "format": TRAJECTORY_FORMAT,
            "metadata": to_jsonable(self.metadata),
            "task_manifest": to_jsonable(self._task_manifest),
            "initial_state_hash": self._initial_state_hash,
            "genesis_hash": self._genesis_hash(),
            "status": "complete" if terminal else "partial",
            "entry_count": len(entries),
            "final_record_hash": final_record_hash,
            "final_state_hash": final_state_hash,
            "entries": entries,
        }
        payload["root_hash"] = _digest(payload)
        return payload

    def save(self, path: str | Path) -> Path:
        """Save one portable JSON trajectory and return its resolved path."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if not self.verify_payload(payload):
            raise ValueError("refusing to save an invalid trajectory")
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, destination)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        return destination

    def verify_hash_chain(self) -> bool:
        return self.verify_payload(self.to_dict())

    def validate_hash_chain(self) -> None:
        if not self.verify_hash_chain():
            raise ValueError("trajectory hash/state chain verification failed")

    @staticmethod
    def verify(path: str | Path) -> bool:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        return TrajectoryRecorder.verify_payload(payload)

    @staticmethod
    def verify_payload(payload: Mapping[str, Any]) -> bool:
        """Verify versions, record hashes, and adjacent state hashes."""

        try:
            if payload.get("format") != TRAJECTORY_FORMAT:
                return False
            metadata = payload["metadata"]
            if not isinstance(metadata, Mapping):
                return False
            required_versions = (
                "schema_version",
                "environment_version",
                "pricing_version",
                "task_version",
                "level",
                "task_hash",
                "environment_config_hash",
                "action_schema_version",
                "reward_schema_version",
                "constraint_schema_version",
                "package_version",
                "git_sha",
                "pricing_implementation_hash",
            )
            if any(not isinstance(metadata.get(key), str) or not metadata[key] for key in required_versions):
                return False
            initial_state_hash = payload.get("initial_state_hash")
            if not isinstance(initial_state_hash, str) or not initial_state_hash:
                return False
            task_manifest = payload.get("task_manifest")
            if not isinstance(task_manifest, Mapping):
                return False
            if _digest(task_manifest) != metadata["task_hash"]:
                return False
            if metadata.get("schema_version") != task_manifest.get("schema"):
                return False
            genesis = _digest(
                {
                    "format": TRAJECTORY_FORMAT,
                    "metadata": metadata,
                    "initial_state_hash": initial_state_hash,
                }
            )
            if payload.get("genesis_hash") != genesis:
                return False

            entries = payload.get("entries")
            if not isinstance(entries, list):
                return False
            if payload.get("entry_count") != len(entries):
                return False
            status = payload.get("status")
            if status not in {"complete", "partial"}:
                return False
            previous_record_hash = genesis
            previous_state_hash = initial_state_hash
            terminal_seen = False
            for index, entry in enumerate(entries):
                if terminal_seen or not isinstance(entry, Mapping):
                    return False
                if entry.get("index") != index:
                    return False
                if entry.get("previous_record_hash") != previous_record_hash:
                    return False
                transition = entry.get("transition")
                policy_metadata = entry.get("policy_metadata")
                if not isinstance(transition, Mapping) or not isinstance(policy_metadata, Mapping):
                    return False
                if transition.get("state_hash_before") != previous_state_hash:
                    return False
                state_hash_after = transition.get("state_hash_after")
                if not isinstance(state_hash_after, str) or not state_hash_after:
                    return False
                info = transition.get("info")
                if not isinstance(info, Mapping) or info.get("state_hash") != state_hash_after:
                    return False
                expected_record_hash = _digest(
                    {
                        "metadata": to_jsonable(metadata),
                        "index": index,
                        "previous_record_hash": previous_record_hash,
                        "policy_metadata": to_jsonable(policy_metadata),
                        "transition": to_jsonable(transition),
                    }
                )
                if entry.get("record_hash") != expected_record_hash:
                    return False
                previous_record_hash = expected_record_hash
                previous_state_hash = state_hash_after
                terminal_seen = bool(
                    transition.get("terminated") or transition.get("truncated")
                )
            if status == "complete" and (not entries or not terminal_seen):
                return False
            if status == "partial" and terminal_seen:
                return False
            expected_final_record = (
                previous_record_hash if entries else genesis
            )
            if payload.get("final_record_hash") != expected_final_record:
                return False
            if payload.get("final_state_hash") != previous_state_hash:
                return False
            root_payload = dict(payload)
            root_hash = root_payload.pop("root_hash", None)
            if not isinstance(root_hash, str) or root_hash != _digest(root_payload):
                return False
            return True
        except (KeyError, TypeError, ValueError, OverflowError):
            return False


def verify_trajectory(path: str | Path) -> bool:
    """Module-level convenience wrapper."""

    return TrajectoryRecorder.verify(path)


__all__ = [
    "ACTION_SCHEMA_VERSION",
    "CONSTRAINT_SCHEMA_VERSION",
    "TRAJECTORY_FORMAT",
    "TrajectoryEntry",
    "TrajectoryMetadata",
    "TrajectoryRecorder",
    "to_jsonable",
    "verify_trajectory",
]
