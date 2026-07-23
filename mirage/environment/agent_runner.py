"""External-agent adapters and rollout runner for the MIRAGE v3 environment.

The deterministic environment intentionally contains no LLM calls.  This
module is the explicit boundary for testing either:

* an API-backed model through :class:`LLMAgentPolicy`; or
* an arbitrary local executable through :class:`CommandAgentPolicy`.

Both adapters consume the same leakage-safe observation and produce the same
typed v3 actions.  The command protocol is one JSON request on stdin and one
JSON action on stdout for every environment step.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import shlex
import shutil
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from ..benchmark_runner import extract_action_json
from ..llm import BaseLLMClient, LLMError, ModelConfig, create_client
from ..products import ProductError, parse_product_spec
from .core import MirageStructurerEnv
from .trajectory import TrajectoryRecorder, to_jsonable
from .types import (
    AskClient,
    EnvironmentAction,
    InvalidAction,
    Observation,
    RequestQuote,
    Skip,
    SubmitDesign,
    SubmitProduct,
)


AGENT_REQUEST_SCHEMA = "mirage.agent-request.v3"
AGENT_RUN_SCHEMA = "mirage.agent-run.v3"
AGENT_PROMPT_VERSION = "mirage.agent-prompt.v3"
DEFAULT_HISTORY_RECORDS = 32
DEFAULT_HISTORY_CHARS = 96_000
DEFAULT_COMMAND_OUTPUT_CHARS = 200_000


@lru_cache(maxsize=256)
def _file_sha256(
    resolved_path: str,
    size: int,
    mtime_ns: int,
) -> str:
    """Hash one command artifact; stat fields make the cache change-aware."""

    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_artifacts(argv: Sequence[str]) -> tuple[dict[str, Any], ...]:
    artifacts: list[dict[str, Any]] = []
    for index, argument in enumerate(argv):
        candidate = Path(argument).expanduser()
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        stat = resolved.stat()
        artifacts.append(
            {
                "argument_index": index,
                "path": str(resolved),
                "size": stat.st_size,
                "sha256": _file_sha256(
                    str(resolved),
                    stat.st_size,
                    stat.st_mtime_ns,
                ),
            }
        )
    return tuple(artifacts)


@lru_cache(maxsize=1)
def load_agent_system_prompt() -> str:
    """Load the exact v3 agent contract shipped in the installed wheel."""

    prompt = (
        files("mirage.resources")
        .joinpath("v3_agent.md")
        .read_text(encoding="utf-8")
    )
    required = ('"action":"ask_client"', '"action":"submit_product"')
    if any(marker not in prompt for marker in required):
        raise RuntimeError("packaged v3 agent prompt is incomplete")
    return prompt


@dataclass(frozen=True)
class AgentRequest:
    """One public, versioned policy request.

    ``observation`` and ``history`` contain only values already returned by the
    non-privileged environment.  The task manifest and hidden client profile
    are deliberately absent.
    """

    request_id: str
    run_seed: int
    action_schema: Mapping[str, Any]
    observation: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    history_omitted: int
    system_prompt: str
    schema: str = AGENT_REQUEST_SCHEMA

    def to_dict(self, *, include_system_prompt: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "request_id": self.request_id,
            "run_seed": self.run_seed,
            "action_schema": to_jsonable(self.action_schema),
            "instruction": (
                "Choose exactly one currently available action and return one "
                "JSON object. Do not return prose outside the JSON object."
            ),
            "observation": to_jsonable(self.observation),
            "history": to_jsonable(self.history),
            "history_omitted": self.history_omitted,
        }
        if include_system_prompt:
            payload["system_prompt"] = self.system_prompt
        return payload


@dataclass(frozen=True)
class AgentDecision:
    """A typed action plus the raw policy response needed for audit."""

    action: EnvironmentAction
    raw_output: str
    parser_error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class AgentPolicy(Protocol):
    """Minimal asynchronous policy boundary used by the v3 runner."""

    kind: str
    name: str
    total_usage: Mapping[str, Any]

    async def act(self, request: AgentRequest) -> AgentDecision:
        """Return one decision for the supplied public request."""

    def reproducibility_metadata(self) -> Mapping[str, Any]:
        """Return public configuration needed to interpret a run."""


class AgentActionError(ValueError):
    """The policy response does not satisfy the v3 action contract."""


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    *,
    action: str,
    allowed: set[str],
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise AgentActionError(
            f"{action} contains unknown fields: {sorted(unknown)}"
        )


def _parse_level0_product(payload: Mapping[str, Any]):
    product = parse_product_spec(dict(payload))
    if product.product_type == "custom":
        raise AgentActionError(
            "custom is not part of the Level-0 finite action grammar"
        )
    return product


def parse_environment_action(raw: str | Mapping[str, Any]) -> EnvironmentAction:
    """Parse one strict external response into a typed v3 environment action."""

    if isinstance(raw, str):
        try:
            payload = extract_action_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            raise AgentActionError(f"no valid action JSON: {exc}") from exc
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        raise AgentActionError("agent response must be text or a JSON object")

    if not isinstance(payload, dict):
        raise AgentActionError("action JSON must be an object")
    action = payload.get("action")
    if not isinstance(action, str):
        raise AgentActionError("action must be a string")

    try:
        if action == "ask_client":
            _reject_unknown_fields(
                payload,
                action=action,
                allowed={"action", "topic"},
            )
            topic = payload.get("topic")
            if not isinstance(topic, str) or not topic.strip():
                raise AgentActionError("ask_client.topic must be a non-empty string")
            return AskClient(topic=topic.strip())

        if action == "request_quote":
            _reject_unknown_fields(
                payload,
                action=action,
                allowed={"action", "product"},
            )
            product = payload.get("product")
            if not isinstance(product, dict):
                raise AgentActionError("request_quote.product must be an object")
            return RequestQuote(product=_parse_level0_product(product))

        if action == "submit_design":
            _reject_unknown_fields(
                payload,
                action=action,
                allowed={"action", "quote_id", "explanation"},
            )
            quote_id = payload.get("quote_id")
            explanation = payload.get("explanation", "")
            if not isinstance(quote_id, str) or not quote_id.strip():
                raise AgentActionError(
                    "submit_design.quote_id must be a non-empty string"
                )
            if not isinstance(explanation, str):
                raise AgentActionError("submit_design.explanation must be a string")
            return SubmitDesign(
                quote_id=quote_id.strip(),
                explanation=explanation,
            )

        if action == "submit_product":
            _reject_unknown_fields(
                payload,
                action=action,
                allowed={"action", "product", "explanation"},
            )
            product = payload.get("product")
            explanation = payload.get("explanation", "")
            if not isinstance(product, dict):
                raise AgentActionError("submit_product.product must be an object")
            if not isinstance(explanation, str):
                raise AgentActionError("submit_product.explanation must be a string")
            return SubmitProduct(
                product=_parse_level0_product(product),
                explanation=explanation,
            )

        if action == "skip":
            _reject_unknown_fields(
                payload,
                action=action,
                allowed={"action", "reason"},
            )
            reason = payload.get("reason", "")
            if not isinstance(reason, str):
                raise AgentActionError("skip.reason must be a string")
            return Skip(reason=reason)
    except ProductError as exc:
        raise AgentActionError(f"invalid product: {exc}") from exc

    raise AgentActionError(
        "unknown action; expected ask_client, request_quote, "
        "submit_design, submit_product, or skip"
    )


def _decision_from_raw(
    raw_output: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> AgentDecision:
    try:
        action = parse_environment_action(raw_output)
    except AgentActionError as exc:
        message = str(exc)
        return AgentDecision(
            action=InvalidAction(reason=message, raw=raw_output),
            raw_output=raw_output,
            parser_error=message,
            metadata=dict(metadata or {}),
        )
    return AgentDecision(
        action=action,
        raw_output=raw_output,
        metadata=dict(metadata or {}),
    )


class LLMAgentPolicy:
    """Adapt a :class:`BaseLLMClient` to the v3 policy protocol."""

    kind = "api"

    def __init__(
        self,
        client: BaseLLMClient,
        *,
        name: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        system_prompt: str | None = None,
    ) -> None:
        if not isinstance(client, BaseLLMClient):
            raise TypeError("client must be a BaseLLMClient")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.client = client
        self.name = name.strip()
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or load_agent_system_prompt()

    @property
    def total_usage(self) -> Mapping[str, Any]:
        return dict(self.client.total_usage)

    def reproducibility_metadata(self) -> Mapping[str, Any]:
        config = getattr(self.client, "config", None)
        provider = str(getattr(config, "provider", "mock"))
        model = str(getattr(config, "model", "") or self.name)
        base_url = _sanitized_base_url(
            str(getattr(config, "base_url", ""))
        )
        return {
            "policy_kind": self.kind,
            "policy_name": self.name,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key_env": str(getattr(config, "api_key_env", "")),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": getattr(config, "timeout", None),
            **_prompt_metadata(self.system_prompt),
        }

    async def act(self, request: AgentRequest) -> AgentDecision:
        payload = request.to_dict(include_system_prompt=False)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            },
        ]
        started = time.monotonic()
        raw = await self.client.chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=request.run_seed,
        )
        duration_ms = round((time.monotonic() - started) * 1000, 3)
        return _decision_from_raw(
            raw,
            metadata={
                "duration_ms": duration_ms,
                "usage_after_call": dict(self.client.total_usage),
            },
        )


def create_api_agent_policy(
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key_env: str,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    timeout: float = 120.0,
    name: str | None = None,
) -> LLMAgentPolicy:
    """Create a direct API-backed policy without editing ``models.yaml``.

    The key itself is read from ``api_key_env``.  It is never accepted as a
    function argument, copied into a request, or written to a trajectory.
    """

    policy_name = name or f"{provider}:{model}"
    config = ModelConfig(
        name=policy_name,
        provider=provider,
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if provider not in {"openai-compatible", "anthropic"}:
        raise LLMError(
            "direct API provider must be openai-compatible or anthropic"
        )
    if not base_url.strip():
        raise LLMError("direct API base_url must be non-empty")
    if not model.strip():
        raise LLMError("direct API model must be non-empty")
    if not api_key_env.strip():
        raise LLMError("direct API api_key_env must be non-empty")
    if not os.environ.get(api_key_env, "").strip():
        raise LLMError(
            f"missing API key: environment variable {api_key_env} is empty"
        )
    client = create_client(policy_name, {policy_name: config})
    return LLMAgentPolicy(
        client,
        name=policy_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class CommandAgentPolicy:
    """Invoke a local executable once per step using the JSON stdio protocol.

    The command is tokenised with :func:`shlex.split` when supplied as a
    string.  It is executed directly without a shell, so pipes, redirects and
    command substitution are intentionally unsupported.
    """

    kind = "command"

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        timeout: float = 180.0,
        max_output_chars: int = DEFAULT_COMMAND_OUTPUT_CHARS,
        system_prompt: str | None = None,
    ) -> None:
        if isinstance(command, str):
            argv = shlex.split(command)
        else:
            argv = [str(item) for item in command]
        if not argv:
            raise ValueError("agent command must not be empty")
        if timeout <= 0:
            raise ValueError("command timeout must be positive")
        if (
            isinstance(max_output_chars, bool)
            or not isinstance(max_output_chars, int)
            or max_output_chars < 1
        ):
            raise ValueError("max_output_chars must be a positive integer")

        executable = argv[0]
        if os.sep in executable:
            resolved = Path(executable).expanduser()
            if not resolved.is_file() or not os.access(resolved, os.X_OK):
                raise ValueError(f"agent executable is not runnable: {executable}")
            argv[0] = str(resolved.resolve())
        else:
            resolved_name = shutil.which(executable)
            if resolved_name is None:
                raise ValueError(f"agent executable was not found: {executable}")
            argv[0] = resolved_name

        self.argv = tuple(argv)
        argv_hash = hashlib.sha256(
            json.dumps(
                self.argv,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._artifacts = _command_artifacts(self.argv)
        policy_hash = hashlib.sha256(
            json.dumps(
                {
                    "argv_sha256": argv_hash,
                    "artifacts": self._artifacts,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.name = (
            f"command:{Path(self.argv[0]).name}:{policy_hash[:12]}"
        )
        self._argv_hash = argv_hash
        self._policy_hash = policy_hash
        self.timeout = float(timeout)
        self.max_output_chars = max_output_chars
        self.system_prompt = system_prompt or load_agent_system_prompt()
        self._usage = {"calls": 0}

    @property
    def total_usage(self) -> Mapping[str, Any]:
        return dict(self._usage)

    def reproducibility_metadata(self) -> Mapping[str, Any]:
        return {
            "policy_kind": self.kind,
            "policy_name": self.name,
            "executable": self.argv[0],
            "argv_sha256": self._argv_hash,
            "policy_sha256": self._policy_hash,
            "argument_files": self._artifacts,
            "argument_count": len(self.argv),
            "timeout_seconds": self.timeout,
            "max_output_chars": self.max_output_chars,
            **_prompt_metadata(self.system_prompt),
        }

    async def act(self, request: AgentRequest) -> AgentDecision:
        request_bytes = (
            json.dumps(
                request.to_dict(include_system_prompt=True),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        started = time.monotonic()
        self._usage["calls"] += 1
        process_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            process_kwargs["start_new_session"] = True
        process = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_kwargs,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(request_bytes),
                timeout=self.timeout,
            )
        except TimeoutError:
            await _terminate_process_tree(process)
            raise LLMError(
                f"agent command exceeded {self.timeout:g}s timeout"
            ) from None
        except asyncio.CancelledError:
            await _terminate_process_tree(process)
            raise
        duration_ms = round((time.monotonic() - started) * 1000, 3)
        stderr_text = stderr.decode("utf-8", errors="replace")
        stdout_text = stdout.decode("utf-8", errors="replace")
        if len(stdout_text) > self.max_output_chars:
            raise LLMError(
                "agent command stdout exceeded "
                f"{self.max_output_chars} characters"
            )
        if process.returncode != 0:
            raise LLMError(
                f"agent command exited with status {process.returncode}: "
                f"{stderr_text[:2000]}"
            )
        if not stdout_text.strip():
            raise LLMError(
                "agent command returned empty stdout"
                + (f": {stderr_text[:2000]}" if stderr_text else "")
            )
        return _decision_from_raw(
            stdout_text,
            metadata={
                "duration_ms": duration_ms,
                "returncode": process.returncode,
                "stderr": stderr_text[:2000],
            },
        )


@dataclass(frozen=True)
class AgentEpisodeResult:
    """A compact run summary plus the verified trajectory payload."""

    task_hash: str
    policy_kind: str
    policy_name: str
    status: str
    steps: int
    final_round: int
    accepted_submissions: int
    invalid_actions: int
    parser_errors: int
    invocation_errors: int
    infrastructure_error: Mapping[str, Any] | None
    reward_summary: Mapping[str, Any]
    constraint_summary: Mapping[str, Any]
    usage: Mapping[str, Any]
    trajectory_verified: bool
    trajectory_path: str | None
    trajectory: Mapping[str, Any] = field(repr=False)
    schema: str = AGENT_RUN_SCHEMA

    def summary(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_hash": self.task_hash,
            "policy_kind": self.policy_kind,
            "policy_name": self.policy_name,
            "status": self.status,
            "steps": self.steps,
            "final_round": self.final_round,
            "accepted_submissions": self.accepted_submissions,
            "invalid_actions": self.invalid_actions,
            "parser_errors": self.parser_errors,
            "invocation_errors": self.invocation_errors,
            "infrastructure_error": to_jsonable(self.infrastructure_error),
            "reward_summary": to_jsonable(self.reward_summary),
            "constraint_summary": to_jsonable(self.constraint_summary),
            "usage": to_jsonable(self.usage),
            "trajectory_verified": self.trajectory_verified,
            "trajectory_path": self.trajectory_path,
        }


def _trim_history(
    history: list[Mapping[str, Any]],
    *,
    max_records: int,
    max_chars: int,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    kept = list(history[-max_records:])
    while kept:
        encoded = json.dumps(
            to_jsonable(kept),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(encoded) <= max_chars:
            break
        kept.pop(0)
    return tuple(kept), len(history) - len(kept)


def _request_id(public_task_id: str, run_seed: int, step_index: int) -> str:
    """Derive a reproducible request id from public policy inputs only."""

    preimage = f"{public_task_id}:{run_seed}:{step_index}".encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()[:24]


def _sanitized_base_url(value: str) -> str:
    """Remove credentials, query parameters, and fragments from a public URL."""

    if not value:
        return ""
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if parsed.port is not None:
            hostname = f"{hostname}:{parsed.port}"
    except ValueError:
        return "invalid-url-sha256:" + hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()
    return urlunsplit(
        (parsed.scheme, hostname, parsed.path.rstrip("/"), "", "")
    )


def _prompt_metadata(prompt: str) -> dict[str, str]:
    return {
        "system_prompt_version": AGENT_PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
    }


def _policy_reproducibility_metadata(policy: AgentPolicy) -> dict[str, Any]:
    provider = getattr(policy, "reproducibility_metadata", None)
    if callable(provider):
        metadata = provider()
        if not isinstance(metadata, Mapping):
            raise TypeError(
                "policy.reproducibility_metadata() must return a mapping"
            )
        return dict(metadata)
    prompt = str(getattr(policy, "system_prompt", load_agent_system_prompt()))
    return {
        "policy_kind": str(policy.kind),
        "policy_name": str(policy.name),
        **_prompt_metadata(prompt),
    }


async def _terminate_process_tree(
    process: asyncio.subprocess.Process,
) -> None:
    """Best-effort cleanup of a command and every descendant in its session."""

    if process.returncode is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
    else:
        process.kill()
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except TimeoutError:
        if process.returncode is None:
            process.kill()
        await process.wait()


async def run_agent_episode(
    environment: MirageStructurerEnv,
    policy: AgentPolicy,
    *,
    seed: int | None = None,
    trajectory_path: str | Path | None = None,
    max_history_records: int = DEFAULT_HISTORY_RECORDS,
    max_history_chars: int = DEFAULT_HISTORY_CHARS,
    record_raw_output: bool = True,
) -> AgentEpisodeResult:
    """Run one external policy against v3 and record every transition.

    Malformed policy output is a typed invalid action. Transport, provider,
    executable, and timeout failures are infrastructure errors: the environment
    is not stepped and the partial trajectory records the runner outcome.
    Missing credentials and invalid commands should still be rejected when the
    policy is constructed, before this function is called.
    """

    if not isinstance(environment, MirageStructurerEnv):
        raise TypeError("environment must be a MirageStructurerEnv")
    if (
        isinstance(max_history_records, bool)
        or not isinstance(max_history_records, int)
        or max_history_records < 1
    ):
        raise ValueError("max_history_records must be a positive integer")
    if (
        isinstance(max_history_chars, bool)
        or not isinstance(max_history_chars, int)
        or max_history_chars < 1
    ):
        raise ValueError("max_history_chars must be a positive integer")

    # The policy-facing run seed is public experiment configuration.  Never
    # fall back to EpisodeTask.task_seed here because that field belongs to
    # the hidden task manifest and could become a task-identity side channel.
    effective_seed = 0 if seed is None else seed
    observation, info = environment.reset(seed=effective_seed)
    recorder = TrajectoryRecorder(
        environment,
        initial_state_hash=str(info["state_hash"]),
        run_metadata=_policy_reproducibility_metadata(policy),
    )
    history: list[Mapping[str, Any]] = []
    invalid_actions = 0
    parser_errors = 0
    invocation_errors = 0
    accepted_submissions = 0
    infrastructure_error: dict[str, Any] | None = None

    while not environment.done:
        visible_history, history_omitted = _trim_history(
            history,
            max_records=max_history_records,
            max_chars=max_history_chars,
        )
        request = AgentRequest(
            request_id=_request_id(
                environment.task.public_task_id,
                int(info["run_id_seed"]),
                observation.step_index,
            ),
            run_seed=int(info["run_id_seed"]),
            action_schema=environment.action_schema,
            observation=asdict(observation),
            history=visible_history,
            history_omitted=history_omitted,
            system_prompt=str(
                getattr(policy, "system_prompt", load_agent_system_prompt())
            ),
        )

        try:
            candidate = await policy.act(request)
            if not isinstance(candidate, AgentDecision):
                raise TypeError("policy.act() must return AgentDecision")
            if not isinstance(candidate.raw_output, str):
                raise TypeError("AgentDecision.raw_output must be a string")
            decision = candidate
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            invocation_errors += 1
            message = f"{type(exc).__name__}: {exc}"
            infrastructure_error = {
                "error_type": type(exc).__name__,
                "message": message[:4000],
                "request_id": request.request_id,
                "round_num": observation.round_num,
                "step_index": observation.step_index,
            }
            recorder.mark_run_outcome(
                "infrastructure_error",
                infrastructure_error,
            )
            break

        action = decision.action
        if decision.parser_error is not None:
            parser_errors += 1
        if (
            not isinstance(action, InvalidAction)
            and getattr(action, "action", None) not in observation.available_actions
        ):
            action_name = getattr(action, "action", type(action).__name__)
            action = InvalidAction(
                reason=(
                    f"action {action_name!r} is unavailable; "
                    f"available={list(observation.available_actions)}"
                ),
                raw={"unsupported_action_type": type(action).__name__},
            )
        if isinstance(action, InvalidAction):
            invalid_actions += 1

        transition = environment.step(action)
        if transition.constraint_signals.accepted is True:
            accepted_submissions += 1

        raw_hash = hashlib.sha256(
            decision.raw_output.encode("utf-8")
        ).hexdigest()
        policy_metadata = {
            **to_jsonable(dict(decision.metadata)),
            "agent_request_schema": AGENT_REQUEST_SCHEMA,
            "request_id": request.request_id,
            "policy_kind": policy.kind,
            "policy_name": policy.name,
            "raw_output_sha256": raw_hash,
            "raw_output": decision.raw_output if record_raw_output else None,
            "parser_error": decision.parser_error,
            "history_omitted": history_omitted,
        }
        recorder.record(transition, policy_metadata=policy_metadata)
        history.append(
            {
                "request_id": request.request_id,
                "action": to_jsonable(transition.action),
                "tool_result": to_jsonable(transition.tool_result),
                "reward_components": to_jsonable(
                    transition.reward_components
                ),
                "constraint_signals": to_jsonable(
                    transition.constraint_signals
                ),
                "next_observation": to_jsonable(transition.observation),
            }
        )
        observation = transition.observation
        info = dict(transition.info)

    saved_path: str | None = None
    if trajectory_path is not None:
        saved_path = str(recorder.save(trajectory_path))
    verified = recorder.verify_hash_chain()
    trajectory = recorder.to_dict()
    if infrastructure_error is not None:
        status = "infrastructure_error"
    else:
        status = "terminated" if environment.terminated else "truncated"
    episode_summary = info.get("episode_summary", {})
    reward_summary = (
        dict(episode_summary.get("reward", {}))
        if isinstance(episode_summary, Mapping)
        and isinstance(episode_summary.get("reward"), Mapping)
        else {}
    )
    constraint_summary = (
        dict(episode_summary.get("constraints", {}))
        if isinstance(episode_summary, Mapping)
        and isinstance(episode_summary.get("constraints"), Mapping)
        else {}
    )
    return AgentEpisodeResult(
        task_hash=environment.task.task_hash,
        policy_kind=policy.kind,
        policy_name=policy.name,
        status=status,
        steps=len(recorder.entries),
        final_round=observation.round_num,
        accepted_submissions=accepted_submissions,
        invalid_actions=invalid_actions,
        parser_errors=parser_errors,
        invocation_errors=invocation_errors,
        infrastructure_error=infrastructure_error,
        reward_summary=reward_summary,
        constraint_summary=constraint_summary,
        usage=dict(policy.total_usage),
        trajectory_verified=verified,
        trajectory_path=saved_path,
        trajectory=trajectory,
    )


__all__ = [
    "AGENT_PROMPT_VERSION",
    "AGENT_REQUEST_SCHEMA",
    "AGENT_RUN_SCHEMA",
    "AgentActionError",
    "AgentDecision",
    "AgentEpisodeResult",
    "AgentPolicy",
    "AgentRequest",
    "CommandAgentPolicy",
    "LLMAgentPolicy",
    "create_api_agent_policy",
    "load_agent_system_prompt",
    "parse_environment_action",
    "run_agent_episode",
]
