"""Runtime configuration, read from the environment the harness injects.

The evaluation harness sets MODEL_ENDPOINT (origin, no path), MODEL_NAME and
MODEL_TOKEN. Everything prefixed T1A_ is ours and optional.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Config:
    endpoint: str
    model: str
    token: str
    # House allowance: 25 admitted requests per unit, 4,000 output tokens per request.
    max_requests: int = 25
    max_output_tokens: int = 4000
    temperature: float = 0.0
    seed: int = 0
    # Nemotron thinks by default; thinking tokens may share the 4,000-token output cap,
    # so code-writing calls default to thinking off.
    plan_thinking: bool = True
    code_thinking: bool = False
    request_timeout_s: float = 600.0
    # Seconds kept free before the card's agent timeout.
    reserve_s: float = 90.0
    # Upper bound on one execution of a generated script.
    max_script_s: float = 900.0
    log_dir: Path = Path("/tmp/t1agent")

    @property
    def chat_url(self) -> str:
        return self.endpoint.rstrip("/") + "/v1/chat/completions"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            endpoint=os.environ.get("MODEL_ENDPOINT", ""),
            model=os.environ.get("MODEL_NAME", "house"),
            token=os.environ.get("MODEL_TOKEN", ""),
            max_requests=_int("T1A_MAX_REQUESTS", 25),
            max_output_tokens=_int("T1A_MAX_OUTPUT_TOKENS", 4000),
            temperature=_float("T1A_TEMPERATURE", 0.0),
            seed=_int("T1A_SEED", 0),
            plan_thinking=_bool("T1A_PLAN_THINKING", True),
            code_thinking=_bool("T1A_CODE_THINKING", False),
            request_timeout_s=_float("T1A_REQUEST_TIMEOUT_S", 600.0),
            reserve_s=_float("T1A_RESERVE_S", 90.0),
            max_script_s=_float("T1A_MAX_SCRIPT_S", 900.0),
            log_dir=Path(os.environ.get("T1A_LOG_DIR", "/tmp/t1agent")),
        )
