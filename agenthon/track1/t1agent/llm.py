"""OpenAI-compatible client for the House route, with a hard request budget.

A request is charged before it is sent: the House route charges admitted
requests even when the upstream fails, so the local counter must never
under-count. There are no automatic retries here; the agent decides.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Config

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class BudgetExhausted(Exception):
    pass


class LLMError(Exception):
    pass


@dataclass
class Reply:
    content: str
    reasoning: str
    finish_reason: str
    usage: dict[str, Any]
    seconds: float


class LLMClient:
    def __init__(self, cfg: Config, transcript: Path | None = None):
        self.cfg = cfg
        self.used = 0
        self.transcript = transcript
        self._http = httpx.Client(timeout=cfg.request_timeout_s)

    @property
    def remaining(self) -> int:
        return self.cfg.max_requests - self.used

    def close(self) -> None:
        self._http.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        thinking: bool,
        purpose: str,
        timeout_s: float | None = None,
    ) -> Reply:
        if not self.cfg.endpoint or not self.cfg.token:
            raise LLMError("MODEL_ENDPOINT or MODEL_TOKEN is not set")
        if self.remaining <= 0:
            raise BudgetExhausted(f"all {self.cfg.max_requests} requests used")
        self.used += 1
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "max_tokens": self.cfg.max_output_tokens,
            "n": 1,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        headers = {"Authorization": f"Bearer {self.cfg.token}"}
        t0 = time.monotonic()
        record: dict[str, Any] = {"n": self.used, "purpose": purpose, "thinking": thinking}
        try:
            resp = self._http.post(
                self.cfg.chat_url,
                json=body,
                headers=headers,
                timeout=timeout_s or self.cfg.request_timeout_s,
            )
        except httpx.HTTPError as exc:
            record.update(error=f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - t0)
            self._log(record, messages, None)
            raise LLMError(record["error"]) from exc
        seconds = time.monotonic() - t0
        if resp.status_code != 200:
            record.update(error=f"HTTP {resp.status_code}: {resp.text[:500]}", seconds=seconds)
            self._log(record, messages, None)
            raise LLMError(record["error"])
        try:
            data = resp.json()
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            raw = msg.get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            record.update(error=f"bad response: {resp.text[:500]}", seconds=seconds)
            self._log(record, messages, None)
            raise LLMError(record["error"]) from exc
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        thought = "\n".join(_THINK.findall(raw))
        content = _THINK.sub("", raw)
        if "<think>" in content:  # unterminated thinking: the output cap cut it off
            thought += content
            content = ""
        reply = Reply(
            content=content.strip(),
            reasoning=(reasoning + thought).strip(),
            finish_reason=str(choice.get("finish_reason") or ""),
            usage=data.get("usage") or {},
            seconds=seconds,
        )
        record.update(seconds=seconds, finish_reason=reply.finish_reason, usage=reply.usage)
        self._log(record, messages, reply)
        return reply

    def _log(self, record: dict[str, Any], messages: list[dict[str, str]], reply: Reply | None) -> None:
        if self.transcript is None:
            return
        entry = dict(record)
        entry["messages"] = messages
        if reply is not None:
            entry["content"] = reply.content
            entry["reasoning"] = reply.reasoning
        self.transcript.parent.mkdir(parents=True, exist_ok=True)
        with self.transcript.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
