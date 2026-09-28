"""Run one generated solution script in a subprocess with a hard timeout."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_TAIL = 4000


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _tail(text: str, n: int = _TAIL) -> str:
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


def run_script(code: str, script_path: Path, cwd: Path, timeout_s: float) -> RunResult:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    cwd.mkdir(parents=True, exist_ok=True)
    script_path.write_text(code, encoding="utf-8")
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Generated code gets no model credentials.
    for key in ("MODEL_TOKEN",):
        env.pop(key, None)
    t0 = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=max(1.0, timeout_s))
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        timed_out = True
    return RunResult(
        returncode=proc.returncode if not timed_out else -9,
        stdout=_tail(out or ""),
        stderr=_tail(err or ""),
        seconds=time.monotonic() - t0,
        timed_out=timed_out,
    )
