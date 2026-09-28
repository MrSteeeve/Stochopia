"""The solve loop: plan -> code -> (run -> check -> repair)* -> keep the best output.

Budget rules:
- every model call costs one of the unit's requests, whatever its outcome;
- the loop stops early when the checks pass (and one review has been spent on
  any warnings), when requests run out, or when the card's time limit nears;
- after every run the output directory is snapshotted, and the best snapshot
  (fewest errors, then fewest warnings, latest wins ties) is restored at the end.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import contract, prompts
from .checks import Problem, check_outputs, score
from .config import Config
from .executor import RunResult, run_script
from .llm import BudgetExhausted, LLMClient, LLMError
from .workspace import Workspace, prepare

# Seconds a model call is assumed to need when deciding whether one more fits.
_CALL_ALLOWANCE_S = 120.0


@dataclass
class Attempt:
    n: int
    source: str  # "code" | "repair" | "review"
    returncode: int
    timed_out: bool
    run_seconds: float
    errors: int
    warnings: int
    problems: list[str]


@dataclass
class Stats:
    task_id: str
    status: str = "started"  # passed_checks | checks_failing | no_code | llm_error | error
    requests_used: int = 0
    llm_errors: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    best_attempt: int | None = None
    deliverables: list[dict] = field(default_factory=list)
    plan_ok: bool = False
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _clear(directory: Path) -> None:
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _snapshot(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def _restore(snapshot: Path, out_dir: Path) -> None:
    _clear(out_dir)
    for child in snapshot.iterdir():
        target = out_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


class Solver:
    def __init__(self, cfg: Config, task_dir: Path, out_dir: Path):
        self.cfg = cfg
        self.t0 = time.monotonic()
        self.log_dir = cfg.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ws: Workspace = prepare(task_dir, out_dir)
        self.deadline = self.t0 + self.ws.timeout_s - max(cfg.reserve_s, 0.05 * self.ws.timeout_s)
        self.client = LLMClient(cfg, transcript=self.log_dir / "llm_calls.jsonl")
        self.stats = Stats(task_id=self.ws.task_id, notes=list(self.ws.notes))
        self.deliverables = contract.extract(self.ws.instruction, {f.name for f in self.ws.data_files})
        self.plan: dict | None = None

    # -- helpers -------------------------------------------------------------
    def time_left(self) -> float:
        return self.deadline - time.monotonic()

    def can_call(self) -> bool:
        return self.client.remaining > 0 and self.time_left() > _CALL_ALLOWANCE_S

    def ask(self, messages: list[dict[str, str]], *, thinking: bool, purpose: str) -> str | None:
        """One model call; one extra try on a transport/server error if budget allows."""
        if not self.cfg.endpoint or not self.cfg.token:
            self.stats.llm_errors.append(f"{purpose}: MODEL_ENDPOINT or MODEL_TOKEN is not set")
            return None
        for _ in range(2):
            if not self.can_call():
                return None
            try:
                reply = self.client.chat(
                    messages,
                    thinking=thinking,
                    purpose=purpose,
                    timeout_s=min(self.cfg.request_timeout_s, max(30.0, self.time_left() - 30.0)),
                )
                return reply.content
            except BudgetExhausted:
                return None
            except LLMError as exc:
                self.stats.llm_errors.append(f"{purpose}: {exc}"[:500])
                time.sleep(2.0)
        return None

    def save_stats(self) -> None:
        self.stats.requests_used = self.client.used
        self.stats.deliverables = [d.to_dict() for d in self.deliverables]
        self.stats.seconds = time.monotonic() - self.t0
        (self.log_dir / "agent_stats.json").write_text(json.dumps(asdict(self.stats), indent=1), encoding="utf-8")

    # -- stages --------------------------------------------------------------
    def make_plan(self) -> None:
        text = self.ask(prompts.plan_messages(self.ws, self.deliverables), thinking=self.cfg.plan_thinking, purpose="plan")
        plan = prompts.extract_json(text or "")
        if plan:
            self.plan = plan
            self.stats.plan_ok = True
            self.deliverables = contract.merge(self.deliverables, plan.get("deliverables") or [])

    def execute(self, n: int, source: str, code: str) -> tuple[RunResult, list[Problem]]:
        _clear(self.ws.out_dir)
        budget = min(self.cfg.max_script_s, max(20.0, self.time_left() - 30.0))
        run = run_script(
            code,
            self.log_dir / "attempts" / f"{n:02d}_{source}.py",
            cwd=self.log_dir / "work",
            timeout_s=budget,
        )
        problems = check_outputs(self.deliverables, self.ws.out_dir, self.ws.canaries)
        if not run.ok:
            problems.insert(0, Problem("error", "the script did not finish successfully"))
        errs, warns = score(problems)
        self.stats.attempts.append(
            Attempt(n, source, run.returncode, run.timed_out, round(run.seconds, 2), errs, warns, [str(p) for p in problems])
        )
        _snapshot(self.ws.out_dir, self.log_dir / "attempts" / f"{n:02d}_output")
        return run, problems

    def solve(self) -> Stats:
        try:
            self._solve()
        except Exception as exc:  # noqa: BLE001 - never die without leaving stats
            self.stats.status = "error"
            self.stats.notes.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.client.close()
            self.save_stats()
        return self.stats

    def _solve(self) -> None:
        self.make_plan()
        text = self.ask(prompts.code_messages(self.ws, self.deliverables, self.plan), thinking=self.cfg.code_thinking, purpose="code")
        code = prompts.extract_code(text or "")
        best: tuple[tuple[int, int], int] | None = None
        reviewed = False
        n = 0
        source = "code"
        run: RunResult | None = None
        problems: list[Problem] = []
        while True:
            if code:
                n += 1
                run, problems = self.execute(n, source, code)
                key = score(problems)
                if best is None or key <= best[0]:
                    best = (key, n)
                errors, warnings = key
                if errors == 0 and (warnings == 0 or reviewed):
                    break
                if errors and best[0][0] == 0:
                    break  # a review edit broke a passing script: keep the passing one
                if errors == 0 and self.can_call():
                    reviewed = True
                    text = self.ask(
                        prompts.review_messages(self.ws, self.deliverables, self.plan, code, run, problems),
                        thinking=self.cfg.code_thinking,
                        purpose="review",
                    )
                    new_code = prompts.extract_code(text or "")
                    if not new_code:
                        break  # "OK" or nothing usable: keep what passed
                    code, source = new_code, "review"
                    continue
            if not self.can_call():
                break
            text = self.ask(
                prompts.repair_messages(self.ws, self.deliverables, self.plan, code, run if code else None, problems),
                thinking=self.cfg.code_thinking,
                purpose="repair",
            )
            if text is None:
                break
            new_code = prompts.extract_code(text)
            if new_code:
                code, source = new_code, "repair"
            else:
                code, run, problems = "", None, []
        if best is None:
            self.stats.status = "llm_error" if self.stats.llm_errors and not self.stats.attempts else "no_code"
            return
        self.stats.best_attempt = best[1]
        _restore(self.log_dir / "attempts" / f"{best[1]:02d}_output", self.ws.out_dir)
        self.stats.status = "passed_checks" if best[0][0] == 0 else "checks_failing"


def solve(task_dir: Path, out_dir: Path, cfg: Config | None = None) -> Stats:
    return Solver(cfg or Config.from_env(), task_dir, out_dir).solve()
