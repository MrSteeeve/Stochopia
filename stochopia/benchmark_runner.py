"""LLM-facing episode runner and pre-registered benchmark metrics."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .agents import _find_balanced_end

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def _iter_json_candidates(text: str):
    """按出现顺序产出文本中所有可解析的 JSON 对象（围栏块优先）。"""
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        try:
            yield json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    index = 0
    while index < len(text):
        if text[index] == "{":
            end = _find_balanced_end(text, index)
            if end is not None:
                try:
                    yield json.loads(text[index : end + 1])
                    index = end + 1
                    continue
                except json.JSONDecodeError:
                    pass
        index += 1


def extract_action_json(raw: str):
    """从模型回复中提取动作 JSON，对推理模型的思考块保持公平。

    思考文本（<think>...</think>，含截断产生的未闭合块）不参与解析，
    避免推理过程中引用的 JSON 片段被误当成动作；剩余文本里优先选取
    带 "action" 键的对象，没有再退回第一个可解析对象。
    """
    cleaned = _UNCLOSED_THINK_RE.sub("", _THINK_BLOCK_RE.sub("", raw))
    first = None
    for obj in _iter_json_candidates(cleaned):
        if first is None:
            first = obj
        if isinstance(obj, dict) and "action" in obj:
            return obj
    if first is not None:
        return first
    if cleaned.strip() != raw.strip():
        # 思考文本里只信带 action 键的对象，绝不把推理碎片当动作。
        for obj in _iter_json_candidates(raw):
            if isinstance(obj, dict) and "action" in obj:
                return obj
    raise ValueError("未找到可解析的 JSON 对象")
from .benchmark import (
    BenchmarkError,
    CandidateArchive,
    LongHorizonEnvironment,
    Quote,
    WorkflowOutcome,
    constraint_ledger,
    oracle_best_quote,
)
from .llm import BaseLLMClient
from .pricing import PricingError
from .products import ProductError, parse_product_spec


@dataclass(frozen=True)
class ProtocolPolicy:
    """Immutable interaction contract for one benchmark strategy.

    ``action_budget`` is an absolute override when set; ``None`` preserves the
    caller-supplied ``max_actions_per_round``. Protocol repair means that a
    malformed/forbidden response is returned to the model within that normal
    budget. The forced prompt, when enabled, is one additional last-chance turn
    after the normal budget is exhausted.
    """

    strategy: str
    allowed_actions: frozenset[str]
    action_budget: int | None
    protocol_repair: bool
    forced_prompt: bool
    include_ledger_archive: bool
    prompt: str


_INTERACTIVE_ACTIONS = frozenset({
    "query_client", "consult", "request_quote", "submit_design",
    "submit_product", "skip_round",
})

PROTOCOL_POLICIES: Mapping[str, ProtocolPolicy] = MappingProxyType({
    "one_shot": ProtocolPolicy(
        strategy="one_shot",
        allowed_actions=frozenset({"submit_product", "skip_round"}),
        action_budget=1,
        protocol_repair=False,
        forced_prompt=False,
        include_ledger_archive=False,
        prompt=(
            "基线限制：本轮只有一次动作，只能用 submit_product 一次性提交，"
            "或用 skip_round 放弃；不会提供协议纠错轮或最后机会。"
        ),
    ),
    "quote_and_revise": ProtocolPolicy(
        strategy="quote_and_revise",
        allowed_actions=_INTERACTIVE_ACTIONS,
        action_budget=None,
        protocol_repair=True,
        forced_prompt=True,
        include_ledger_archive=False,
        prompt="你可以查询客户，并在每轮最多三次询价中修改产品。",
    ),
    "ledger_archive": ProtocolPolicy(
        strategy="ledger_archive",
        allowed_actions=_INTERACTIVE_ACTIONS,
        action_budget=None,
        protocol_repair=True,
        forced_prompt=True,
        include_ledger_archive=True,
        prompt=(
            "你可以查询和迭代。每次报价后环境会额外提供 constraint_ledger 与 "
            "candidate_archive，请显式利用这些记忆。"
        ),
    ),
})

STRATEGIES = tuple(PROTOCOL_POLICIES)

def _load_structurer_prompt() -> str:
    """Load the exact versioned prompt shipped inside the wheel."""

    prompt = (
        files("stochopia.resources")
        .joinpath("structurer.md")
        .read_text(encoding="utf-8")
    )
    if '"action":"consult"' not in prompt:
        raise RuntimeError("packaged structurer prompt is missing the consult action")
    return prompt


STRUCTURER_SYSTEM_PROMPT = _load_structurer_prompt()
# Compatibility name for integrations that imported the pre-resource constant.
RUNNER_SYSTEM_PROMPT = STRUCTURER_SYSTEM_PROMPT


@dataclass
class ActionTrace:
    round_num: int
    action_index: int
    action: str
    request: dict
    response: dict


@dataclass
class RoundTrace:
    round_num: int
    actions: list[ActionTrace] = field(default_factory=list)
    submitted: bool = False
    accepted: bool = False
    # "voluntary": submitted within the normal action budget.
    # "forced_prompt": submitted only after the last-chance prompt fired.
    # "none": no valid submission this round (explicit skip, exhausted budget, or
    #         a forced-prompt response that was itself invalid/skip).
    submission_origin: str = "none"
    dealer_margin: float = 0.0
    hard_failures: list[str] = field(default_factory=list)
    all_quote_failures: list[str] = field(default_factory=list)
    quote_failures_before_success: int = 0
    # Deprecated serialized compatibility alias. New code should read
    # one_step_frontier_margin; run_episode writes the same value to both.
    oracle_margin: float | None = None
    # Diagnostic only: best archived feasible quote's margin, never executed
    # (never calls environment.submit_design, never touches portfolio/client state,
    # never enters any primary metric).
    imputed_counterfactual_margin: float | None = None
    # Pre-wired for offline judge scoring; populated only when a submission
    # actually goes through (voluntary or forced_prompt).
    submitted_product: dict | None = None
    submitted_explanation: str = ""
    client_brief_snapshot: dict | None = None
    submitted_quote_id: str | None = None
    # v2 LLM dialogue-layer bookkeeping (all zero / None in the deterministic
    # env_agents=None fallback, so primary metrics are byte-for-byte unchanged).
    consult_count: int = 0
    degraded_consult_count: int = 0
    workflow: WorkflowOutcome | None = None
    # New fields are appended to preserve the positional constructor layout of
    # legacy RoundTrace consumers.
    # Deterministic best hard-executable margin over the exact environment
    # domain and quote policy at the start of this round.
    one_step_frontier_margin: float | None = None
    # Two-gate deterministic settlement decomposition. These remain False when
    # there was no submission; accepted is their conjunction after submission.
    hard_executable: bool = False
    client_contract_pass: bool = False
    contract_failures: list[str] = field(default_factory=list)
    # Canonical diagnostic events.  Compatibility lists above are retained for
    # old artifacts, but metrics deduplicate on this event identity.
    failure_events: list[dict] = field(default_factory=list)


@dataclass
class EpisodeTrace:
    episode_id: str
    condition: str
    strategy: str
    rounds: list[RoundTrace]
    usage: dict
    seed: int | None = None


def _protocol_prompt(policy: ProtocolPolicy) -> str:
    return policy.prompt


def _forced_prompt_payload(environment: LongHorizonEnvironment) -> dict:
    return {
        "notice": "本轮动作预算已用尽，你还未提交本轮设计。",
        "final_chance": True,
        "allowed_actions": ["submit_design", "skip_round"],
        "available_quotes": [
            environment.public_quote_payload(quote)
            for quote, _ in environment.quotes.values()
        ],
        "instruction": (
            "这是本轮最后一次机会，只能返回以下两种 JSON 之一："
            '{"action":"submit_design","quote_id":"...","explanation":"..."}'
            "（quote_id 必须取自上面 available_quotes 中已有的报价），"
            '或 {"action":"skip_round"}。'
            "其他任何动作都无效，本轮将不会成交。"
        ),
    }


def _record_submission_result(
    trace: RoundTrace,
    result: dict,
    *,
    origin: str,
    product: dict,
    explanation: str,
    brief: dict,
    quote_id: str,
) -> None:
    """Copy the deterministic two-gate settlement into a round trace."""
    trace.submitted = True
    trace.accepted = bool(result["accepted"])
    trace.hard_executable = bool(result["hard_executable"])
    trace.client_contract_pass = bool(result["client_contract_pass"])
    trace.dealer_margin = float(result["dealer_margin"])
    trace.hard_failures = [item["check_id"] for item in result["hard_failures"]]
    trace.contract_failures = [item["check_id"] for item in result["contract_failures"]]
    trace.submission_origin = origin
    trace.submitted_product = product
    trace.submitted_explanation = explanation
    trace.client_brief_snapshot = brief
    trace.submitted_quote_id = quote_id
    existing = {
        (
            event.get("quote_id"),
            event.get("check_id"),
            event.get("event_type"),
        )
        for event in trace.failure_events
    }
    for check_id in trace.hard_failures:
        key = (quote_id, check_id, "hard_check")
        if key not in existing:
            trace.failure_events.append({
                "round": trace.round_num,
                "quote_id": quote_id,
                "check_id": check_id,
                "event_type": "hard_check",
            })
            existing.add(key)


async def _run_forced_prompt(
    environment: LongHorizonEnvironment,
    client: BaseLLMClient,
    messages: list[dict],
    trace: "RoundTrace",
    *,
    brief: dict,
    seed: int | None,
    max_actions_per_round: int,
) -> None:
    """Give the tested model one last-chance turn after its action budget is spent.

    Only submit_design (referencing an already-quoted quote_id) or skip_round are
    accepted. A successful submit_design here is tagged submission_origin="forced_prompt";
    anything else (skip, invalid action, parse/validation error) leaves the round
    unsubmitted with submission_origin="none".
    """
    payload = _forced_prompt_payload(environment)
    messages.append({
        "role": "user",
        "content": "【最后机会】\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True),
    })
    raw = await client.chat(messages, temperature=0.0, seed=seed)
    messages.append({"role": "assistant", "content": raw})
    forced_action_index = max_actions_per_round + 1
    try:
        action = extract_action_json(raw)
        if not isinstance(action, dict):
            raise ValueError("action must be a JSON object")
        action_type = action.get("action")
        if action_type not in {"submit_design", "skip_round"}:
            raise BenchmarkError(
                f"forced last-chance prompt only allows submit_design or skip_round, got {action_type!r}"
            )
        if action_type == "submit_design":
            quote_id = str(action.get("quote_id", ""))
            explanation = str(action.get("explanation", ""))
            result = environment.submit_design(quote_id, explanation)
            _, submitted_product = environment.quotes[quote_id]
            response = result
            _record_submission_result(
                trace,
                result,
                origin="forced_prompt",
                product=asdict(submitted_product),
                explanation=explanation,
                brief=brief,
                quote_id=quote_id,
            )
        else:
            response = {"skipped": True}
            trace.submitted = False
            trace.accepted = False
            trace.submission_origin = "none"
            environment.submitted = True
        trace.actions.append(ActionTrace(
            round_num=trace.round_num,
            action_index=forced_action_index,
            action=f"forced_prompt:{action_type}",
            request=action,
            response=response,
        ))
    except (BenchmarkError, ProductError, PricingError, ArithmeticError, ValueError, TypeError) as exc:
        response = {"error": str(exc)}
        trace.actions.append(ActionTrace(
            round_num=trace.round_num,
            action_index=forced_action_index,
            action="forced_prompt_error",
            request={"raw": raw[:2000]},
            response=response,
        ))
        trace.submission_origin = "none"


async def run_episode(
    environment: LongHorizonEnvironment,
    client: BaseLLMClient,
    *,
    strategy: str = "quote_and_revise",
    max_actions_per_round: int = 9,
    seed: int | None = None,
) -> EpisodeTrace:
    """Run one frozen episode against the strict tool protocol."""
    policy = PROTOCOL_POLICIES.get(strategy)
    if policy is None:
        raise BenchmarkError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    action_budget = policy.action_budget if policy.action_budget is not None else max_actions_per_round
    messages = [{"role": "system", "content": STRUCTURER_SYSTEM_PROMPT + "\n" + _protocol_prompt(policy)}]
    round_traces: list[RoundTrace] = []

    for round_offset in range(len(environment.snapshots)):
        trace = RoundTrace(round_num=environment.snapshot.round_num)
        # One-step frontier: compute before the model acts, over exactly the same
        # frozen domain and quote economics used by this environment.
        frontier = oracle_best_quote(
            environment.domain,
            environment.snapshot,
            environment.client,
            environment.portfolio,
            environment.desk.hard_checks.risk_budget,
            policy=environment.quote_policy,
        )
        frontier_margin = frontier[1].dealer_margin if frontier is not None else None
        trace.one_step_frontier_margin = frontier_margin
        trace.oracle_margin = frontier_margin  # deprecated compatibility alias
        archive = CandidateArchive()
        brief = environment.get_round_brief()
        messages.append({
            "role": "user",
            "content": "【新一轮】\n" + json.dumps(brief, ensure_ascii=False, sort_keys=True),
        })
        last_quote: Quote | None = None

        for action_index in range(1, action_budget + 1):
            raw = await client.chat(messages, temperature=0.0, seed=seed)
            messages.append({"role": "assistant", "content": raw})
            try:
                action = extract_action_json(raw)
                if not isinstance(action, dict):
                    raise ValueError("action must be a JSON object")
                action_type = action.get("action")
                response: dict

                if action_type not in policy.allowed_actions:
                    allowed = ", ".join(sorted(policy.allowed_actions))
                    raise BenchmarkError(
                        f"{policy.strategy} protocol only allows [{allowed}], got {action_type!r}"
                    )

                if action_type == "query_client":
                    topic = str(action.get("topic", ""))
                    if environment.has_client_llm():
                        response = await environment.query_client_llm(topic)
                    else:
                        response = environment.query_client(topic)

                elif action_type == "consult":
                    draft = action.get("draft")
                    response = await environment.consult(
                        str(action.get("role", "")),
                        str(action.get("message", "")),
                        draft if isinstance(draft, dict) else None,
                    )
                    trace.consult_count += 1
                    if response.get("degraded"):
                        trace.degraded_consult_count += 1

                elif action_type in {"request_quote", "submit_product"}:
                    product = parse_product_spec(action.get("product"))
                    quote_payload = environment.request_quote(product)
                    quote, _ = environment.quotes[quote_payload["quote_id"]]
                    archive.add(product, quote)
                    last_quote = quote
                    response = quote_payload
                    if not quote.hard_pass:
                        trace.quote_failures_before_success += 1
                        trace.all_quote_failures.extend(
                            item.check_id for item in quote.checks if not item.passed
                        )
                        for item in quote.checks:
                            if item.passed:
                                continue
                            event = {
                                "round": trace.round_num,
                                "quote_id": quote.quote_id,
                                "check_id": item.check_id,
                                "event_type": "hard_check",
                            }
                            key = (
                                event["quote_id"],
                                event["check_id"],
                                event["event_type"],
                            )
                            if key not in {
                                (
                                    row.get("quote_id"),
                                    row.get("check_id"),
                                    row.get("event_type"),
                                )
                                for row in trace.failure_events
                            }:
                                trace.failure_events.append(event)
                    if action_type == "submit_product":
                        explanation = str(action.get("explanation", product.pitch))
                        result = environment.submit_design(quote.quote_id, explanation)
                        response = {"quote": quote_payload, "submission": result}
                        _record_submission_result(
                            trace,
                            result,
                            origin="voluntary",
                            product=asdict(product),
                            explanation=explanation,
                            brief=brief,
                            quote_id=quote.quote_id,
                        )

                elif action_type == "submit_design":
                    quote_id = str(action.get("quote_id", ""))
                    explanation = str(action.get("explanation", ""))
                    result = environment.submit_design(quote_id, explanation)
                    _, submitted_product = environment.quotes[quote_id]
                    response = result
                    _record_submission_result(
                        trace,
                        result,
                        origin="voluntary",
                        product=asdict(submitted_product),
                        explanation=explanation,
                        brief=brief,
                        quote_id=quote_id,
                    )

                elif action_type == "skip_round":
                    response = {"skipped": True}
                    trace.submitted = False
                    trace.accepted = False
                    trace.submission_origin = "none"
                    environment.submitted = True

                else:
                    raise BenchmarkError(f"unknown benchmark action: {action_type!r}")

                trace.actions.append(ActionTrace(
                    round_num=trace.round_num,
                    action_index=action_index,
                    action=str(action_type),
                    request=action,
                    response=response,
                ))
                tool_payload: dict = {"tool_result": response}
                if policy.include_ledger_archive and last_quote is not None:
                    tool_payload["constraint_ledger"] = constraint_ledger(
                        last_quote, environment.portfolio, environment.desk.hard_checks.risk_budget
                    )
                    tool_payload["candidate_archive"] = json.loads(archive.prompt_summary())
                messages.append({
                    "role": "user",
                    "content": json.dumps(tool_payload, ensure_ascii=False, sort_keys=True),
                })
                if action_type in {"submit_design", "submit_product", "skip_round"}:
                    break
            except (
                BenchmarkError,
                ProductError,
                PricingError,
                ArithmeticError,
                ValueError,
                TypeError,
            ) as exc:
                response = {"error": str(exc)}
                trace.actions.append(ActionTrace(
                    round_num=trace.round_num,
                    action_index=action_index,
                    action="protocol_error",
                    request={"raw": raw[:2000]},
                    response=response,
                ))
                if policy.protocol_repair:
                    messages.append({"role": "user", "content": json.dumps(response, ensure_ascii=False)})

        # Imputed counterfactual diagnostic only: read-only over this round's
        # archive of previously requested quotes. Never calls submit_design,
        # never mutates portfolio/client state, never feeds a primary metric.
        best = archive.best_feasible()
        trace.imputed_counterfactual_margin = best.quote.dealer_margin if best is not None else None

        if policy.forced_prompt and not environment.submitted:
            await _run_forced_prompt(
                environment, client, messages, trace,
                brief=brief, seed=seed, max_actions_per_round=action_budget,
            )
        if not environment.submitted:
            environment.submitted = True

        # Second-layer workflow review: only when env roles are wired and an
        # actual submission happened. Never affects the primary settlement above.
        if (
            environment.env_agents is not None
            and trace.submitted
            and trace.submitted_quote_id is not None
        ):
            trace.workflow = await environment.workflow_review(
                trace.submitted_quote_id, trace.submitted_explanation,
            )

        round_traces.append(trace)
        if round_offset < len(environment.snapshots) - 1:
            environment.advance_round()

    return EpisodeTrace(
        episode_id=environment.snapshots[0].episode_id,
        condition=environment.condition.id,
        strategy=strategy,
        rounds=round_traces,
        usage=dict(client.total_usage),
        seed=seed,
    )


def compute_metrics(trace: EpisodeTrace, oracle_margins: list[float | None] | None = None) -> dict:
    """Compute separate pre-registered outcomes; never collapse to one score."""
    rounds = trace.rounds
    n = len(rounds)
    accepted = [item for item in rounds if item.accepted]
    # Primary economics only ever count voluntary submissions: the model acted within
    # its normal action budget. forced_prompt (last-chance nudge after budget exhaustion)
    # and none (no valid submission at all) are reported separately and never pooled in.
    voluntary_accepted = [item for item in accepted if item.submission_origin == "voluntary"]
    forced_prompt_accepted = [item for item in accepted if item.submission_origin == "forced_prompt"]
    submitted_rounds = [item for item in rounds if item.submitted]
    hard_passed_submissions = [item for item in submitted_rounds if item.hard_executable]
    settlement_acceptance_rate = len(accepted) / n if n else 0.0
    hard_execution_rate = len(hard_passed_submissions) / n if n else 0.0
    mean_voluntary_accepted_margin = (
        sum(item.dealer_margin for item in voluntary_accepted) / len(voluntary_accepted)
        if voluntary_accepted else 0.0
    )
    def round_failure_ids(item: RoundTrace) -> list[str]:
        if item.failure_events:
            return [str(event["check_id"]) for event in item.failure_events]
        # Old artifacts have no event identity.  At minimum, do not double
        # count the same submitted quote/check through both compatibility lists.
        return list(dict.fromkeys(item.all_quote_failures + item.hard_failures))

    failures_by_round = [round_failure_ids(item) for item in rounds]
    all_failures = [
        failure for failures in failures_by_round for failure in failures
    ]
    repeated = sum(
        len(set(failures_by_round[index - 1]) & set(failures_by_round[index]))
        for index in range(1, len(failures_by_round))
    )
    attempts_after_failure = [item for item in rounds if item.quote_failures_before_success > 0]
    revision_success = [item for item in attempts_after_failure if item.accepted]
    result = {
        "episode_id": trace.episode_id,
        "condition": trace.condition,
        "strategy": trace.strategy,
        "rounds": n,
        # Always-defined primary estimand: a round with no submission is not a
        # hard-executable outcome. The conditional companion below is diagnostic.
        "hard_execution_rate": hard_execution_rate,
        "hard_execution_rate_given_submission": (
            len(hard_passed_submissions) / len(submitted_rounds)
            if submitted_rounds else None
        ),
        "contract_acceptance_rate_given_hard_pass": (
            sum(item.client_contract_pass for item in hard_passed_submissions)
            / len(hard_passed_submissions)
            if hard_passed_submissions else None
        ),
        "settlement_acceptance_rate": settlement_acceptance_rate,
        # Deprecated compatibility alias. Despite its old name, this has always
        # meant final settlement acceptance per round, not hard execution alone.
        "hard_feasibility_rate": settlement_acceptance_rate,
        "submission_rate": sum(item.submitted for item in rounds) / n if n else 0.0,
        "total_dealer_margin": sum(item.dealer_margin for item in voluntary_accepted),
        "mean_dealer_margin_per_voluntary_accepted_trade": mean_voluntary_accepted_margin,
        # Compatibility alias for the formerly ambiguous metric name.
        "mean_dealer_margin": mean_voluntary_accepted_margin,
        "forced_prompt_margin": sum(item.dealer_margin for item in forced_prompt_accepted),
        "voluntary_submission_rate": (
            sum(item.submission_origin == "voluntary" for item in rounds) / n if n else 0.0
        ),
        "forced_prompt_rate": (
            sum(item.submission_origin == "forced_prompt" for item in rounds) / n if n else 0.0
        ),
        "no_submission_rate": (
            sum(item.submission_origin == "none" for item in rounds) / n if n else 0.0
        ),
        "quote_failures": sum(item.quote_failures_before_success for item in rounds),
        "repeated_hard_violations": repeated,
        "revision_success_rate": len(revision_success) / len(attempts_after_failure) if attempts_after_failure else None,
        "failure_counts": {failure: all_failures.count(failure) for failure in sorted(set(all_failures))},
        "seed": trace.seed,
    }
    # v2 secondary (dialogue-layer) metrics. These are additive; every primary
    # metric above keeps its exact env_agents=None value. workflow_deal_rate and
    # llm_action_vs_formal_truth are None when no round had a workflow review;
    # degraded_consult_rate is None when no consult happened.
    workflow_rounds = [item for item in rounds if item.workflow is not None]
    if workflow_rounds:
        result["workflow_deal_rate"] = (
            sum(item.workflow.workflow_deal for item in workflow_rounds) / len(workflow_rounds)
        )
        result["llm_action_vs_formal_truth"] = (
            sum(bool(item.workflow.workflow_deal) != bool(item.accepted) for item in workflow_rounds)
            / len(workflow_rounds)
        )
    else:
        result["workflow_deal_rate"] = None
        result["llm_action_vs_formal_truth"] = None
    total_consults = sum(item.consult_count for item in rounds)
    degraded_consults = sum(item.degraded_consult_count for item in rounds)
    result["degraded_consult_rate"] = (
        degraded_consults / total_consults if total_consults else None
    )
    if oracle_margins is None:
        recorded = [
            item.one_step_frontier_margin
            if item.one_step_frontier_margin is not None
            else item.oracle_margin  # deprecated trace compatibility field
            for item in rounds
        ]
        if any(value is not None for value in recorded):
            oracle_margins = recorded
    if oracle_margins is not None:
        if len(oracle_margins) != n:
            raise BenchmarkError("oracle margin list must align with episode rounds")
        # one_step_attainment (formerly oracle_margin_attainment): only voluntary,
        # accepted rounds enter the ratio, matching the voluntary-only primary margin.
        ratios: list[float] = []
        for item, frontier_margin in zip(rounds, oracle_margins):
            if not (
                item.accepted
                and item.submission_origin == "voluntary"
                and frontier_margin is not None
                and frontier_margin > 0
            ):
                continue
            ratio = item.dealer_margin / frontier_margin
            if ratio > 1.0 + 1e-9:
                raise BenchmarkError(
                    "voluntary accepted trade exceeds the one-step frontier "
                    f"in round {item.round_num}: {ratio:.12g} > 1 + 1e-9"
                )
            ratios.append(ratio)
        result["one_step_attainment"] = sum(ratios) / len(ratios) if ratios else None
    return result


def trace_to_dict(trace: EpisodeTrace) -> dict:
    return asdict(trace)
