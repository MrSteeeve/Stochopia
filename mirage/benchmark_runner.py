"""LLM-facing episode runner and pre-registered benchmark metrics."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    oracle_candidate_grid,
)
from .llm import BaseLLMClient
from .products import ProductError, parse_product_spec


STRATEGIES = ("one_shot", "quote_and_revise", "ledger_archive")

RUNNER_SYSTEM_PROMPT = """你是受测的结构化产品设计智能体。市场数据、定价和硬检查只可
通过环境工具获得。每次只输出一个 JSON 对象，可用动作：
{"action":"query_client","topic":"capital|loss_tolerance|maturity|product_types|protection|preferences"}
{"action":"request_quote","product":{...ProductSpec...}}
{"action":"submit_design","quote_id":"Q-...","explanation":"..."}
{"action":"submit_product","product":{...ProductSpec...},"explanation":"..."}
{"action":"skip_round"}

ProductSpec 字段规范（所有字段必填，可空字段用 null）：
- product_type: 只能取 "vanilla_call" | "vanilla_put" | "barrier_call" | "barrier_put" | "autocallable" | "snowball"
- notional: 名义本金（人民币，正数）
- maturity_months: 期限（整数，1-60 个月）
- strike_pct: 行权价/期初价比例，1.0 表示平价（不要填 100）
- barrier_pct: 障碍价比例，null 或正数；必须与 barrier_type 同时设置或同时为 null
- barrier_type: null 或 "knock_in" | "knock_out"（没有其他取值）
- coupon_rate: 年化票息，null 或 [0,5]；autocallable/snowball 必须设置
- participation_rate: 参与率 [0,10]；autocallable/snowball 必须为 1.0
- principal_protected: 布尔值；声明保本但结构不支持会被判违规
- target_client: 客户 id（用回合简报中给出的 id）
- pitch: 一句话推介（字符串）
- hedging_plan: 对冲计划（字符串）

回合规则：每轮最多 3 次 query_client、3 次 request_quote；submit_design 必须引用
本轮未过期的 quote_id；submit_product 等价于报价并立即提交。提交或跳过后本轮结束。
不要声称能够覆盖工具返回的硬约束。"""


def _load_structurer_prompt() -> str:
    """Load the structurer system prompt from the scenario file (consult protocol
    included); fall back to the built-in constant if the file is unavailable."""
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "scenarios" / "mirage_csi" / "prompts" / "structurer.md"
    )
    try:
        return prompt_path.read_text(encoding="utf-8")
    except OSError:
        return RUNNER_SYSTEM_PROMPT


STRUCTURER_SYSTEM_PROMPT = _load_structurer_prompt()


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


@dataclass
class EpisodeTrace:
    episode_id: str
    condition: str
    strategy: str
    rounds: list[RoundTrace]
    usage: dict
    seed: int | None = None


def _protocol_prompt(strategy: str) -> str:
    if strategy == "one_shot":
        return "基线限制：不要查询客户或迭代；用 submit_product 一次性提交。"
    if strategy == "quote_and_revise":
        return "你可以查询客户，并在每轮最多三次询价中修改产品。"
    return (
        "你可以查询和迭代。每次报价后环境会额外提供 constraint_ledger 与 "
        "candidate_archive，请显式利用这些记忆。"
    )


def _forced_prompt_payload(environment: LongHorizonEnvironment) -> dict:
    return {
        "notice": "本轮动作预算已用尽，你还未提交本轮设计。",
        "final_chance": True,
        "allowed_actions": ["submit_design", "skip_round"],
        "available_quotes": [quote.public_payload() for quote, _ in environment.quotes.values()],
        "instruction": (
            "这是本轮最后一次机会，只能返回以下两种 JSON 之一："
            '{"action":"submit_design","quote_id":"...","explanation":"..."}'
            "（quote_id 必须取自上面 available_quotes 中已有的报价），"
            '或 {"action":"skip_round"}。'
            "其他任何动作都无效，本轮将不会成交。"
        ),
    }


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
            trace.submitted = True
            trace.accepted = result["accepted"]
            trace.dealer_margin = result["dealer_margin"]
            trace.hard_failures = [item["check_id"] for item in result["hard_failures"]]
            trace.submission_origin = "forced_prompt"
            trace.submitted_product = asdict(submitted_product)
            trace.submitted_explanation = explanation
            trace.client_brief_snapshot = brief
            trace.submitted_quote_id = quote_id
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
    except (BenchmarkError, ProductError, ValueError, TypeError) as exc:
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
    if strategy not in STRATEGIES:
        raise BenchmarkError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    messages = [{"role": "system", "content": STRUCTURER_SYSTEM_PROMPT + "\n" + _protocol_prompt(strategy)}]
    round_traces: list[RoundTrace] = []

    for round_offset in range(len(environment.snapshots)):
        trace = RoundTrace(round_num=environment.snapshot.round_num)
        # 冻结网格 oracle：在智能体行动前、以当前组合状态计算本轮 margin 上界。
        oracle = oracle_best_quote(
            oracle_candidate_grid(environment.client),
            environment.snapshot,
            environment.client,
            environment.portfolio,
            environment.desk.hard_checks.risk_budget,
        )
        trace.oracle_margin = oracle[1].dealer_margin if oracle is not None else None
        archive = CandidateArchive()
        brief = environment.get_round_brief()
        messages.append({
            "role": "user",
            "content": "【新一轮】\n" + json.dumps(brief, ensure_ascii=False, sort_keys=True),
        })
        last_quote: Quote | None = None

        for action_index in range(1, max_actions_per_round + 1):
            raw = await client.chat(messages, temperature=0.0, seed=seed)
            messages.append({"role": "assistant", "content": raw})
            try:
                action = extract_action_json(raw)
                if not isinstance(action, dict):
                    raise ValueError("action must be a JSON object")
                action_type = action.get("action")
                response: dict

                if action_type == "query_client":
                    if strategy == "one_shot":
                        raise BenchmarkError("one_shot baseline cannot query the client")
                    topic = str(action.get("topic", ""))
                    if environment.has_client_llm():
                        response = await environment.query_client_llm(topic)
                    else:
                        response = environment.query_client(topic)

                elif action_type == "consult":
                    if strategy == "one_shot":
                        raise BenchmarkError("one_shot baseline cannot consult env roles")
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
                    if action_type == "submit_product":
                        explanation = str(action.get("explanation", product.pitch))
                        result = environment.submit_design(quote.quote_id, explanation)
                        response = {"quote": quote_payload, "submission": result}
                        trace.submitted = True
                        trace.accepted = result["accepted"]
                        trace.dealer_margin = result["dealer_margin"]
                        trace.hard_failures = [item["check_id"] for item in result["hard_failures"]]
                        trace.submission_origin = "voluntary"
                        trace.submitted_product = asdict(product)
                        trace.submitted_explanation = explanation
                        trace.client_brief_snapshot = brief
                        trace.submitted_quote_id = quote.quote_id

                elif action_type == "submit_design":
                    quote_id = str(action.get("quote_id", ""))
                    explanation = str(action.get("explanation", ""))
                    result = environment.submit_design(quote_id, explanation)
                    _, submitted_product = environment.quotes[quote_id]
                    response = result
                    trace.submitted = True
                    trace.accepted = result["accepted"]
                    trace.dealer_margin = result["dealer_margin"]
                    trace.hard_failures = [item["check_id"] for item in result["hard_failures"]]
                    trace.submission_origin = "voluntary"
                    trace.submitted_product = asdict(submitted_product)
                    trace.submitted_explanation = explanation
                    trace.client_brief_snapshot = brief
                    trace.submitted_quote_id = quote_id

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
                if strategy == "ledger_archive" and last_quote is not None:
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
            except (BenchmarkError, ProductError, ValueError, TypeError) as exc:
                response = {"error": str(exc)}
                trace.actions.append(ActionTrace(
                    round_num=trace.round_num,
                    action_index=action_index,
                    action="protocol_error",
                    request={"raw": raw[:2000]},
                    response=response,
                ))
                messages.append({"role": "user", "content": json.dumps(response, ensure_ascii=False)})

        # Imputed counterfactual diagnostic only: read-only over this round's
        # archive of previously requested quotes. Never calls submit_design,
        # never mutates portfolio/client state, never feeds a primary metric.
        best = archive.best_feasible()
        trace.imputed_counterfactual_margin = best.quote.dealer_margin if best is not None else None

        if not environment.submitted:
            await _run_forced_prompt(
                environment, client, messages, trace,
                brief=brief, seed=seed, max_actions_per_round=max_actions_per_round,
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
    all_failures = [failure for item in rounds for failure in item.all_quote_failures + item.hard_failures]
    repeated = sum(
        failure in (rounds[index - 1].all_quote_failures + rounds[index - 1].hard_failures)
        for index, item in enumerate(rounds[1:], start=1)
        for failure in item.all_quote_failures + item.hard_failures
    )
    attempts_after_failure = [item for item in rounds if item.quote_failures_before_success > 0]
    revision_success = [item for item in attempts_after_failure if item.accepted]
    result = {
        "episode_id": trace.episode_id,
        "condition": trace.condition,
        "strategy": trace.strategy,
        "rounds": n,
        # hard_feasibility_rate keeps its existing semantics: share of rounds that
        # settled accepted (hard_pass), regardless of submission_origin. It does NOT
        # separate voluntary from forced_prompt; use voluntary/forced_prompt/no
        # submission rates below for that breakdown.
        "hard_feasibility_rate": len(accepted) / n if n else 0.0,
        "submission_rate": sum(item.submitted for item in rounds) / n if n else 0.0,
        "total_dealer_margin": sum(item.dealer_margin for item in voluntary_accepted),
        "mean_dealer_margin": (
            sum(item.dealer_margin for item in voluntary_accepted) / len(voluntary_accepted)
            if voluntary_accepted else 0.0
        ),
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
        recorded = [item.oracle_margin for item in rounds]
        if any(value is not None for value in recorded):
            oracle_margins = recorded
    if oracle_margins is not None:
        if len(oracle_margins) != n:
            raise BenchmarkError("oracle margin list must align with episode rounds")
        # one_step_attainment (formerly oracle_margin_attainment): only voluntary,
        # accepted rounds enter the ratio, matching the voluntary-only primary margin.
        ratios = [
            item.dealer_margin / oracle
            for item, oracle in zip(rounds, oracle_margins)
            if item.accepted and item.submission_origin == "voluntary" and oracle is not None and oracle > 0
        ]
        result["one_step_attainment"] = sum(ratios) / len(ratios) if ratios else None
    return result


def trace_to_dict(trace: EpisodeTrace) -> dict:
    return asdict(trace)
