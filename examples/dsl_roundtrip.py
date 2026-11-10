"""Handwritten DSL fixtures through the public command-policy interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from structurerdsl.program import Program

from stochopia.adapters.structurer_dsl import DSL_CONTRACT_VERSION
from stochopia.environment import CommandAgentPolicy, replay_and_evaluate, run_agent_episode
from stochopia.environment.dsl import DSL_AGENT_PROMPT, StochopiaDSLEnv
from examples.structuring_roundtrip import synthetic_task


BODIES = {
    "vanilla": "(premium (pay (max (lin 1 (terminal (spot u)) -0.95) 0) maturity))",
    "spread": "(premium (pay (clip (lin 1 (terminal (spot u)) -0.95) 0 0.10) maturity))",
    "asian": "(premium (pay (max (lin 1 (avg_arith (spot u) obs) -0.95) 0) maturity))",
    "range": "(funded 1 $issue_price (pay (mul_const 0.01 (count_within (spot u) 0.9 1.1 obs)) maturity))",
    "autocall": "(funded 1 $issue_price (terminate_if (gt (spot u) 1.01) obs (mul_const 0.12 time)))",
}


def fixture_payload(case: str) -> dict:
    program = Program.from_sexpr(
        f"(contract (meta (underlyings u) (maturity 3) (schedules (obs 1 2 3))) {BODIES[case]})"
    )
    return {"version": DSL_CONTRACT_VERSION, "program": program.to_json(), "notional": 1_000_000.0}


def action(request: dict, case: str) -> dict:
    observation = request["observation"]
    if observation["round_num"] != 1:
        return {"action": "skip", "reason": "hold the issued contract"}
    for topic in ("capital", "return_hurdle"):
        if topic not in observation["disclosed_client"]:
            return {"action": "ask_client", "topic": topic}
    if observation["quotes"]:
        return {"action": "submit_design", "quote_id": observation["quotes"][-1]["quote_id"]}
    return {"action": "request_dsl_quote", "contract": fixture_payload(case)}


async def rollout(output: Path, case: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    # Permissive public synthetic fixture; economic gates still run unchanged.
    task = replace(synthetic_task(), version="synthetic-dsl-design.v1")
    env = StochopiaDSLEnv(task, dsl_paths=4096)
    policy = CommandAgentPolicy(
        [sys.executable, str(Path(__file__).resolve()), "--agent", "--case", case],
        system_prompt=DSL_AGENT_PROMPT,
    )
    result = await run_agent_episode(env, policy, trajectory_path=output / "trajectory.json")
    if result.status != "terminated":
        raise RuntimeError(f"DSL rollout failed: {result.infrastructure_error}")
    evaluation = replay_and_evaluate(output / "trajectory.json", output_path=output / "evaluation.json")
    portfolio = env._require_state().backend.portfolio
    summary = {
        "case": case, "status": result.status, "steps": result.steps,
        "accepted_submissions": result.accepted_submissions,
        "invalid_actions": result.invalid_actions, "replay_verified": evaluation["replay_verified"],
        "issue_payments": [cf.amount for cf in portfolio.cashflow_ledger if cf.event_type == "issuance"],
        "settlements": [e.settlement_amount for e in portfolio.lifecycle_events],
        "close_reasons": [p.status for p in portfolio.closed_positions],
        "client_available_cash": portfolio.client_account.available_cash,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if result.accepted_submissions != 1 or result.invalid_actions or len(portfolio.lifecycle_events) != 1:
        raise RuntimeError(f"DSL fixture did not trade and settle: {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--case", choices=BODIES, default="autocall")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.agent:
        print(json.dumps(action(json.load(sys.stdin), args.case)))
    else:
        output = args.output or Path("artifacts/dsl-integration-20261009") / args.case
        print(json.dumps(asyncio.run(rollout(output, args.case)), indent=2))


if __name__ == "__main__":
    main()
