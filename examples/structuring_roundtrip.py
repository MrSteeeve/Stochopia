"""Public synthetic quote / submission / maturity / independent replay demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

from stochopia.benchmark import MarketSnapshot, RiskBudget
from stochopia.environment import (
    CommandAgentPolicy, EpisodeTask, StochopiaStructurerEnv,
    parse_environment_action, replay_and_evaluate, run_agent_episode,
)
from stochopia.pricing import QuotePolicy, mc_diagnostics, quote_economics, solve_quote_equilibrium
from stochopia.products import ClientProfile


def synthetic_task() -> EpisodeTask:
    """An explicitly synthetic, permissive client; existing gates still apply."""
    snapshots = tuple(
        MarketSnapshot(
            "synthetic-structuring", i + 1, day, "CSI500", spot, 0.02,
            realized_vol_20d=0.20, source="synthetic-fixed-monthly-grid",
        )
        for i, (day, spot) in enumerate(zip(
            (date(2023, 1, 31), date(2023, 2, 28), date(2023, 3, 31), date(2023, 4, 30)),
            (6000.0, 6100.0, 6200.0, 6300.0),
        ))
    )
    return EpisodeTask(
        snapshots=snapshots,
        client=ClientProfile(
            id="synthetic-client", name="Synthetic Client", capital=10_000_000.0,
            max_loss_pct=1.0, min_return_pct=0.02, risk_appetite="moderate",
            max_maturity_months=12, min_hit_prob=0.25,
        ),
        risk_budget=RiskBudget(
            notional=50_000_000.0, net_delta=50_000_000.0,
            gross_delta=100_000_000.0, net_vega=10_000_000.0,
            stress_loss=50_000_000.0,
        ),
        quote_policy=QuotePolicy(diagnostic_paths=256), task_seed=17,
        version="synthetic-structuring-baseline.v1",
    )


def baseline_action(request: dict) -> dict:
    observation = request["observation"]
    if observation["round_num"] != 1:
        return {"action": "skip", "reason": "hold issued contract to maturity"}
    disclosed = observation["disclosed_client"]
    for topic in ("capital", "return_hurdle"):
        if topic not in disclosed:
            return {"action": "ask_client", "topic": topic}
    if observation["quotes"]:
        return {
            "action": "submit_design", "quote_id": observation["quotes"][-1]["quote_id"],
            "explanation": "fixed synthetic baseline",
        }
    return {
        "action": "request_quote",
        "product": {
            "product_type": "vanilla_call", "notional": 1_000_000.0,
            "maturity_months": 3, "strike_pct": 0.95,
            "barrier_pct": None, "barrier_type": None, "coupon_rate": None,
            "participation_rate": 1.0, "principal_protected": True,
            "target_client": "current_client", "pitch": "clear risk disclosure",
            "hedging_plan": "monthly static delta hedge",
        },
    }


async def rollout(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    env = StochopiaStructurerEnv(synthetic_task())
    policy = CommandAgentPolicy([sys.executable, str(Path(__file__).resolve()), "--agent"])
    result = await run_agent_episode(env, policy, trajectory_path=output / "trajectory.json")
    if result.status != "terminated":
        raise RuntimeError(f"rollout failed: {result.trajectory['run_outcome']}")
    evaluation = replay_and_evaluate(
        output / "trajectory.json", output_path=output / "evaluation.json",
    )
    portfolio = env._require_state().backend.portfolio
    summary = {
        "status": result.status, "steps": result.steps,
        "invalid_actions": result.invalid_actions,
        "accepted_submissions": evaluation["constraint_summary"]["accepted_submissions"],
        "replay_verified": evaluation["replay_verified"],
        "closed_positions": len(portfolio.closed_positions),
        "close_reasons": [p.status for p in portfolio.closed_positions],
        "client_available_cash": portfolio.client_account.available_cash,
        "settlements": [e.settlement_amount for e in portfolio.lifecycle_events],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not (result.status == "terminated" and result.invalid_actions == 0
            and summary["accepted_submissions"] == 1
            and summary["close_reasons"] == ["matured"]):
        raise RuntimeError(f"baseline did not complete: {summary}")
    return summary


def pricing_example() -> dict:
    """Compare fixed issuance with automatic pricing, without submitting a trade."""
    env = StochopiaStructurerEnv(synthetic_task())
    env.reset()
    request = {"observation": {
        "round_num": 1, "disclosed_client": {"capital": {}, "return_hurdle": {}}, "quotes": [],
    }}
    env.step(parse_environment_action(baseline_action(request)))
    backend = env._require_state().backend
    quote, product = next(iter(backend.quotes.values()))
    market = backend.snapshot.to_market_state(product.maturity_months)
    diagnostics = mc_diagnostics(product, market, n_paths=256, seed=42)
    kwargs = dict(
        dealer_stress_loss=quote.dealer_hedged_stress_loss,
        post_notional=product.notional, capacity_notional=backend.desk.hard_checks.risk_budget.notional,
        policy=backend.quote_policy, client=backend.client,
    )
    fixed = quote_economics(quote.pricing, diagnostics, product=product, **kwargs)
    # A calculation illustration: the automatic-price case is not issued.
    automatic = replace(product, funding_style="premium_paid", issue_price_pct=None)
    solved = solve_quote_equilibrium(
        automatic, market, backend.client, quote.pricing, diagnostics,
        **{key: value for key, value in kwargs.items() if key != "client"},
    )
    return {
        "notional": product.notional, "fair_value": quote.fair_value,
        "fixed_issue_price": quote.cash_outlay, "fixed_case": fixed.breakdown,
        "auto_case": solved.economics.breakdown,
        "auto_price": solved.economics.client_price, "auto_hurdle": solved.hurdle_hit_prob,
        "auto_converged": solved.converged, "auto_iterations": solved.iterations,
        "auto_price_residual": solved.price_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--pricing-example", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/structuring_roundtrip"))
    args = parser.parse_args()
    if args.agent:
        print(json.dumps(baseline_action(json.load(sys.stdin))))
    elif args.pricing_example:
        print(json.dumps(pricing_example(), indent=2))
    else:
        print(json.dumps(asyncio.run(rollout(args.output)), indent=2))


if __name__ == "__main__":
    main()
