"""Tests for the v3 external CLI/API agent boundary."""

from __future__ import annotations

import json
import shlex
import sys
from datetime import date
from pathlib import Path

import pytest

from mirage.benchmark import MarketSnapshot, RiskBudget
from mirage.benchmark_cli import main
from mirage.environment import (
    AgentActionError,
    CommandAgentPolicy,
    EpisodeTask,
    LLMAgentPolicy,
    MirageStructurerEnv,
    Skip,
    create_api_agent_policy,
    parse_environment_action,
    run_agent_episode,
    verify_trajectory,
)
from mirage.llm import LLMError, MockLLMClient
from mirage.products import ClientProfile


def _task() -> EpisodeTask:
    snapshots = tuple(
        MarketSnapshot(
            episode_id="agent-test",
            round_num=round_num,
            as_of=date(2023, round_num, 28),
            underlying="CSI500",
            spot=6000.0 + round_num,
            risk_free_rate=0.02,
            realized_vol_20d=0.20,
            source="test",
        )
        for round_num in (1, 2)
    )
    return EpisodeTask(
        snapshots=snapshots,
        client=ClientProfile(
            id="hidden-client",
            name="Hidden Client",
            capital=10_000_000.0,
            max_loss_pct=1.0,
            min_return_pct=0.02,
            risk_appetite="moderate",
        ),
        risk_budget=RiskBudget(
            notional=50_000_000.0,
            net_delta=50_000_000.0,
            gross_delta=100_000_000.0,
            net_vega=10_000_000.0,
            stress_loss=50_000_000.0,
        ),
        task_seed=19,
    )


def test_action_parser_returns_typed_actions_and_rejects_unknown_fields():
    action = parse_environment_action(
        '{"action":"ask_client","topic":"capital"}'
    )
    assert action.action == "ask_client"
    assert action.topic == "capital"

    skipped = parse_environment_action({"action": "skip", "reason": "done"})
    assert skipped == Skip(reason="done")

    with pytest.raises(AgentActionError, match="unknown fields"):
        parse_environment_action(
            {"action": "skip", "reason": "done", "secret": True}
        )


@pytest.mark.asyncio
async def test_api_policy_runs_same_v3_trajectory_without_privileged_state(tmp_path):
    policy = LLMAgentPolicy(
        MockLLMClient(
            [
                '{"action":"skip","reason":"round one"}',
                '{"action":"skip","reason":"round two"}',
            ]
        ),
        name="mock-api-agent",
    )
    output = tmp_path / "trajectory.json"

    result = await run_agent_episode(
        MirageStructurerEnv(_task()),
        policy,
        trajectory_path=output,
    )

    assert result.status == "terminated"
    assert result.steps == 2
    assert result.invalid_actions == 0
    assert result.trajectory_verified is True
    assert verify_trajectory(output)
    first_observation = result.trajectory["entries"][0]["transition"][
        "previous_observation"
    ]
    assert "privileged_state" not in first_observation
    assert result.usage["calls"] == 2


@pytest.mark.asyncio
async def test_command_policy_uses_versioned_json_stdio_protocol(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "request = json.load(sys.stdin)",
                "assert request['schema'] == 'mirage.agent-request.v1'",
                "assert 'system_prompt' in request",
                "assert 'task_manifest' not in request",
                "assert 'privileged_state' not in request['observation']",
                "print(json.dumps({'action': 'skip', 'reason': 'cli smoke'}))",
            ]
        ),
        encoding="utf-8",
    )
    policy = CommandAgentPolicy([sys.executable, str(script)])

    result = await run_agent_episode(MirageStructurerEnv(_task()), policy)

    assert result.status == "terminated"
    assert result.steps == 2
    assert result.invalid_actions == 0
    assert result.usage == {"calls": 2}
    assert all(
        entry["policy_metadata"]["policy_kind"] == "command"
        for entry in result.trajectory["entries"]
    )


@pytest.mark.asyncio
async def test_malformed_agent_output_is_recorded_and_bounded():
    policy = LLMAgentPolicy(
        MockLLMClient(["not an action"]),
        name="malformed-agent",
    )

    result = await run_agent_episode(
        MirageStructurerEnv(_task(), max_steps_per_round=2),
        policy,
    )

    assert result.status == "truncated"
    assert result.steps == 2
    assert result.invalid_actions == 2
    assert result.parser_errors == 2
    assert result.invocation_errors == 0
    assert result.trajectory_verified is True


def test_direct_api_policy_requires_key_but_never_accepts_key_value(monkeypatch):
    monkeypatch.delenv("MIRAGE_TEST_API_KEY", raising=False)
    with pytest.raises(LLMError, match="MIRAGE_TEST_API_KEY"):
        create_api_agent_policy(
            provider="openai-compatible",
            base_url="https://example.test/v1",
            model="test-model",
            api_key_env="MIRAGE_TEST_API_KEY",
        )

    monkeypatch.setenv("MIRAGE_TEST_API_KEY", "top-secret")
    policy = create_api_agent_policy(
        provider="openai-compatible",
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="MIRAGE_TEST_API_KEY",
    )
    assert policy.client.config.api_key_env == "MIRAGE_TEST_API_KEY"
    assert "top-secret" not in policy.name


def test_test_agent_cli_writes_verified_trajectory_and_summary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "skip_agent.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({'action': 'skip', 'reason': "
        "f\"round {request['observation']['round_num']}\"}))\n",
        encoding="utf-8",
    )
    trajectory = tmp_path / "cli-trajectory.json"
    summary = tmp_path / "cli-summary.json"
    command = " ".join(
        (shlex.quote(sys.executable), shlex.quote(str(script)))
    )

    exit_code = main(
        [
            "test-agent",
            str(root / "scenarios/mirage_csi/market_snapshots.example.csv"),
            "--episode",
            "SYNTHETIC_CSI500_DEMO",
            "--client-json",
            str(root / "scenarios/mirage_csi/client.example.json"),
            "--risk-budget-json",
            str(root / "scenarios/mirage_csi/risk_budget.example.json"),
            "--agent-command",
            command,
            "--output",
            str(trajectory),
            "--summary-output",
            str(summary),
        ]
    )

    assert exit_code == 0
    assert verify_trajectory(trajectory)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["schema"] == "mirage.agent-run.v1"
    assert payload["policy_kind"] == "command"
    assert payload["status"] == "terminated"
    assert payload["trajectory_verified"] is True
