"""Tests for the v3 external CLI/API agent boundary."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from datetime import date
from pathlib import Path

import pytest

import stochopia.llm as llm_module
from stochopia.benchmark import MarketSnapshot, RiskBudget
from stochopia.benchmark_cli import main
from stochopia.environment import (
    AgentActionError,
    CommandAgentPolicy,
    EpisodeTask,
    LLMAgentPolicy,
    StochopiaStructurerEnv,
    Skip,
    create_api_agent_policy,
    parse_environment_action,
    run_agent_episode,
    verify_trajectory,
)
from stochopia.llm import LLMError, MockLLMClient
from stochopia.products import ClientProfile


class _AgentAPIResponse:
    status_code = 200

    def __init__(self) -> None:
        self._payload = {
            "choices": [
                {
                    "message": {
                        "content": '{"action":"skip","reason":"api smoke"}'
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        }
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload


class _AgentAPIFakeHTTP:
    requests: list[dict] = []

    def __init__(self, timeout: float | None = None) -> None:
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        type(self).requests.append(
            {
                "url": url,
                "payload": dict(json),
                "headers": dict(headers),
            }
        )
        return _AgentAPIResponse()


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
        for round_num in (1, 2, 3)
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

    custom = {
        "action": "request_quote",
        "product": {
            "product_type": "custom",
            "notional": 1_000_000,
            "maturity_months": 3,
            "strike_pct": 1.0,
            "barrier_pct": None,
            "barrier_type": None,
            "coupon_rate": None,
            "participation_rate": 1.0,
            "principal_protected": False,
            "target_client": "current_client",
            "pitch": "",
            "hedging_plan": "",
        },
    }
    with pytest.raises(AgentActionError, match="finite action grammar"):
        parse_environment_action(custom)


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
        StochopiaStructurerEnv(_task()),
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
    run_metadata = result.trajectory["policy_run_metadata"]
    assert run_metadata["provider"] == "mock"
    assert run_metadata["model"] == "mock-api-agent"
    assert run_metadata["temperature"] == pytest.approx(0.0)
    assert run_metadata["max_tokens"] == 4000
    assert len(run_metadata["system_prompt_sha256"]) == 64


@pytest.mark.asyncio
async def test_command_policy_uses_versioned_json_stdio_protocol(tmp_path):
    script = tmp_path / "agent.py"
    script.write_text(
        "\n".join(
                [
                    "import json, sys",
                    "request = json.load(sys.stdin)",
                    "assert request['schema'] == 'stochopia.agent-request.v1'",
                    "assert 'system_prompt' in request",
                    "assert 'task_manifest' not in request",
                    "assert 'task_hash' not in request",
                    "assert request['run_seed'] == 0",
                    "assert 'privileged_state' not in request['observation']",
                    "assert 'episode_id' not in "
                    "json.dumps(request['observation'])",
                    "assert 'hidden-client' not in json.dumps(request)",
                    "schema = request['action_schema']",
                    "assert schema['target_client'] == 'current_client'",
                    "assert schema['product_domain_version'] == "
                    "'stochopia.product-domain.csi-action-contract.v1'",
                    "assert set(schema['client_topic_enum']) == "
                    "{'capital','loss_tolerance','maturity','product_types',"
                    "'protection','preferences','purchase_status',"
                    "'risk_appetite','return_hurdle'}",
                    "assert len(schema['client_topic_enum']) == 9",
                    "assert 'custom' not in schema['allowed_product_types']",
                    "assert schema['notional_rule']['allowed_values']",
                    "assert schema['funding_constraints']",
                    "print(json.dumps({'action': 'skip', 'reason': 'cli smoke'}))",
                ]
        ),
        encoding="utf-8",
    )
    policy = CommandAgentPolicy([sys.executable, str(script)])

    result = await run_agent_episode(StochopiaStructurerEnv(_task()), policy)

    assert result.status == "terminated"
    assert result.steps == 2
    assert result.invalid_actions == 0
    assert result.usage == {"calls": 2}
    assert all(
        entry["policy_metadata"]["policy_kind"] == "command"
        for entry in result.trajectory["entries"]
    )
    assert result.trajectory["public_action_schema"][
        "schema_version"
    ] == "stochopia.environment.action-schema.v1"


@pytest.mark.asyncio
async def test_malformed_agent_output_is_recorded_and_bounded():
    policy = LLMAgentPolicy(
        MockLLMClient(["not an action"]),
        name="malformed-agent",
    )

    result = await run_agent_episode(
        StochopiaStructurerEnv(_task(), max_steps_per_round=2),
        policy,
    )

    assert result.status == "truncated"
    assert result.steps == 2
    assert result.invalid_actions == 2
    assert result.parser_errors == 2
    assert result.invocation_errors == 0
    assert result.trajectory_verified is True


@pytest.mark.asyncio
async def test_command_failure_is_infrastructure_not_agent_action(tmp_path):
    script = tmp_path / "failing_agent.py"
    script.write_text(
        "import sys\nsys.stderr.write('provider unavailable')\nsys.exit(7)\n",
        encoding="utf-8",
    )
    output = tmp_path / "partial-trajectory.json"

    result = await run_agent_episode(
        StochopiaStructurerEnv(_task()),
        CommandAgentPolicy([sys.executable, str(script)]),
        trajectory_path=output,
    )

    assert result.status == "infrastructure_error"
    assert result.steps == 0
    assert result.invalid_actions == 0
    assert result.parser_errors == 0
    assert result.invocation_errors == 1
    assert result.infrastructure_error["error_type"] == "LLMError"
    assert result.constraint_summary["steps"] == 0
    assert result.trajectory["status"] == "partial"
    assert result.trajectory["run_outcome"]["status"] == "infrastructure_error"
    assert verify_trajectory(output)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
@pytest.mark.asyncio
async def test_command_timeout_kills_descendant_process_group(tmp_path):
    marker = tmp_path / "orphan-was-alive"
    script = tmp_path / "spawning_agent.py"
    child_code = (
        "import pathlib,time;"
        "time.sleep(0.4);"
        f"pathlib.Path({str(marker)!r}).write_text('orphan')"
    )
    script.write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )

    result = await run_agent_episode(
        StochopiaStructurerEnv(_task()),
        CommandAgentPolicy(
            [sys.executable, str(script)],
            timeout=0.05,
        ),
    )
    await asyncio.sleep(0.7)

    assert result.status == "infrastructure_error"
    assert result.invalid_actions == 0
    assert result.steps == 0
    assert not marker.exists()


def test_direct_api_policy_requires_key_but_never_accepts_key_value(monkeypatch):
    monkeypatch.delenv("STOCHOPIA_TEST_API_KEY", raising=False)
    with pytest.raises(LLMError, match="STOCHOPIA_TEST_API_KEY"):
        create_api_agent_policy(
            provider="openai-compatible",
            base_url="https://example.test/v1",
            model="test-model",
            api_key_env="STOCHOPIA_TEST_API_KEY",
        )

    monkeypatch.setenv("STOCHOPIA_TEST_API_KEY", "top-secret")
    policy = create_api_agent_policy(
        provider="openai-compatible",
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="STOCHOPIA_TEST_API_KEY",
    )
    assert policy.client.config.api_key_env == "STOCHOPIA_TEST_API_KEY"
    assert "top-secret" not in policy.name


@pytest.mark.asyncio
async def test_openai_compatible_adapter_runs_end_to_end_without_secret_leak(
    monkeypatch,
):
    monkeypatch.setenv("STOCHOPIA_TEST_API_KEY", "adapter-secret")
    monkeypatch.setattr(
        llm_module.httpx,
        "AsyncClient",
        _AgentAPIFakeHTTP,
    )
    _AgentAPIFakeHTTP.requests = []
    policy = create_api_agent_policy(
        provider="openai-compatible",
        base_url="https://example.test/v1",
        model="provider-model-id",
        api_key_env="STOCHOPIA_TEST_API_KEY",
        temperature=0.1,
        max_tokens=1234,
        timeout=9.0,
    )

    result = await run_agent_episode(StochopiaStructurerEnv(_task()), policy)

    assert result.status == "terminated"
    assert len(_AgentAPIFakeHTTP.requests) == 2
    assert all(
        item["url"] == "https://example.test/v1/chat/completions"
        for item in _AgentAPIFakeHTTP.requests
    )
    first_request = _AgentAPIFakeHTTP.requests[0]["payload"]
    policy_request = json.loads(first_request["messages"][1]["content"])
    assert policy_request["schema"] == "stochopia.agent-request.v1"
    assert policy_request["action_schema"]["client_topic_enum"]
    assert first_request["model"] == "provider-model-id"
    assert first_request["temperature"] == pytest.approx(0.1)
    assert first_request["max_tokens"] == 1234
    assert result.trajectory["policy_run_metadata"]["base_url"] == (
        "https://example.test/v1"
    )
    assert "adapter-secret" not in json.dumps(
        result.trajectory,
        ensure_ascii=False,
    )


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
            str(root / "scenarios/stochopia_csi/market_snapshots.example.csv"),
            "--episode",
            "SYNTHETIC_CSI500_DEMO",
            "--client-json",
            str(root / "scenarios/stochopia_csi/client.example.json"),
            "--risk-budget-json",
            str(root / "scenarios/stochopia_csi/risk_budget.example.json"),
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
    assert payload["schema"] == "stochopia.agent-run.v1"
    assert payload["policy_kind"] == "command"
    assert payload["status"] == "terminated"
    assert payload["constraint_summary"]["steps"] == payload["steps"]
    assert payload["reward_summary"]["total_steps"] == payload["steps"]
    assert payload["trajectory_verified"] is True


def test_test_agent_cli_returns_nonzero_for_infrastructure_failure(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "failing_agent.py"
    script.write_text("raise SystemExit(9)\n", encoding="utf-8")
    trajectory = tmp_path / "partial-trajectory.json"
    summary = tmp_path / "partial-summary.json"
    command = " ".join(
        (shlex.quote(sys.executable), shlex.quote(str(script)))
    )

    exit_code = main(
        [
            "test-agent",
            str(root / "scenarios/stochopia_csi/market_snapshots.example.csv"),
            "--episode",
            "SYNTHETIC_CSI500_DEMO",
            "--client-json",
            str(root / "scenarios/stochopia_csi/client.example.json"),
            "--risk-budget-json",
            str(root / "scenarios/stochopia_csi/risk_budget.example.json"),
            "--agent-command",
            command,
            "--output",
            str(trajectory),
            "--summary-output",
            str(summary),
        ]
    )

    assert exit_code == 2
    assert verify_trajectory(trajectory)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["status"] == "infrastructure_error"
    assert payload["steps"] == 0
    assert payload["invalid_actions"] == 0
    assert payload["invocation_errors"] == 1


def test_v3_suite_run_replay_and_aggregate_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scenario = root / "scenarios/stochopia_csi"
    script = tmp_path / "suite_agent.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert 'task_hash' not in request\n"
        "assert request['action_schema']['target_client'] == 'current_client'\n"
        "print(json.dumps({'action': 'skip', 'reason': 'suite smoke'}))\n",
        encoding="utf-8",
    )
    command = " ".join(
        (shlex.quote(sys.executable), shlex.quote(str(script)))
    )
    suite = tmp_path / "suite.v3.json"
    outputs = tmp_path / "outputs"

    assert main(
        [
            "make-v3-suite",
            str(scenario / "market_snapshots.example.csv"),
            "--episodes",
            "SYNTHETIC_CSI500_DEMO",
            "--client-json",
            str(scenario / "client.example.json"),
            "--risk-budget-json",
            str(scenario / "risk_budget.example.json"),
            "--name",
            "cli-smoke",
            "--version",
            "1",
            "--split",
            "test",
            "--output",
            str(suite),
        ]
    ) == 0
    assert main(
        [
            "run-v3-suite",
            str(suite),
            "--output-dir",
            str(outputs),
            "--agent-command",
            command,
            "--replicates",
            "2",
            "--bootstrap-resamples",
            "20",
        ]
    ) == 0
    assert main(
        [
            "run-v3-suite",
            str(suite),
            "--output-dir",
            str(outputs),
            "--agent-command",
            command,
            "--replicates",
            "2",
            "--bootstrap-resamples",
            "20",
        ]
    ) == 0
    run_manifest = json.loads(
        (outputs / "run-manifest.v3.json").read_text(encoding="utf-8")
    )
    assert run_manifest["resumed_runs"] == 2
    script.write_text(
        script.read_text(encoding="utf-8") + "# policy changed\n",
        encoding="utf-8",
    )
    assert main(
        [
            "run-v3-suite",
            str(suite),
            "--output-dir",
            str(outputs),
            "--agent-command",
            command,
            "--replicates",
            "2",
            "--bootstrap-resamples",
            "20",
        ]
    ) == 0
    changed_manifest = json.loads(
        (outputs / "run-manifest.v3.json").read_text(encoding="utf-8")
    )
    assert changed_manifest["resumed_runs"] == 0

    trajectory = next(outputs.glob("*.trajectory.json"))
    evaluation = tmp_path / "manual.evaluation.json"
    assert main(
        [
            "evaluate-trajectory",
            str(trajectory),
            "--suite",
            str(suite),
            "--output",
            str(evaluation),
        ]
    ) == 0
    aggregate = tmp_path / "aggregate.json"
    aggregate_csv = tmp_path / "aggregate.csv"
    assert main(
        [
            "aggregate-v3",
            str(outputs),
            "--output-json",
            str(aggregate),
            "--output-csv",
            str(aggregate_csv),
            "--bootstrap-resamples",
            "20",
        ]
    ) == 0

    assert verify_trajectory(trajectory)
    evaluation_payload = json.loads(evaluation.read_text(encoding="utf-8"))
    assert evaluation_payload["replay_verified"] is True
    aggregate_payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert aggregate_payload["evaluation_count"] == 2
    assert aggregate_csv.exists()
