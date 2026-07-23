"""Command-line utilities for validating and smoke-testing MIRAGE episodes."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .benchmark import (
    BenchmarkCondition,
    BenchmarkError,
    LongHorizonEnvironment,
    PortfolioState,
    ProductDomainSpec,
    RiskBudget,
    calibrate_risk_budget,
    enumerate_domain,
    load_market_snapshots,
    oracle_candidate_grid,
    resolve_client_profile,
)
from .products import ClientProfile, parse_product_spec
from .market_builder import build_monthly_snapshots, load_daily_closes, write_market_snapshots
from .benchmark_runner import STRATEGIES, compute_metrics, run_episode, trace_to_dict
from .env_agents import EnvResponseCache, FrozenEnvAgent
from .environment import (
    TaskSuite,
    aggregate_evaluations,
    CommandAgentPolicy,
    EpisodeTask,
    LLMAgentPolicy,
    MirageStructurerEnv,
    TrajectoryRecorder,
    create_api_agent_policy,
    load_evaluations,
    replay_and_evaluate,
    run_agent_episode as run_v3_agent_episode,
    save_evaluation_aggregate,
)
from .role_config import RoleConfigError, load_judges_config, load_role_specs
from .llm import BaseLLMClient, LLMError, create_client, load_model_registry
from .experiment import build_experiment_manifest, manifest_payload, paired_condition_contrasts
from .pricing import QuotePolicy, calibrate_quote_policy, evaluate_quote_policy, scale_quote_policy_markup
from .stats import cluster_bootstrap_ci, derive_seed, holm_adjust
from .judge import (
    DIMENSIONS,
    DimensionScore,
    JudgeResult,
    aggregate_judge_bundle,
    build_judge_input,
    judge_soft_quality,
    reliability_summary,
)

RUN_OUTPUT_SCHEMA_VERSION = "mirage.run-output.v3"
RUN_FINGERPRINT_VERSION = "mirage.run-fingerprint.v1"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _source_implementation_sha256() -> str:
    """Fingerprint the executable package sources and bundled prompts."""

    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = sorted(
        (
            path
            for pattern in ("*.py", "*.md")
            for path in package_root.rglob(pattern)
            if "__pycache__" not in path.parts
        ),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    """Durably replace one JSON result; partial files are never final results."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _quarantine_invalid_result(path: Path) -> Path:
    """Move an incomplete/stale result aside so the job can be rerun safely."""

    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()[:12]
    candidate = path.with_name(f"{path.name}.invalid-{digest}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.invalid-{digest}-{suffix}")
        suffix += 1
    os.replace(path, candidate)
    return candidate


def _client() -> ClientProfile:
    return ClientProfile(
        id="institutional_demo",
        name="Synthetic Institutional Client",
        capital=50_000_000,
        max_loss_pct=1.0,
        min_return_pct=0.03,
        risk_appetite="moderate",
        max_maturity_months=12,
        min_hit_prob=0.5,
        preferences="balanced yield and drawdown protection",
    )


def _load_quote_policy(path: Path | None) -> QuotePolicy | None:
    """Load a QuotePolicy from --quote-policy-json; None keeps the default policy."""
    if path is None:
        return None
    return QuotePolicy(**json.loads(path.read_text(encoding="utf-8")))


def _budget() -> RiskBudget:
    return RiskBudget(
        notional=200_000_000,
        net_delta=100_000_000,
        gross_delta=250_000_000,
        net_vega=20_000_000,
        stress_loss=100_000_000,
    )


# ---------------------------------------------------------------------------
# v2 env-role wiring (opt-in via --roles-config). Absent it, every environment
# stays pure-deterministic and behaves exactly like the pre-wiring CLI.
# ---------------------------------------------------------------------------

_ENV_ROLE_TYPES = ("client", "risk_control", "trading_desk")


def _roles_config_meta(path: Path) -> tuple[str, str]:
    """Return (npc_lineup_id, roles_config_sha256) for reproducible audit."""
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = yaml.safe_load(raw) or {}
    return str(data.get("main_npc_lineup_id", "")), sha


def _load_env_role_specs(roles_config: Path, registry):
    """Load role specs, fail fast (SystemExit) on any config error."""
    try:
        return load_role_specs(roles_config, registry)
    except RoleConfigError as exc:
        raise SystemExit(f"failed to load roles config {roles_config}: {exc}")


def _make_env_agents(specs, registry, cache: EnvResponseCache | None) -> dict[str, FrozenEnvAgent]:
    """Build a fresh FrozenEnvAgent per env role (keyed by role type).

    Fresh agents per episode keep role conversation history from leaking across
    episodes; the response cache (if any) is shared so repeated identical
    requests still hit a frozen NPC reply.
    """
    agents: dict[str, FrozenEnvAgent] = {}
    for spec in specs.values():
        if spec.role not in _ENV_ROLE_TYPES:
            continue
        env_client = create_client(spec.inference.model_ref, registry)
        system_prompt = Path(spec.system_prompt_file).read_text(encoding="utf-8")
        agents[spec.role] = FrozenEnvAgent(
            spec, env_client, system_prompt=system_prompt, cache=cache,
        )
    return agents


def _env_cache(env_cache_dir: Path | None) -> EnvResponseCache | None:
    if env_cache_dir is None:
        return None
    env_cache_dir.mkdir(parents=True, exist_ok=True)
    return EnvResponseCache(env_cache_dir / "env_cache.jsonl")


def _add_v3_policy_arguments(parser: argparse.ArgumentParser) -> None:
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument(
        "--agent-command",
        help="local executable command; receives one JSON request on stdin per step",
    )
    backend.add_argument(
        "--model",
        help="registered model name from --models-config",
    )
    backend.add_argument(
        "--api-model",
        help="direct API model id; does not require editing models.yaml",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=Path("config/models.yaml"),
    )
    parser.add_argument(
        "--api-provider",
        choices=("openai-compatible", "anthropic"),
        default="openai-compatible",
    )
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--command-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps-per-round", type=int, default=12)
    parser.add_argument("--history-records", type=int, default=32)
    parser.add_argument("--history-chars", type=int, default=96_000)
    parser.add_argument("--redact-raw-output", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIRAGE long-horizon benchmark utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-market", help="validate monthly market snapshot CSV")
    validate.add_argument("csv", type=Path)
    build = sub.add_parser("build-market", help="build past-only month-end snapshots from daily closes")
    build.add_argument("daily_csv", type=Path)
    build.add_argument("output_csv", type=Path)
    build.add_argument("--risk-free-rate", type=float, default=0.02)
    demo = sub.add_parser("demo", help="run one deterministic desk quote")
    demo.add_argument("csv", type=Path)
    demo.add_argument("--episode", required=True)
    demo.add_argument("--product-json", type=Path, required=True)
    demo.add_argument("--full", action="store_true")
    demo.add_argument("--static", action="store_true")
    run = sub.add_parser("run-episode", help="run one LLM against a frozen episode")
    run.add_argument("csv", type=Path)
    run.add_argument("--episode", required=True)
    run.add_argument("--client-json", type=Path, required=True)
    run.add_argument("--risk-budget-json", type=Path, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    run.add_argument("--strategy", choices=STRATEGIES, default="quote_and_revise")
    run.add_argument("--full", action="store_true")
    run.add_argument("--static", action="store_true")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--roles-config", type=Path, default=None,
        help="benchmark_roles.yaml; when omitted env roles stay pure-deterministic",
    )
    run.add_argument(
        "--env-cache-dir", type=Path, default=None,
        help="directory for the shared append-only env response cache (jsonl)",
    )
    run.add_argument(
        "--quote-policy-json", type=Path, default=None,
        help="QuotePolicy JSON (e.g. data/derived/quote_policy.v2.candidate.json); "
             "when omitted the environment uses the default QuotePolicy",
    )
    agent = sub.add_parser(
        "test-agent",
        help="run a local CLI agent or API-backed LLM against the v3 environment",
    )
    agent.add_argument("csv", type=Path)
    agent.add_argument("--episode", required=True)
    agent.add_argument("--client-json", type=Path, required=True)
    agent.add_argument("--risk-budget-json", type=Path, required=True)
    backend = agent.add_mutually_exclusive_group(required=True)
    backend.add_argument(
        "--agent-command",
        help="local executable command; receives one JSON request on stdin per step",
    )
    backend.add_argument(
        "--model",
        help="registered model name from --models-config",
    )
    backend.add_argument(
        "--api-model",
        help="direct API model id; does not require editing models.yaml",
    )
    agent.add_argument(
        "--models-config",
        type=Path,
        default=Path("config/models.yaml"),
    )
    agent.add_argument(
        "--api-provider",
        choices=("openai-compatible", "anthropic"),
        default="openai-compatible",
    )
    agent.add_argument(
        "--api-base-url",
        help="direct API base URL, e.g. https://api.openai.com/v1",
    )
    agent.add_argument(
        "--api-key-env",
        help="environment variable containing the direct API key",
    )
    agent.add_argument("--temperature", type=float, default=0.0)
    agent.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="agent response limit (registered model default when omitted)",
    )
    agent.add_argument(
        "--api-timeout",
        type=float,
        default=120.0,
        help="direct API request timeout in seconds",
    )
    agent.add_argument(
        "--command-timeout",
        type=float,
        default=180.0,
        help="per-step local command timeout in seconds",
    )
    agent.add_argument("--seed", type=int, default=None)
    agent.add_argument("--max-steps-per-round", type=int, default=12)
    agent.add_argument("--history-records", type=int, default=32)
    agent.add_argument("--history-chars", type=int, default=96_000)
    agent.add_argument(
        "--quote-policy-json",
        type=Path,
        default=None,
        help="optional frozen QuotePolicy JSON",
    )
    agent.add_argument(
        "--redact-raw-output",
        action="store_true",
        help="store only the response hash, not raw agent text, in the trajectory",
    )
    agent.add_argument(
        "--output",
        type=Path,
        required=True,
        help="verified v3 trajectory JSON",
    )
    agent.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="optional compact run-summary JSON",
    )
    suite = sub.add_parser(
        "make-v3-suite",
        help="freeze selected episodes and hidden truth into a versioned v3 task suite",
    )
    suite.add_argument("csv", type=Path)
    suite.add_argument("--episodes", nargs="+", required=True)
    suite.add_argument("--client-json", type=Path, required=True)
    suite.add_argument("--risk-budget-json", type=Path, required=True)
    suite.add_argument("--quote-policy-json", type=Path, default=None)
    suite.add_argument("--name", required=True)
    suite.add_argument("--version", required=True)
    suite.add_argument(
        "--split",
        choices=("train", "dev", "test", "private_test"),
        required=True,
    )
    suite.add_argument(
        "--public-notional-base",
        type=float,
        default=10_000_000.0,
    )
    suite.add_argument("--seed", type=int, default=0)
    suite.add_argument("--output", type=Path, required=True)

    run_suite = sub.add_parser(
        "run-v3-suite",
        help="run one external policy over every task and replay-evaluate each trajectory",
    )
    run_suite.add_argument("suite", type=Path)
    run_suite.add_argument("--output-dir", type=Path, required=True)
    run_suite.add_argument("--replicates", type=int, default=1)
    run_suite.add_argument(
        "--force",
        action="store_true",
        help="rerun matching complete trajectories instead of resuming them",
    )
    run_suite.add_argument("--bootstrap-resamples", type=int, default=10_000)
    run_suite.add_argument("--aggregate-seed", type=int, default=20260802)
    _add_v3_policy_arguments(run_suite)

    evaluate = sub.add_parser(
        "evaluate-trajectory",
        help="cryptographically verify and economically replay one v3 trajectory",
    )
    evaluate.add_argument("trajectory", type=Path)
    evaluate.add_argument("--suite", type=Path, default=None)
    evaluate.add_argument("--output", type=Path, required=True)

    aggregate_v3 = sub.add_parser(
        "aggregate-v3",
        help="aggregate replay-verified v3 evaluation artifacts by policy",
    )
    aggregate_v3.add_argument("results_dir", type=Path)
    aggregate_v3.add_argument("--output-json", type=Path, required=True)
    aggregate_v3.add_argument("--output-csv", type=Path, default=None)
    aggregate_v3.add_argument("--bootstrap-resamples", type=int, default=10_000)
    aggregate_v3.add_argument("--seed", type=int, default=20260802)

    manifest = sub.add_parser("make-manifest", help="freeze the factorial experiment job list")
    manifest.add_argument("--episodes", nargs="+", required=True)
    manifest.add_argument("--models", nargs="+", required=True)
    manifest.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    manifest.add_argument("--output", type=Path, required=True)
    calib = sub.add_parser("calibrate-budget", help="calibrate the risk budget on development episodes, then freeze")
    calib.add_argument("csv", type=Path)
    calib.add_argument("--episodes", nargs="+", required=True)
    calib.add_argument("--client-json", type=Path, required=True)
    calib.add_argument("--base-budget-json", type=Path, required=True)
    calib.add_argument("--report-output", type=Path, required=True)
    calib.add_argument("--budget-output", type=Path, required=True)
    calib.add_argument("--factors", nargs="+", type=float, default=None)
    calib.add_argument(
        "--quote-policy-json", type=Path, default=None,
        help="QuotePolicy JSON to price candidates with during calibration; "
             "when omitted the default QuotePolicy is used",
    )
    margin = sub.add_parser(
        "calibrate-margin",
        help="calibrate mirage.pricing.QuotePolicy's markup scale on development episodes, then freeze",
    )
    margin.add_argument("csv", type=Path)
    margin.add_argument("--episodes", nargs="+", required=True)
    margin.add_argument("--client-json", type=Path, required=True)
    margin.add_argument(
        "--risk-budget-json",
        type=Path,
        default=None,
        help="frozen risk budget whose notional capacity runtime quotes use",
    )
    margin.add_argument(
        "--sensitivity-factors", nargs="+", type=float, default=[0.8, 1.0, 1.2],
        help="post-calibration overall a_* scale multipliers to report robustness for",
    )
    margin.add_argument("--report-output", type=Path, required=True)
    margin.add_argument("--policy-output", type=Path, required=True)
    margin.add_argument(
        "--candidates-per-snapshot", type=int, default=30,
        help="deterministic sample size of lattice candidates priced per (episode, round) snapshot",
    )
    margin.add_argument("--factors", nargs="+", type=float, default=None, help="calibration grid override")
    margin.add_argument("--seed", type=int, default=42)
    runm = sub.add_parser("run-manifest", help="execute frozen manifest jobs sequentially, resume on rerun")
    runm.add_argument("csv", type=Path)
    runm.add_argument("--manifest", type=Path, required=True)
    runm.add_argument("--client-json", type=Path, required=True)
    runm.add_argument("--risk-budget-json", type=Path, required=True)
    runm.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    runm.add_argument("--outputs-dir", type=Path, default=Path("outputs/csi_benchmark"))
    runm.add_argument("--only-models", nargs="+", default=None)
    runm.add_argument(
        "--roles-config", type=Path, default=None,
        help="benchmark_roles.yaml; when omitted env roles stay pure-deterministic",
    )
    runm.add_argument(
        "--env-cache-dir", type=Path, default=None,
        help="directory for the shared append-only env response cache (jsonl)",
    )
    runm.add_argument(
        "--quote-policy-json", type=Path, default=None,
        help="QuotePolicy JSON (e.g. data/derived/quote_policy.v2.candidate.json); "
             "when omitted the environment uses the default QuotePolicy",
    )
    agg = sub.add_parser("aggregate", help="aggregate run outputs into CSV and per-condition markdown")
    agg.add_argument("results_dir", type=Path)
    agg.add_argument("--output-csv", type=Path, required=True)
    agg.add_argument("--output-md", type=Path, default=None)
    agg.add_argument("--bootstrap-resamples", type=int, default=10_000)
    agg.add_argument("--n-permutations", type=int, default=20_000)
    agg.add_argument("--alpha", type=float, default=0.05)
    agg.add_argument("--seed", type=int, default=20260802)
    jr = sub.add_parser(
        "judge-runs",
        help="offline blind two-judge x repeats batch over frozen voluntary submissions",
    )
    jr.add_argument("results_dir", type=Path)
    jr.add_argument(
        "--judge-models", nargs=2, default=None, metavar=("JUDGE_A", "JUDGE_B"),
        help="exactly two judge model names from --models-config; overrides --roles-config's "
             "judges.models when given, required if --roles-config is not",
    )
    jr.add_argument(
        "--repeats", type=int, default=None,
        help="defaults to --roles-config's judges.repeats when given, else 3",
    )
    jr.add_argument(
        "--roles-config", type=Path, default=None,
        help="benchmark_roles.yaml; supplies default judge models/repeats "
             "(judges: block) when the corresponding CLI flag is omitted",
    )
    jr.add_argument("--sample", type=int, default=60, help="stratified (condition x model) sample size")
    jr.add_argument("--salt", required=True, help="blind_id salt; freeze and record for reproducible re-review")
    jr.add_argument("--seed", type=int, default=20260802)
    jr.add_argument("--models-config", type=Path, default=Path("config/models.yaml"))
    return parser


def _registered_v3_policy(args: argparse.Namespace) -> LLMAgentPolicy:
    """Create a v3 policy from the existing model registry, failing early."""

    registry = load_model_registry(args.models_config)
    if args.model not in registry:
        available = ", ".join(sorted(registry)) or "(none)"
        raise SystemExit(
            f"model {args.model!r} is not registered; available: {available}"
        )
    config = registry[args.model]
    if config.provider != "mock":
        if not config.api_key_env:
            raise SystemExit(
                f"registered model {args.model!r} has no api_key_env"
            )
        if not os.environ.get(config.api_key_env, "").strip():
            raise SystemExit(
                f"missing API key: environment variable "
                f"{config.api_key_env} is empty"
            )
    max_tokens = args.max_tokens or config.max_tokens
    return LLMAgentPolicy(
        create_client(args.model, registry),
        name=args.model,
        temperature=args.temperature,
        max_tokens=max_tokens,
    )


def _create_v3_policy(args: argparse.Namespace):
    """Construct any supported v3 policy with identical validation."""

    try:
        if args.agent_command is not None:
            return CommandAgentPolicy(
                args.agent_command,
                timeout=args.command_timeout,
            )
        if args.model is not None:
            return _registered_v3_policy(args)
        if not args.api_base_url:
            raise SystemExit("--api-base-url is required with --api-model")
        if not args.api_key_env:
            raise SystemExit("--api-key-env is required with --api-model")
        return create_api_agent_policy(
            provider=args.api_provider,
            base_url=args.api_base_url,
            model=args.api_model,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens or 4000,
            timeout=args.api_timeout,
        )
    except (LLMError, ValueError) as exc:
        raise SystemExit(f"failed to configure v3 agent: {exc}") from exc


def _cmd_test_agent(args: argparse.Namespace, snapshots: list) -> int:
    """Run one external policy through the canonical v3 reset/step boundary."""

    selected = tuple(
        snapshot
        for snapshot in snapshots
        if snapshot.episode_id == args.episode
    )
    if not selected:
        raise SystemExit(f"unknown episode: {args.episode}")
    client = ClientProfile(
        **json.loads(args.client_json.read_text(encoding="utf-8"))
    )
    risk_budget = RiskBudget(
        **json.loads(args.risk_budget_json.read_text(encoding="utf-8"))
    )
    task = EpisodeTask(
        snapshots=selected,
        client=client,
        risk_budget=risk_budget,
        quote_policy=_load_quote_policy(args.quote_policy_json) or QuotePolicy(),
        task_seed=0 if args.seed is None else args.seed,
    )
    environment = MirageStructurerEnv(
        task,
        max_steps_per_round=args.max_steps_per_round,
        expose_privileged_info=False,
    )

    policy = _create_v3_policy(args)

    result = asyncio.run(
        run_v3_agent_episode(
            environment,
            policy,
            seed=args.seed,
            trajectory_path=args.output,
            max_history_records=args.history_records,
            max_history_chars=args.history_chars,
            record_raw_output=not args.redact_raw_output,
        )
    )
    summary = result.summary()
    if args.summary_output is not None:
        _atomic_write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 2 if result.status == "infrastructure_error" else 0


def _cmd_make_v3_suite(args: argparse.Namespace) -> int:
    snapshots = load_market_snapshots(args.csv)
    requested = list(dict.fromkeys(args.episodes))
    grouped = {
        episode: tuple(
            item for item in snapshots if item.episode_id == episode
        )
        for episode in requested
    }
    missing = [episode for episode, rows in grouped.items() if not rows]
    if missing:
        raise SystemExit(f"unknown episodes: {missing}")
    client_payload = json.loads(args.client_json.read_text(encoding="utf-8"))
    budget_payload = json.loads(
        args.risk_budget_json.read_text(encoding="utf-8")
    )
    client = ClientProfile(**client_payload)
    budget = RiskBudget(**budget_payload)
    policy = _load_quote_policy(args.quote_policy_json) or QuotePolicy()
    domain = ProductDomainSpec(
        public_notional_base=args.public_notional_base
    )
    tasks = tuple(
        EpisodeTask(
            snapshots=rows,
            client=client,
            risk_budget=budget,
            domain=domain,
            quote_policy=policy,
            task_seed=derive_seed(
                "mirage.v3.task-suite",
                args.name,
                args.version,
                episode,
                args.seed,
            ),
        )
        for episode, rows in grouped.items()
    )
    suite = TaskSuite(
        name=args.name,
        version=args.version,
        split=args.split,
        tasks=tasks,
        metadata={
            "market_csv_sha256": _file_sha256(args.csv),
            "client_json_sha256": _file_sha256(args.client_json),
            "risk_budget_json_sha256": _file_sha256(args.risk_budget_json),
            "quote_policy_json_sha256": _file_sha256(
                args.quote_policy_json
            ),
            "episode_ids": requested,
            "generator": "frozen-market-episode-import-v1",
            "generator_seed": args.seed,
        },
    )
    suite.save(args.output)
    print(
        json.dumps(
            {
                "format": suite.format,
                "suite_hash": suite.suite_hash,
                "tasks": len(suite.tasks),
                "split": suite.split,
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _write_v3_aggregate_csv(
    path: Path,
    aggregate: dict,
) -> None:
    rows: list[dict] = []
    for policy, policy_payload in aggregate["policies"].items():
        for metric, values in policy_payload["metrics"].items():
            rows.append(
                {
                    "policy": policy,
                    "policy_config_hash": policy_payload[
                        "policy_config_hash"
                    ],
                    "metric": metric,
                    **values,
                    "suite_hash": aggregate["suite_hash"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "policy",
            "policy_config_hash",
            "metric",
            "mean",
            "ci_low",
            "ci_high",
            "n_runs",
            "n_tasks",
            "suite_hash",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cmd_evaluate_trajectory(args: argparse.Namespace) -> int:
    suite_hash = None
    suite = None
    if args.suite is not None:
        suite = TaskSuite.load(args.suite)
        suite_hash = suite.suite_hash
    result = replay_and_evaluate(
        args.trajectory,
        suite_hash=suite_hash,
    )
    if suite is not None and result["task_hash"] not in {
        task.task_hash for task in suite.tasks
    }:
        raise SystemExit(
            "trajectory task is not a member of the supplied v3 task suite"
        )
    _atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "replay_verified": True,
                "evaluation_hash": result["evaluation_hash"],
                "task_hash": result["task_hash"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _cmd_aggregate_v3(args: argparse.Namespace) -> int:
    paths = sorted(args.results_dir.rglob("*.evaluation.json"))
    if not paths:
        raise SystemExit(
            f"no *.evaluation.json files found in {args.results_dir}"
        )
    aggregate = aggregate_evaluations(
        load_evaluations(paths),
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    save_evaluation_aggregate(args.output_json, aggregate)
    if args.output_csv is not None:
        _write_v3_aggregate_csv(args.output_csv, aggregate)
    print(
        json.dumps(
            {
                "evaluations": aggregate["evaluation_count"],
                "policies": aggregate["policy_count"],
                "suite_hash": aggregate["suite_hash"],
                "output": str(args.output_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _cmd_run_v3_suite(args: argparse.Namespace) -> int:
    suite = TaskSuite.load(args.suite)
    if (
        isinstance(args.replicates, bool)
        or not isinstance(args.replicates, int)
        or args.replicates < 1
    ):
        raise SystemExit("--replicates must be a positive integer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluations: list[dict] = []
    failures = 0
    resumed = 0
    for index, task in enumerate(suite.tasks):
        for replicate in range(args.replicates):
            stem = (
                f"{index:04d}-r{replicate:03d}-{task.task_hash[:12]}"
            )
            trajectory_path = args.output_dir / f"{stem}.trajectory.json"
            summary_path = args.output_dir / f"{stem}.summary.json"
            evaluation_path = args.output_dir / f"{stem}.evaluation.json"
            environment = MirageStructurerEnv(
                task,
                max_steps_per_round=args.max_steps_per_round,
                expose_privileged_info=False,
            )
            policy = _create_v3_policy(args)
            # Replicate seeds are public experiment configuration and are
            # intentionally shared across tasks. Deriving them from hidden
            # task hashes would create a policy-input side channel.
            run_seed = derive_seed(
                "mirage.v3.suite-policy-replicate",
                0 if args.seed is None else args.seed,
                replicate,
            )
            if (
                not args.force
                and trajectory_path.exists()
                and summary_path.exists()
                and TrajectoryRecorder.verify(trajectory_path)
            ):
                try:
                    existing = json.loads(
                        trajectory_path.read_text(encoding="utf-8")
                    )
                    existing_summary = json.loads(
                        summary_path.read_text(encoding="utf-8")
                    )
                    expected_policy = dict(
                        policy.reproducibility_metadata()
                    )
                    matches = (
                        existing.get("status") == "complete"
                        and existing.get("run_id_seed") == run_seed
                        and existing.get("metadata", {}).get("task_hash")
                        == task.task_hash
                        and existing.get("environment_configuration")
                        == environment.configuration
                        and _sha256_json(
                            existing.get("policy_run_metadata")
                        )
                        == _sha256_json(expected_policy)
                        and existing_summary.get("suite_hash")
                        == suite.suite_hash
                        and existing_summary.get("replicate") == replicate
                    )
                    if matches:
                        evaluation = replay_and_evaluate(
                            trajectory_path,
                            output_path=evaluation_path,
                            suite_hash=suite.suite_hash,
                        )
                        evaluations.append(evaluation)
                        resumed += 1
                        continue
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ):
                    pass
            result = asyncio.run(
                run_v3_agent_episode(
                    environment,
                    policy,
                    seed=run_seed,
                    trajectory_path=trajectory_path,
                    max_history_records=args.history_records,
                    max_history_chars=args.history_chars,
                    record_raw_output=not args.redact_raw_output,
                )
            )
            summary = {
                **result.summary(),
                "suite_hash": suite.suite_hash,
                "suite_name": suite.name,
                "suite_version": suite.version,
                "suite_split": suite.split,
                "replicate": replicate,
                "run_seed": run_seed,
            }
            _atomic_write_json(summary_path, summary)
            evaluation = replay_and_evaluate(
                trajectory_path,
                output_path=evaluation_path,
                suite_hash=suite.suite_hash,
            )
            evaluations.append(evaluation)
            if result.status == "infrastructure_error":
                failures += 1

    aggregate = aggregate_evaluations(
        evaluations,
        n_resamples=args.bootstrap_resamples,
        seed=args.aggregate_seed,
    )
    aggregate_path = args.output_dir / "aggregate.v3.json"
    save_evaluation_aggregate(aggregate_path, aggregate)
    _write_v3_aggregate_csv(
        args.output_dir / "aggregate.v3.csv",
        aggregate,
    )
    run_manifest = {
        "format": "mirage-suite-run.v3",
        "suite_hash": suite.suite_hash,
        "task_count": len(suite.tasks),
        "replicates": args.replicates,
        "run_count": len(suite.tasks) * args.replicates,
        "evaluation_count": len(evaluations),
        "infrastructure_failures": failures,
        "resumed_runs": resumed,
        "aggregate_hash": aggregate["aggregate_hash"],
        "aggregate_path": str(aggregate_path),
    }
    _atomic_write_json(args.output_dir / "run-manifest.v3.json", run_manifest)
    print(json.dumps(run_manifest, ensure_ascii=False, sort_keys=True))
    return 2 if failures else 0


def _condition_from_id(condition_id: str) -> BenchmarkCondition:
    try:
        info, horizon = condition_id.split("_")
        return BenchmarkCondition(
            full_information={"full": True, "partial": False}[info],
            dynamic={"dynamic": True, "static": False}[horizon],
        )
    except (ValueError, KeyError):
        raise SystemExit(f"unknown condition id: {condition_id}")


def _cmd_calibrate(args: argparse.Namespace, snapshots: list) -> int:
    client = ClientProfile(**json.loads(args.client_json.read_text(encoding="utf-8")))
    base = RiskBudget(**json.loads(args.base_budget_json.read_text(encoding="utf-8")))
    wanted = set(args.episodes)
    missing = wanted - {snapshot.episode_id for snapshot in snapshots}
    if missing:
        raise SystemExit(f"unknown development episodes: {sorted(missing)}")
    cases = [
        (
            snapshot,
            resolve_client_profile(client, snapshot.round_num),
            PortfolioState(),
        )
        for snapshot in snapshots
        if snapshot.episode_id in wanted
    ]
    kwargs = {"factors": tuple(args.factors)} if args.factors else {}
    kwargs["policy"] = _load_quote_policy(args.quote_policy_json)
    report = calibrate_risk_budget(ProductDomainSpec(), cases, base, **kwargs)
    report["development_episodes"] = sorted(wanted)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.budget_output.parent.mkdir(parents=True, exist_ok=True)
    args.budget_output.write_text(
        json.dumps(report["selected_budget"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "selected_factor": report["selected_factor"],
        "feasibility_rate": report["selected_feasibility_rate"],
        "within_target": report["within_target"],
        "budget_output": str(args.budget_output),
    }, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# calibrate-margin: CLI wiring for mirage.pricing.calibrate_quote_policy
# (REDESIGN_PLAN.md §4/§5). Samples a fixed-seed subset of the shared
# ProductDomainSpec lattice per (episode, round) development snapshot -- the
# full lattice is tens of thousands of candidates, more than needed to hit the
# 20-60% positive-margin / non-degenerate-ranking target and far more MC than
# a few minutes affords -- then grid-searches the a_* markup scale and reports
# 0.8/1.0/1.2 sensitivity of the selected policy.
# ---------------------------------------------------------------------------


def _cmd_calibrate_margin(args: argparse.Namespace, snapshots: list) -> int:
    client = ClientProfile(**json.loads(args.client_json.read_text(encoding="utf-8")))
    wanted = set(args.episodes)
    missing = wanted - {snapshot.episode_id for snapshot in snapshots}
    if missing:
        raise SystemExit(f"unknown development episodes: {sorted(missing)}")
    wanted_snapshots = [snapshot for snapshot in snapshots if snapshot.episode_id in wanted]
    if not wanted_snapshots:
        raise SystemExit("no snapshots found for the requested development episodes")

    domain = ProductDomainSpec()
    # Deterministic per-snapshot subsample: seeded from (episode_id, round_num,
    # --seed) so re-running with the same inputs reproduces the same dev cases
    # regardless of csv row order.
    dev_cases: list[tuple] = []
    for snapshot in wanted_snapshots:
        round_client = resolve_client_profile(client, snapshot.round_num)
        candidates = list(enumerate_domain(round_client, domain))
        if not candidates:
            raise SystemExit(
                f"product domain lattice is empty for round {snapshot.round_num}"
            )
        sample_size = min(args.candidates_per_snapshot, len(candidates))
        stratum_seed = derive_seed(
            "mirage.calibrate_margin.snapshot", snapshot.episode_id, snapshot.round_num, args.seed,
        )
        for product in random.Random(stratum_seed).sample(candidates, sample_size):
            market = snapshot.to_market_state(product.maturity_months)
            dev_cases.append((product, market, round_client))

    kwargs = {"factors": tuple(args.factors)} if args.factors else {}
    risk_budget_path = getattr(args, "risk_budget_json", None)
    if risk_budget_path is not None:
        frozen_budget = RiskBudget(
            **json.loads(risk_budget_path.read_text(encoding="utf-8"))
        )
        capacity_fn = lambda _client: frozen_budget.notional
    else:
        frozen_budget = None
        capacity_fn = lambda current_client: current_client.capital
    calibrated, report = calibrate_quote_policy(
        dev_cases,
        seed=args.seed,
        capacity_fn=capacity_fn,
        **kwargs,
    )

    sensitivity_rows = []
    for factor in args.sensitivity_factors:
        scaled = scale_quote_policy_markup(calibrated, float(factor))
        result = evaluate_quote_policy(
            scaled,
            dev_cases,
            seed=args.seed,
            capacity_fn=capacity_fn,
        )
        sensitivity_rows.append({"sensitivity_factor": float(factor), **result})

    report["development_episodes"] = sorted(wanted)
    report["candidates_per_snapshot"] = args.candidates_per_snapshot
    report["n_snapshots"] = len(wanted_snapshots)
    report["n_dev_cases"] = len(dev_cases)
    report["seed"] = args.seed
    report["capacity_source"] = (
        "risk_budget.notional" if frozen_budget is not None else "client.capital"
    )
    report["sensitivity"] = sensitivity_rows

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.write_text(
        json.dumps(asdict(calibrated), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "selected_factor": report["selected_factor"],
        "selected_positive_margin_rate": report["selected_positive_margin_rate"],
        "ranking_nondegenerate": report["ranking_nondegenerate"],
        "within_target": report["within_target"],
        "n_dev_cases": len(dev_cases),
        "sensitivity": sensitivity_rows,
        "policy_output": str(args.policy_output),
    }, ensure_ascii=False))
    return 0


def _run_fingerprint(
    *,
    manifest: dict,
    snapshots: list,
    client_payload: dict,
    budget_payload: dict,
    quote_policy: QuotePolicy | None,
    models_config: Path,
    roles_config: Path | None,
) -> tuple[str, dict]:
    inputs = {
        "fingerprint_version": RUN_FINGERPRINT_VERSION,
        "run_output_schema_version": RUN_OUTPUT_SCHEMA_VERSION,
        "manifest": manifest,
        "market_snapshots": [asdict(snapshot) for snapshot in snapshots],
        "client": client_payload,
        "risk_budget": budget_payload,
        "quote_policy": asdict(quote_policy or QuotePolicy()),
        "models_config_sha256": _file_sha256(models_config),
        "roles_config_sha256": _file_sha256(roles_config),
        "implementation_sha256": _source_implementation_sha256(),
    }
    return _sha256_json(inputs), inputs


def _completed_result_matches(
    path: Path,
    *,
    run_fingerprint: str,
    job_fingerprint: str,
    job_id: str,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    job = payload.get("job")
    return (
        payload.get("schema_version") == RUN_OUTPUT_SCHEMA_VERSION
        and payload.get("complete") is True
        and payload.get("run_fingerprint") == run_fingerprint
        and payload.get("job_fingerprint") == job_fingerprint
        and isinstance(job, dict)
        and job.get("job_id") == job_id
        and isinstance(payload.get("trace"), dict)
        and isinstance(payload.get("metrics"), dict)
    )


def _cmd_run_manifest(args: argparse.Namespace, snapshots: list) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("jobs"), list):
        raise SystemExit(f"invalid experiment manifest: {args.manifest}")
    jobs = manifest["jobs"]
    if args.only_models:
        keep = set(args.only_models)
        jobs = [job for job in jobs if job["model"] in keep]
    client_payload = json.loads(args.client_json.read_text(encoding="utf-8"))
    budget_payload = json.loads(args.risk_budget_json.read_text(encoding="utf-8"))
    quote_policy = _load_quote_policy(args.quote_policy_json)
    registry = load_model_registry(args.models_config)
    # Load roles config once (fail fast) and share one response cache across jobs;
    # fresh FrozenEnvAgents are built per job so role history never leaks.
    env_specs = None
    env_cache = None
    roles_meta: dict = {}
    if args.roles_config is not None:
        env_specs = _load_env_role_specs(args.roles_config, registry)
        env_cache = _env_cache(args.env_cache_dir)
        npc_lineup_id, roles_sha = _roles_config_meta(args.roles_config)
        roles_meta = {"npc_lineup_id": npc_lineup_id, "roles_config_sha256": roles_sha}
    run_fingerprint, fingerprint_inputs = _run_fingerprint(
        manifest=manifest,
        snapshots=snapshots,
        client_payload=client_payload,
        budget_payload=budget_payload,
        quote_policy=quote_policy,
        models_config=args.models_config,
        roles_config=args.roles_config,
    )
    by_episode: dict[str, list] = {}
    for snapshot in snapshots:
        by_episode.setdefault(snapshot.episode_id, []).append(snapshot)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = failed = 0
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict) or not isinstance(job.get("job_id"), str):
            raise SystemExit(f"manifest contains an invalid job at index {index - 1}")
        out_path = args.outputs_dir / f"{job['job_id']}.json"
        job_fingerprint = _sha256_json(
            {"run_fingerprint": run_fingerprint, "job": job}
        )
        if out_path.exists():
            if _completed_result_matches(
                out_path,
                run_fingerprint=run_fingerprint,
                job_fingerprint=job_fingerprint,
                job_id=job["job_id"],
            ):
                skipped += 1
                continue
            quarantined = _quarantine_invalid_result(out_path)
            print(
                json.dumps(
                    {
                        "job_id": job["job_id"],
                        "quarantined_stale_or_incomplete_result": str(quarantined),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        episode_snapshots = by_episode.get(job["episode_id"])
        if not episode_snapshots:
            raise SystemExit(f"manifest references unknown episode: {job['episode_id']}")
        print(f"[{index}/{len(jobs)}] {job['job_id']} ...", flush=True)
        try:
            env_agents = (
                _make_env_agents(env_specs, registry, env_cache) if env_specs is not None else None
            )
            env = LongHorizonEnvironment(
                episode_snapshots,
                ClientProfile(**client_payload),
                RiskBudget(**budget_payload),
                _condition_from_id(job["condition"]),
                quote_policy=quote_policy,
                env_agents=env_agents,
            )
            llm = create_client(job["model"], registry)
            trace = asyncio.run(run_episode(env, llm, strategy=job["strategy"], seed=job.get("seed")))
        except Exception as exc:  # noqa: BLE001 — 单个作业失败不应终止整批夜跑
            failed += 1
            print(json.dumps({"job_id": job["job_id"], "error": str(exc)[:300]}, ensure_ascii=False), flush=True)
            continue
        job_record = {**job, **roles_meta} if roles_meta else job
        payload = {
            "schema_version": RUN_OUTPUT_SCHEMA_VERSION,
            "complete": True,
            "run_fingerprint": run_fingerprint,
            "job_fingerprint": job_fingerprint,
            "fingerprint_inputs": fingerprint_inputs,
            "job": job_record,
            "trace": trace_to_dict(trace),
            "metrics": compute_metrics(trace),
        }
        _atomic_write_json(out_path, payload)
        done += 1
    print(json.dumps({
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "total": len(jobs),
        "run_fingerprint": run_fingerprint,
    }, ensure_ascii=False))
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# judge-runs: offline blind judge batch (REDESIGN_PLAN.md §4/§6,
# draft_codex.md §6, draft_opus.md §6). Runs entirely outside run-manifest:
# it only *reads* frozen `*.json` results and writes independent
# `<file>.judges.json` + `judge_manifest.json`; it never touches the original
# result files.
# ---------------------------------------------------------------------------

JUDGE_RETRIES = 2


def _collect_voluntary_samples(results_dir: Path) -> list[dict]:
    """Scan results_dir/*.json for voluntary, explicitly-submitted rounds.

    Only submission_origin=="voluntary" rounds with a non-empty
    submitted_product enter the judge pool (forced_prompt/none never do —
    the same voluntary-only boundary the primary economic metrics use).
    """
    samples: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.endswith(".judges.json") or path.name == "judge_manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        trace = payload.get("trace")
        if not isinstance(trace, dict):
            continue
        job = payload.get("job") or {}
        model = job.get("model") or payload.get("model") or ""
        condition = job.get("condition") or trace.get("condition") or ""
        episode_id = job.get("episode_id") or trace.get("episode_id") or ""
        job_id = job.get("job_id") or path.stem
        for round_dict in trace.get("rounds") or []:
            if round_dict.get("submission_origin") != "voluntary":
                continue
            if not round_dict.get("submitted_product"):
                continue
            round_num = round_dict.get("round_num")
            samples.append({
                "submission_id": f"{job_id}::round{round_num}",
                "file": path.name,
                "job_id": job_id,
                "model": model,
                "condition": condition,
                "episode_id": episode_id,
                "round_num": round_num,
                "round": round_dict,
            })
    return samples


def _stratified_sample(samples: list[dict], sample_size: int, seed: int) -> list[dict]:
    """Deterministic (condition, model)-balanced sample, frozen by seed.

    Each stratum is sorted by submission_id (removes filesystem-glob-order
    nondeterminism) then shuffled with a seed derived from the stratum key,
    so the same (samples, sample_size, seed) always yields the same picks
    regardless of input ordering. Strata are then drawn round-robin (in
    sorted key order) until sample_size is reached or every stratum is
    exhausted, so no single (condition, model) cell can dominate the batch.
    """
    if sample_size <= 0 or not samples:
        return []
    strata: dict[tuple[str, str], list[dict]] = {}
    for item in samples:
        strata.setdefault((item["condition"], item["model"]), []).append(item)
    for key, items in strata.items():
        items.sort(key=lambda item: item["submission_id"])
        rng = random.Random(derive_seed("mirage.judge.sample.stratum", key[0], key[1], seed))
        rng.shuffle(items)
    keys = sorted(strata)
    cursors = {key: 0 for key in keys}
    selected: list[dict] = []
    progressed = True
    while len(selected) < sample_size and progressed:
        progressed = False
        for key in keys:
            if len(selected) >= sample_size:
                break
            cursor = cursors[key]
            if cursor < len(strata[key]):
                selected.append(strata[key][cursor])
                cursors[key] = cursor + 1
                progressed = True
    return selected


async def _judge_call_with_retries(
    client: BaseLLMClient,
    *,
    client_brief: dict,
    product: dict,
    explanation: str,
    model_name: str,
    repeat: int,
    seed: int,
    retries: int = JUDGE_RETRIES,
) -> tuple[JudgeResult | None, str | None]:
    """A single judge call is never allowed to abort the batch: after
    ``retries`` failed retries (temperature is already fixed at 0 inside
    judge_soft_quality) the entry is recorded as missing with its error."""
    last_error: str | None = None
    for _ in range(retries + 1):
        try:
            result = await judge_soft_quality(
                client, client_brief=client_brief, product=product, explanation=explanation,
                model_name=model_name, repeat=repeat, seed=seed,
            )
            return result, None
        except Exception as exc:  # noqa: BLE001 — one bad judge call must not kill the batch
            last_error = str(exc)[:300]
    return None, last_error


async def _run_judge_runs(
    *,
    results_dir: Path,
    judge_models: list[str],
    clients: dict[str, BaseLLMClient],
    repeats: int,
    sample_size: int,
    salt: str,
    seed: int,
) -> dict:
    candidates = _collect_voluntary_samples(results_dir)
    selected = _stratified_sample(candidates, sample_size, seed)

    strata_candidate_counts: dict[str, int] = {}
    for item in candidates:
        key = f"{item['condition']}|{item['model']}"
        strata_candidate_counts[key] = strata_candidate_counts.get(key, 0) + 1

    selected_by_file: dict[str, list[dict]] = {}
    for item in selected:
        selected_by_file.setdefault(item["file"], []).append(item)

    manifest_selected: list[dict] = []
    self_judge_skips: list[dict] = []
    written_files: list[str] = []
    total_calls = 0
    total_missing = 0

    for file_name in sorted(selected_by_file):
        entries: list[dict] = []
        for item in selected_by_file[file_name]:
            round_for_judge = dict(item["round"])
            round_for_judge["submission_id"] = item["submission_id"]
            judge_input = build_judge_input(round_for_judge, salt)
            manifest_selected.append({
                "submission_id": item["submission_id"],
                "blind_id": judge_input.blind_id,
                "file": item["file"],
                "job_id": item["job_id"],
                "model": item["model"],
                "condition": item["condition"],
                "episode_id": item["episode_id"],
                "round_num": item["round_num"],
            })
            for judge_model in judge_models:
                base_entry = {
                    "submission_id": item["submission_id"],
                    "blind_id": judge_input.blind_id,
                    "round_num": item["round_num"],
                    "episode_id": item["episode_id"],
                    "condition": item["condition"],
                    "judge_model": judge_model,
                }
                if judge_model == item["model"]:
                    # No self-judging: a model must never grade its own submission.
                    self_judge_skips.append({
                        "submission_id": item["submission_id"],
                        "model": item["model"],
                        "judge_model": judge_model,
                    })
                    entries.append({
                        **base_entry, "repeat": None, "seed": None,
                        "status": "skipped_self_judge", "error": None, "judge_result": None,
                    })
                    continue
                client = clients[judge_model]
                for repeat in range(repeats):
                    call_seed = derive_seed(
                        "mirage.judge.call", item["submission_id"], judge_model, repeat, seed,
                    )
                    total_calls += 1
                    result, error = await _judge_call_with_retries(
                        client,
                        client_brief=judge_input.client_brief,
                        product=judge_input.product,
                        explanation=judge_input.explanation,
                        model_name=judge_model,
                        repeat=repeat,
                        seed=call_seed,
                    )
                    if result is None:
                        total_missing += 1
                    entries.append({
                        **base_entry, "repeat": repeat, "seed": call_seed,
                        "status": "ok" if result is not None else "missing",
                        "error": error,
                        "judge_result": result.to_dict() if result is not None else None,
                    })
        judges_path = results_dir / f"{Path(file_name).stem}.judges.json"
        judges_payload = {
            "source_file": file_name,
            "salt": salt,
            "seed": seed,
            "judge_models": judge_models,
            "repeats": repeats,
            "entries": entries,
        }
        judges_path.write_text(json.dumps(judges_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written_files.append(judges_path.name)

    manifest = {
        "seed": seed,
        "salt": salt,
        "sample_size_requested": sample_size,
        "sample_size_selected": len(selected),
        "judge_models": judge_models,
        "repeats": repeats,
        "candidates_total": len(candidates),
        "strata_candidate_counts": strata_candidate_counts,
        "selected": manifest_selected,
        "self_judge_skips": self_judge_skips,
    }
    manifest_path = results_dir / "judge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "manifest": str(manifest_path),
        "judges_files": written_files,
        "candidates_total": len(candidates),
        "sample_size_selected": len(selected),
        "judge_calls": total_calls,
        "judge_calls_missing": total_missing,
        "self_judge_skips": len(self_judge_skips),
    }


def _cmd_judge_runs(args: argparse.Namespace) -> int:
    if not args.results_dir.is_dir():
        raise SystemExit(f"results_dir does not exist: {args.results_dir}")
    registry = load_model_registry(args.models_config)

    judge_models = args.judge_models
    repeats = args.repeats
    if args.roles_config is not None:
        try:
            judges_config = load_judges_config(args.roles_config, registry)
        except RoleConfigError as exc:
            raise SystemExit(f"failed to load roles config {args.roles_config}: {exc}")
        if judge_models is None:
            if len(judges_config.models) != 2:
                raise SystemExit(
                    f"{args.roles_config} judges.models has {len(judges_config.models)} entries; "
                    "pass --judge-models explicitly to pick exactly two"
                )
            judge_models = list(judges_config.models)
        if repeats is None:
            repeats = judges_config.repeats
    if judge_models is None:
        raise SystemExit("--judge-models is required unless --roles-config supplies judges.models")
    if repeats is None:
        repeats = 3

    if len(set(judge_models)) != 2:
        raise SystemExit("--judge-models must name exactly two distinct models")
    clients = {name: create_client(name, registry) for name in judge_models}
    summary = asyncio.run(_run_judge_runs(
        results_dir=args.results_dir,
        judge_models=list(judge_models),
        clients=clients,
        repeats=repeats,
        sample_size=args.sample,
        salt=args.salt,
        seed=args.seed,
    ))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


# v2 canonical primary/diagnostic metric names (see REDESIGN_PLAN.md §4 and
# draft_codex.md §5/§9). Older or partially-finished result directories may
# be missing some of these (runner migration in progress); aggregate reads
# whatever is present and reports the rest as missing rather than failing.
CANONICAL_METRICS = (
    "hard_execution_rate",
    "hard_execution_rate_given_submission",
    "contract_acceptance_rate_given_hard_pass",
    "settlement_acceptance_rate",
    "total_dealer_margin",
    "mean_dealer_margin_per_voluntary_accepted_trade",
    "one_step_attainment",
    "voluntary_submission_rate",
    "forced_prompt_rate",
    "no_submission_rate",
)


def _judge_result_from_entry(entry: dict) -> JudgeResult | None:
    payload = entry.get("judge_result")
    if not isinstance(payload, dict):
        return None
    dims_payload = payload.get("dimensions") or {}
    dimensions = {
        name: DimensionScore(
            score=item.get("score"), evidence=item.get("evidence", ""), reason=item.get("reason", ""),
        )
        for name, item in dims_payload.items()
    }
    return JudgeResult(dimensions, model=payload.get("model", ""), repeat=payload.get("repeat", 0))


def _load_judge_entries(results_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for path in sorted(results_dir.glob("*.judges.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries.extend(payload.get("entries") or [])
    return entries


def _judge_markdown_lines(results_dir: Path) -> list[str]:
    """Six-dimension median/IQR/missing plus inter-judge reliability, appended
    to the aggregate markdown when *.judges.json files exist (REDESIGN_PLAN.md
    §4/§6: judge dimensions are reported separately, never merged into the
    economic ranking)."""
    entries = _load_judge_entries(results_dir)
    ok_entries = [entry for entry in entries if entry.get("status") == "ok"]
    if not ok_entries:
        return []
    lines = ["", "## Judge (offline blind review; six dimensions, no composite score)", ""]

    all_results = [r for r in (_judge_result_from_entry(e) for e in ok_entries) if r is not None]
    bundle = aggregate_judge_bundle(all_results)
    lines.append(
        f"pooled across {len(all_results)} ok judge calls "
        f"({len(entries) - len(ok_entries)} missing/skipped of {len(entries)} total)"
    )
    lines.append("")
    header = ["dimension", "median", "iqr", "n_scored", "n_total", "missing_rate"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for name in DIMENSIONS:
        cell = bundle[name]
        lines.append("| " + " | ".join([
            name,
            f"{cell['median']:.3g}" if cell["median"] is not None else "-",
            f"{cell['iqr']:.3g}" if cell["iqr"] is not None else "-",
            str(cell["n_scored"]), str(cell["n_total"]),
            f"{cell['missing_rate']:.3g}",
        ]) + " |")

    # Inter-judge reliability on aligned (submission_id, repeat) pairs across the
    # two judge models present -- reliability only, never claimed as expert validity.
    judge_models = sorted({entry["judge_model"] for entry in ok_entries})
    if len(judge_models) == 2:
        by_key: dict[tuple, dict[str, JudgeResult]] = {}
        for entry in ok_entries:
            result = _judge_result_from_entry(entry)
            if result is None:
                continue
            key = (entry["submission_id"], entry["repeat"])
            by_key.setdefault(key, {})[entry["judge_model"]] = result
        left_model, right_model = judge_models
        left: list[JudgeResult] = []
        right: list[JudgeResult] = []
        for mapping in by_key.values():
            if left_model not in mapping or right_model not in mapping:
                continue
            l_result, r_result = mapping[left_model], mapping[right_model]
            if l_result.total is None or r_result.total is None:
                continue
            if any(
                l_result.dimensions[name].score is None or r_result.dimensions[name].score is None
                for name in DIMENSIONS
            ):
                continue
            left.append(l_result)
            right.append(r_result)
        lines.append("")
        lines.append(f"### Inter-judge reliability: {left_model} vs {right_model}")
        lines.append("")
        if len(left) >= 2:
            report = reliability_summary(left, right)
            lines.append(
                f"n={report['n']} aligned fully-scored pairs; "
                f"exact_total_agreement={report['exact_total_agreement']:.3g}; "
                f"spearman_total={report['spearman_total']:.3g}; {report['claim']}"
            )
            lines.append("")
            kappa_header = ["dimension", "weighted_kappa"]
            lines.append("| " + " | ".join(kappa_header) + " |")
            lines.append("|" + "---|" * len(kappa_header))
            for name in DIMENSIONS:
                lines.append(f"| {name} | {report['dimension_weighted_kappa'][name]:.3g} |")
        else:
            lines.append(f"insufficient aligned fully-scored pairs for reliability metrics (n={len(left)})")
    return lines


def _cmd_aggregate(args: argparse.Namespace) -> int:
    rows: list[dict] = []
    run_fingerprints: set[str] = set()
    legacy_unfingerprinted = 0
    for path in sorted(args.results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            continue
        fingerprint = payload.get("run_fingerprint")
        if fingerprint is None:
            legacy_unfingerprinted += 1
        elif isinstance(fingerprint, str) and fingerprint:
            run_fingerprints.add(fingerprint)
        else:
            raise SystemExit(f"invalid run_fingerprint in result: {path}")
        if (
            payload.get("schema_version") == RUN_OUTPUT_SCHEMA_VERSION
            and payload.get("complete") is not True
        ):
            raise SystemExit(f"incomplete v3 result cannot be aggregated: {path}")
        job = payload.get("job") or {}
        rows.append({
            "file": path.name,
            "model": job.get("model") or payload.get("model") or "",
            "strategy": job.get("strategy"),
            "run_fingerprint": fingerprint,
            **metrics,
        })
    if not rows:
        raise SystemExit(f"no result json with metrics found in {args.results_dir}")
    if len(run_fingerprints) > 1 or (run_fingerprints and legacy_unfingerprinted):
        details = sorted(run_fingerprints)
        raise SystemExit(
            "refusing to aggregate mixed run fingerprints: "
            f"fingerprints={details}, legacy_unfingerprinted={legacy_unfingerprinted}"
        )

    missing_counts = {
        name: sum(1 for row in rows if not isinstance(row.get(name), (int, float)))
        for name in CANONICAL_METRICS
    }
    fully_missing = sorted(name for name, count in missing_counts.items() if count == len(rows))
    partially_missing = sorted(
        name for name, count in missing_counts.items() if 0 < count < len(rows)
    )

    fieldnames = sorted(
        {key for row in rows for key in row},
        key=lambda name: (name != "file", name != "model", name),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })
    if args.output_md:
        lines = _aggregate_markdown(rows, args, fully_missing, partially_missing)
        lines.extend(_judge_markdown_lines(args.results_dir))
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "csv": str(args.output_csv),
        "md": str(args.output_md) if args.output_md else None,
        "run_fingerprint": next(iter(run_fingerprints), None),
        "legacy_unfingerprinted_rows": legacy_unfingerprinted,
        "fully_missing_metrics": fully_missing,
        "partially_missing_metrics": partially_missing,
    }, ensure_ascii=False))
    return 0


def _aggregate_markdown(
    rows: list[dict], args: argparse.Namespace, fully_missing: list[str], partially_missing: list[str],
) -> list[str]:
    available_metrics = [name for name in CANONICAL_METRICS if name not in fully_missing]

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"] or "?", row.get("condition") or "?"), []).append(row)

    lines = ["# Aggregate report", ""]
    if fully_missing:
        lines.append(
            "> missing metrics (absent from every result file in this directory -- "
            "likely an old-schema or partially-finished run): " + ", ".join(fully_missing)
        )
    if partially_missing:
        lines.append(
            "> partially missing metrics (absent from some result files; those rows "
            "are skipped for the metric): " + ", ".join(partially_missing)
        )
    if fully_missing or partially_missing:
        lines.append("")

    lines.append("## Per (model, condition): mean +/- 95% CI (cluster bootstrap over episode_id)")
    lines.append("")
    header = ["model", "condition", "n_rows", "n_episodes"] + available_metrics
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for (model, condition), group in sorted(groups.items()):
        n_episodes = len({row.get("episode_id") for row in group if row.get("episode_id") is not None})
        cells = [model, condition, str(len(group)), str(n_episodes)]
        for name in available_metrics:
            usable = [row for row in group if isinstance(row.get(name), (int, float))]
            if not usable:
                cells.append("missing")
                continue
            metric_seed = derive_seed("mirage.aggregate.cell_ci", model, condition, name, args.seed)
            mean, lo, hi = cluster_bootstrap_ci(
                usable, name, cluster_key="episode_id",
                n_resamples=args.bootstrap_resamples, alpha=args.alpha, seed=metric_seed,
            )
            cells.append(f"{mean:.4g} [{lo:.4g}, {hi:.4g}]")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Condition contrasts per model (dynamic degradation, partial-observability degradation)")
    lines.append(
        "episode-level matched differences; paired Wilcoxon signed-rank + sign-flip "
        "permutation p-values; `wilcoxon_p_holm` Holm-corrected within each model x "
        "metric's 2-contrast family (`*` marks holm p < alpha)"
    )
    lines.append("")
    contrast_header = [
        "model", "metric", "contrast", "mean_drop", "ci_95", "n_pairs",
        "wilcoxon_p", "wilcoxon_p_holm", "permutation_p",
    ]
    lines.append("| " + " | ".join(contrast_header) + " |")
    lines.append("|" + "---|" * len(contrast_header))
    models = sorted({row["model"] for row in rows if row["model"]})
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        for metric in available_metrics:
            usable = [
                row for row in model_rows
                if isinstance(row.get(metric), (int, float))
                and {"episode_id", "strategy", "condition"} <= set(row)
            ]
            if not usable:
                continue
            contrast_seed = derive_seed("mirage.aggregate.contrast", model, metric, args.seed)
            try:
                result = paired_condition_contrasts(
                    usable, metric, seed=contrast_seed,
                    n_resamples=args.bootstrap_resamples, n_permutations=args.n_permutations,
                )
            except BenchmarkError:
                continue
            holm = result["holm_adjusted_wilcoxon_p"]
            for contrast_name in ("dynamic_degradation", "partial_observability_degradation"):
                stats = result[contrast_name]
                if not stats["n_pairs"]:
                    continue
                ci = stats["ci"]
                ci_text = f"[{ci[0]:.4g}, {ci[1]:.4g}]" if ci else "-"
                holm_p = holm.get(contrast_name)
                flag = "*" if holm_p is not None and holm_p < args.alpha else ""
                lines.append("| " + " | ".join([
                    model, metric, contrast_name,
                    f"{stats['mean']:.4g}" if stats["mean"] is not None else "-",
                    ci_text,
                    str(stats["n_pairs"]),
                    f"{stats['wilcoxon_p']:.4g}" if stats["wilcoxon_p"] is not None else "-",
                    f"{holm_p:.4g}{flag}" if holm_p is not None else "-",
                    f"{stats['permutation_p']:.4g}" if stats["permutation_p"] is not None else "-",
                ]) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "aggregate":
        return _cmd_aggregate(args)
    if args.command == "judge-runs":
        return _cmd_judge_runs(args)
    if args.command == "make-v3-suite":
        return _cmd_make_v3_suite(args)
    if args.command == "run-v3-suite":
        return _cmd_run_v3_suite(args)
    if args.command == "evaluate-trajectory":
        return _cmd_evaluate_trajectory(args)
    if args.command == "aggregate-v3":
        return _cmd_aggregate_v3(args)
    if args.command == "make-manifest":
        payload = manifest_payload(
            build_experiment_manifest(args.episodes, args.models, strategies=tuple(args.strategies))
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"jobs": payload["count"], "output": str(args.output)}, ensure_ascii=False))
        return 0
    if args.command == "build-market":
        daily = load_daily_closes(args.daily_csv)
        built = build_monthly_snapshots(daily, risk_free_rate=args.risk_free_rate)
        write_market_snapshots(args.output_csv, built)
        print(json.dumps({"rows": len(built), "output": str(args.output_csv)}, ensure_ascii=False))
        return 0
    snapshots = load_market_snapshots(args.csv)
    if args.command == "validate-market":
        episodes = sorted({snapshot.episode_id for snapshot in snapshots})
        print(json.dumps({"rows": len(snapshots), "episodes": episodes}, ensure_ascii=False))
        return 0
    if args.command == "calibrate-budget":
        return _cmd_calibrate(args, snapshots)
    if args.command == "calibrate-margin":
        return _cmd_calibrate_margin(args, snapshots)
    if args.command == "run-manifest":
        return _cmd_run_manifest(args, snapshots)
    if args.command == "test-agent":
        return _cmd_test_agent(args, snapshots)

    selected = [snapshot for snapshot in snapshots if snapshot.episode_id == args.episode]
    if not selected:
        raise SystemExit(f"unknown episode: {args.episode}")
    if args.command == "run-episode":
        client_payload = json.loads(args.client_json.read_text(encoding="utf-8"))
        budget_payload = json.loads(args.risk_budget_json.read_text(encoding="utf-8"))
        registry = load_model_registry(args.models_config)
        env_agents = None
        roles_meta: dict = {}
        if args.roles_config is not None:
            specs = _load_env_role_specs(args.roles_config, registry)
            env_agents = _make_env_agents(specs, registry, _env_cache(args.env_cache_dir))
            npc_lineup_id, roles_sha = _roles_config_meta(args.roles_config)
            roles_meta = {"npc_lineup_id": npc_lineup_id, "roles_config_sha256": roles_sha}
        episode_env = LongHorizonEnvironment(
            selected,
            ClientProfile(**client_payload),
            RiskBudget(**budget_payload),
            BenchmarkCondition(full_information=args.full, dynamic=not args.static),
            quote_policy=_load_quote_policy(args.quote_policy_json),
            env_agents=env_agents,
        )
        llm = create_client(args.model, registry)
        trace = asyncio.run(run_episode(episode_env, llm, strategy=args.strategy, seed=args.seed))
        payload = {"model": args.model, "trace": trace_to_dict(trace), "metrics": compute_metrics(trace)}
        if roles_meta:
            payload.update(roles_meta)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "metrics": payload["metrics"]}, ensure_ascii=False))
        return 0

    product = parse_product_spec(json.loads(args.product_json.read_text(encoding="utf-8")))
    env = LongHorizonEnvironment(
        selected,
        _client(),
        _budget(),
        BenchmarkCondition(full_information=args.full, dynamic=not args.static),
    )
    payload = {
        "brief": env.get_round_brief(),
        "quote": env.request_quote(product),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
