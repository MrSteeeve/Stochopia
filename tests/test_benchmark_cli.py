"""CLI-level wiring: calibrate-margin end-to-end, --quote-policy-json plumbing,
and judge-runs reading its judge-model/repeats defaults from
config/benchmark_roles.yaml's judges: block (pre-freeze cleanup items 1 and 2)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pytest

from stochopia.benchmark import (
    BenchmarkCondition,
    LongHorizonEnvironment,
    MarketSnapshot,
    PortfolioState,
    ProductDomainSpec,
    RiskBudget,
    calibrate_risk_budget,
    enumerate_domain,
)
from stochopia.benchmark_cli import (
    RUN_OUTPUT_SCHEMA_VERSION,
    _atomic_write_json,
    _cmd_calibrate_margin,
    _cmd_judge_runs,
    _completed_result_matches,
    _load_quote_policy,
)
from stochopia.pricing import QuotePolicy
from stochopia.products import ClientProfile, ProductSpec

ROOT = Path(__file__).resolve().parents[1]

# Deliberately tiny lattice so the calibrate-margin test stays well under a
# second: enumerate_domain() over the real ProductDomainSpec default is tens
# of thousands of candidates, far more than a unit test needs.
SMALL_DOMAIN = ProductDomainSpec(
    product_types=("vanilla_call", "snowball"),
    notional_fractions=(.05, .10),
    maturities=(3, 6),
    strikes=(1.00,),
    barriers=(.85,),
    coupons=(.08,),
    participations=(1.0,),
    principal_protected=(False,),
)


def _client() -> ClientProfile:
    return ClientProfile(
        id="dev_client",
        name="Dev Client",
        capital=20_000_000,
        max_loss_pct=1.0,
        min_return_pct=0.02,
        risk_appetite="moderate",
        max_maturity_months=12,
        min_hit_prob=0.0,
        preferences="test fixture",
    )


def _snapshots(episode: str, n_rounds: int = 2) -> list[MarketSnapshot]:
    return [
        MarketSnapshot(
            episode_id=episode,
            round_num=i,
            as_of=date(2023, i, 28),
            underlying="CSI500",
            spot=6000.0 + 50.0 * i,
            risk_free_rate=0.02,
            realized_vol_20d=0.22,
            realized_vol_60d=0.20,
            regime="sideways",
            source="test-fixture",
        )
        for i in range(1, n_rounds + 1)
    ]


def _margin_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    client_json = tmp_path / "client.json"
    client_json.write_text(json.dumps(asdict(_client())), encoding="utf-8")
    budget_json = tmp_path / "risk_budget.json"
    budget_json.write_text(
        json.dumps(asdict(RiskBudget(50_000_000, 50_000_000, 100_000_000, 10_000_000, 50_000_000))),
        encoding="utf-8",
    )
    base = dict(
        csv=tmp_path / "unused.csv",
        episodes=["E1"],
        client_json=client_json,
        risk_budget_json=budget_json,
        sensitivity_factors=[0.8, 1.0, 1.2],
        report_output=tmp_path / "report.json",
        policy_output=tmp_path / "policy.json",
        candidates_per_snapshot=4,
        factors=[0.5, 1.0, 2.0],
        seed=7,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# calibrate-margin
# ---------------------------------------------------------------------------


def test_calibrate_margin_end_to_end_report_structure(tmp_path, monkeypatch):
    import stochopia.benchmark_cli as cli_mod

    monkeypatch.setattr(cli_mod, "ProductDomainSpec", lambda: SMALL_DOMAIN)

    snapshots = _snapshots("E1", n_rounds=2)
    args = _margin_args(tmp_path)

    rc = _cmd_calibrate_margin(args, snapshots)
    assert rc == 0

    report = json.loads(args.report_output.read_text(encoding="utf-8"))
    for key in (
        "selected_factor", "selected_positive_margin_rate", "ranking_nondegenerate",
        "target", "within_target", "grid", "warning", "development_episodes",
        "candidates_per_snapshot", "n_snapshots", "n_dev_cases", "seed", "sensitivity",
    ):
        assert key in report
    assert report["development_episodes"] == ["E1"]
    assert report["n_snapshots"] == 2
    assert report["n_dev_cases"] == 2 * min(args.candidates_per_snapshot, len(list(enumerate_domain(_client(), SMALL_DOMAIN))))
    assert report["candidates_per_snapshot"] == 4
    assert "freeze" in report["warning"]
    assert len(report["sensitivity"]) == 3
    for row in report["sensitivity"]:
        assert set(row) == {
            "sensitivity_factor",
            "positive_margin_rate",
            "ranking_nondegenerate",
            "n_cases",
            "nonconverged_quotes",
        }
    assert {row["sensitivity_factor"] for row in report["sensitivity"]} == {0.8, 1.0, 1.2}

    policy_payload = json.loads(args.policy_output.read_text(encoding="utf-8"))
    calibrated = QuotePolicy(**policy_payload)
    assert isinstance(calibrated, QuotePolicy)


def test_calibrate_margin_is_deterministic(tmp_path, monkeypatch):
    import stochopia.benchmark_cli as cli_mod

    monkeypatch.setattr(cli_mod, "ProductDomainSpec", lambda: SMALL_DOMAIN)
    snapshots = _snapshots("E1", n_rounds=2)

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    args_a = _margin_args(tmp_path / "a")
    args_b = _margin_args(tmp_path / "b")

    _cmd_calibrate_margin(args_a, snapshots)
    _cmd_calibrate_margin(args_b, snapshots)

    report_a = json.loads(args_a.report_output.read_text(encoding="utf-8"))
    report_b = json.loads(args_b.report_output.read_text(encoding="utf-8"))
    assert report_a == report_b

    policy_a = json.loads(args_a.policy_output.read_text(encoding="utf-8"))
    policy_b = json.loads(args_b.policy_output.read_text(encoding="utf-8"))
    assert policy_a == policy_b


def test_calibrate_margin_unknown_episode_rejected(tmp_path, monkeypatch):
    import stochopia.benchmark_cli as cli_mod

    monkeypatch.setattr(cli_mod, "ProductDomainSpec", lambda: SMALL_DOMAIN)
    snapshots = _snapshots("E1")
    args = _margin_args(tmp_path, episodes=["NOT_A_REAL_EPISODE"])
    with pytest.raises(SystemExit, match="unknown development episodes"):
        _cmd_calibrate_margin(args, snapshots)


# ---------------------------------------------------------------------------
# --quote-policy-json: loading + wiring parity with the omitted default
# ---------------------------------------------------------------------------


def test_load_quote_policy_none_when_omitted():
    assert _load_quote_policy(None) is None


def test_load_quote_policy_round_trips_default(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(asdict(QuotePolicy())), encoding="utf-8")
    loaded = _load_quote_policy(path)
    assert loaded == QuotePolicy()


def test_quote_policy_json_default_matches_omitted_default(tmp_path):
    """A JSON file containing the default QuotePolicy must quote identically
    to omitting --quote-policy-json altogether (LongHorizonEnvironment's own
    ``quote_policy or QuotePolicy()`` fallback)."""
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(asdict(QuotePolicy())), encoding="utf-8")

    snapshots = _snapshots("E1", n_rounds=1)
    client = _client()
    budget = RiskBudget(50_000_000, 50_000_000, 100_000_000, 10_000_000, 50_000_000)
    prod = ProductSpec(
        product_type="vanilla_call", notional=1_000_000, maturity_months=6,
        strike_pct=1.0, barrier_pct=None, barrier_type=None, coupon_rate=None,
        participation_rate=1.0, principal_protected=False, target_client=client.id,
        pitch="p", hedging_plan="h",
    )

    env_omitted = LongHorizonEnvironment(snapshots, client, budget, BenchmarkCondition(True, False))
    env_loaded = LongHorizonEnvironment(
        snapshots, client, budget, BenchmarkCondition(True, False),
        quote_policy=_load_quote_policy(path),
    )

    quote_omitted = env_omitted.desk.quote(prod, snapshots[0], client, PortfolioState(), "s", 1)
    quote_loaded = env_loaded.desk.quote(prod, snapshots[0], client, PortfolioState(), "s", 1)

    assert quote_omitted.dealer_margin == pytest.approx(quote_loaded.dealer_margin)
    assert quote_omitted.client_price == pytest.approx(quote_loaded.client_price)
    assert quote_omitted.fair_value == pytest.approx(quote_loaded.fair_value)


def test_quote_policy_json_nondefault_actually_changes_the_quote(tmp_path):
    """Sanity check that --quote-policy-json is really wired through (not
    silently ignored): a policy with markedly larger markup coefficients must
    change dealer_margin relative to the default."""
    scaled = QuotePolicy(a_f=QuotePolicy().a_f * 5, a_v=QuotePolicy().a_v * 5,
                          a_p=QuotePolicy().a_p * 5, a_b=QuotePolicy().a_b * 5)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(asdict(scaled)), encoding="utf-8")

    snapshots = _snapshots("E1", n_rounds=1)
    client = _client()
    budget = RiskBudget(50_000_000, 50_000_000, 100_000_000, 10_000_000, 50_000_000)
    prod = ProductSpec(
        product_type="vanilla_call", notional=1_000_000, maturity_months=6,
        strike_pct=1.0, barrier_pct=None, barrier_type=None, coupon_rate=None,
        participation_rate=1.0, principal_protected=False, target_client=client.id,
        pitch="p", hedging_plan="h",
    )

    env_default = LongHorizonEnvironment(snapshots, client, budget, BenchmarkCondition(True, False))
    env_scaled = LongHorizonEnvironment(
        snapshots, client, budget, BenchmarkCondition(True, False),
        quote_policy=_load_quote_policy(path),
    )

    quote_default = env_default.desk.quote(prod, snapshots[0], client, PortfolioState(), "s", 1)
    quote_scaled = env_scaled.desk.quote(prod, snapshots[0], client, PortfolioState(), "s", 1)

    assert quote_scaled.dealer_margin != pytest.approx(quote_default.dealer_margin)


def test_calibrate_budget_quote_policy_kwarg_accepted():
    """calibrate_risk_budget's new ``policy`` kwarg (wired from
    --quote-policy-json in calibrate-budget) must not change its call
    contract: passing a custom policy still returns a well-formed report."""
    client = _client()
    products = list(enumerate_domain(client, SMALL_DOMAIN))
    snapshots = _snapshots("E1", n_rounds=1)
    base = RiskBudget(10_000_000, 10_000_000, 20_000_000, 2_000_000, 10_000_000)

    report = calibrate_risk_budget(
        products,
        [(snapshots[0], client, PortfolioState())],
        base,
        target=(0.1, 0.9),
        factors=(0.5, 1.0, 2.0),
        policy=QuotePolicy(),
    )
    assert "selected_factor" in report
    assert "freeze" in report["warning"]


# ---------------------------------------------------------------------------
# run-manifest durability / provenance
# ---------------------------------------------------------------------------


def test_completed_result_is_skippable_only_when_complete(tmp_path):
    path = tmp_path / "job.json"
    payload = {
        "schema_version": RUN_OUTPUT_SCHEMA_VERSION,
        "complete": True,
        "job": {"job_id": "J"},
        "trace": {},
        "metrics": {},
    }
    _atomic_write_json(path, payload)

    assert _completed_result_matches(path, job_id="J")
    assert not _completed_result_matches(path, job_id="K")

    payload["complete"] = False
    _atomic_write_json(path, payload)
    assert not _completed_result_matches(path, job_id="J")


# ---------------------------------------------------------------------------
# judge-runs: reads defaults from config/benchmark_roles.yaml's judges: block
# ---------------------------------------------------------------------------


def _judge_runs_args(results_dir: Path, **overrides) -> argparse.Namespace:
    base = dict(
        results_dir=results_dir,
        judge_models=None,
        repeats=None,
        roles_config=None,
        sample=5,
        salt="test-salt",
        seed=1,
        models_config=ROOT / "config" / "models.yaml",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_judge_runs_cli_reads_defaults_from_roles_config(tmp_path):
    # No result files -> _run_judge_runs never calls a model's .chat(), so this
    # exercises only the CLI-level judge_models/repeats resolution.
    args = _judge_runs_args(
        tmp_path, roles_config=ROOT / "config" / "benchmark_roles.yaml",
    )
    rc = _cmd_judge_runs(args)
    assert rc == 0
    manifest = json.loads((tmp_path / "judge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_models"] == ["deepseek-v4-pro", "qwen-max"]
    assert manifest["repeats"] == 3


def test_judge_runs_cli_explicit_flags_override_roles_config(tmp_path):
    args = _judge_runs_args(
        tmp_path,
        roles_config=ROOT / "config" / "benchmark_roles.yaml",
        judge_models=["deepseek-v4-flash", "qwen-max"],
        repeats=7,
    )
    rc = _cmd_judge_runs(args)
    assert rc == 0
    manifest = json.loads((tmp_path / "judge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_models"] == ["deepseek-v4-flash", "qwen-max"]
    assert manifest["repeats"] == 7


def test_judge_runs_cli_requires_judge_models_without_roles_config(tmp_path):
    args = _judge_runs_args(tmp_path)
    with pytest.raises(SystemExit, match="judge-models"):
        _cmd_judge_runs(args)


# ---------------------------------------------------------------------------
# Review fixes: v2 result inputs, resume, per-job metrics, n<2 CI rendering
# ---------------------------------------------------------------------------


def _run_manifest_args(tmp_path, manifest_path, *, client_json=None):
    return argparse.Namespace(
        csv=ROOT / "scenarios" / "stochopia_csi" / "market_snapshots.example.csv",
        manifest=manifest_path,
        only_models=None,
        client_json=client_json or ROOT / "scenarios" / "stochopia_csi" / "client.example.json",
        risk_budget_json=ROOT / "scenarios" / "stochopia_csi" / "risk_budget.example.json",
        quote_policy_json=None,
        models_config=ROOT / "config" / "models.yaml",
        roles_config=None,
        env_cache_dir=None,
        outputs_dir=tmp_path / "out",
    )


def test_run_manifest_records_inputs_resumes_on_match_and_isolates_metric_errors(
    tmp_path, monkeypatch,
):
    import stochopia.benchmark_cli as cli
    from stochopia.benchmark import load_market_snapshots
    from stochopia.benchmark_runner import EpisodeTrace

    jobs = [
        {"job_id": job_id, "episode_id": "SYNTHETIC_CSI500_DEMO", "model": "mock",
         "strategy": "quote_and_revise", "condition": "partial_dynamic", "seed": 1}
        for job_id in ("J1", "J2")
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")

    async def fake_run_episode(env, llm, *, strategy, seed):
        return EpisodeTrace(env.snapshot.episode_id, "partial_dynamic", strategy, [], {}, seed)

    calls = {"n": 0}

    def flaky_metrics(trace):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ZeroDivisionError("metric bug on first job")
        return {"episode_id": trace.episode_id}

    monkeypatch.setattr(cli, "run_episode", fake_run_episode)
    monkeypatch.setattr(cli, "compute_metrics", flaky_metrics)
    snapshots = load_market_snapshots(ROOT / "scenarios" / "stochopia_csi" / "market_snapshots.example.csv")
    args = _run_manifest_args(tmp_path, manifest_path)

    assert cli._cmd_run_manifest(args, snapshots) == 1  # J1 failed, J2 written
    # A failed job leaves a visible, non-resumable record.
    failed_payload = json.loads((args.outputs_dir / "J1.json").read_text(encoding="utf-8"))
    assert failed_payload["complete"] is False
    assert failed_payload["status"] == "failed"
    assert "metric bug" in failed_payload["error"]
    assert not _completed_result_matches(
        args.outputs_dir / "J1.json", job_id="J1", inputs=failed_payload["inputs"],
    )
    payload = json.loads((args.outputs_dir / "J2.json").read_text(encoding="utf-8"))
    inputs = payload["inputs"]
    assert inputs["market_csv"] == str(args.csv.resolve())
    assert inputs["tested_model"]["provider"] == "mock"
    assert inputs["client_profile"]["id"] == "institutional_demo"
    assert inputs["quote_policy"] == json.loads(json.dumps(asdict(QuotePolicy())))
    assert set(inputs["risk_budget"]) >= {"notional", "net_delta"}
    assert inputs["role_models"] == {}

    # Same inputs -> J2 resumes; changed client profile -> J2 is rerun.
    assert _completed_result_matches(args.outputs_dir / "J2.json", job_id="J2", inputs=inputs)
    changed = json.loads(args.client_json.read_text(encoding="utf-8"))
    changed["max_loss_pct"] = 0.5
    changed_path = tmp_path / "client.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    changed_inputs = cli._economic_inputs(
        args.csv, changed, json.loads(args.risk_budget_json.read_text(encoding="utf-8")), None, None,
    )
    assert not _completed_result_matches(
        args.outputs_dir / "J2.json", job_id="J2", inputs=changed_inputs,
    )


def test_aggregate_markdown_marks_single_episode_ci_not_estimable():
    from stochopia.benchmark_cli import CANONICAL_METRICS, _aggregate_markdown

    rows = [{"file": "a.json", "model": "M", "condition": "partial_dynamic",
             "episode_id": "E1", "strategy": "S", "hard_execution_rate": 0.5}]
    args = argparse.Namespace(seed=1, bootstrap_resamples=50, alpha=0.05, n_permutations=50)
    fully_missing = [name for name in CANONICAL_METRICS if name != "hard_execution_rate"]
    text = "\n".join(_aggregate_markdown(rows, args, fully_missing, []))
    assert "0.5 [CI n/a: 1 episode]" in text
    assert "[0.5, 0.5]" not in text


# ---------------------------------------------------------------------------
# 2026-11-09 (三): aggregate poolability/coverage, alpha, calibrate-margin budget
# ---------------------------------------------------------------------------


def _agg_result(job_id: str, model: str, condition: str, episode: str, *, inputs: dict) -> dict:
    return {
        "schema_version": RUN_OUTPUT_SCHEMA_VERSION,
        "complete": True,
        "job": {"job_id": job_id, "model": model, "condition": condition,
                "episode_id": episode, "strategy": "quote_and_revise"},
        "inputs": {**inputs, "tested_model": {"name": model}},
        "trace": {"condition": condition, "rounds": []},
        "metrics": {"episode_id": episode, "condition": condition,
                    "hard_execution_rate": 0.5,
                    "mean_dealer_margin_per_voluntary_accepted_trade": None},
    }


def _agg_args(results_dir: Path, **overrides) -> argparse.Namespace:
    base = dict(results_dir=results_dir, output_csv=results_dir / "agg.csv",
                output_md=results_dir / "agg.md", bootstrap_resamples=50,
                n_permutations=50, alpha=0.10, seed=1, manifest=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_aggregate_reports_coverage_and_refuses_mixed_inputs(tmp_path):
    from stochopia.benchmark_cli import _cmd_aggregate

    results = tmp_path / "res"
    results.mkdir()
    same = {"market_csv": "/m.csv"}
    _atomic_write_json(results / "J1.json", _agg_result("J1", "M", "full_static", "E1", inputs=same))
    _atomic_write_json(results / "J2.json", _agg_result("J2", "N", "full_static", "E1", inputs=same))
    _atomic_write_json(results / "J3.json", {
        "schema_version": RUN_OUTPUT_SCHEMA_VERSION, "complete": False, "status": "failed",
        "error": "LLMError: 400", "job": {"job_id": "J3", "model": "M", "condition": "partial_static"},
        "inputs": same, "trace": None,
    })
    _atomic_write_json(results / "old.json", {"metrics": {"hard_execution_rate": 1.0}})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"jobs": [
        {"job_id": job, "model": "M", "condition": cond}
        for job, cond in (("J1", "full_static"), ("J3", "partial_static"), ("J4", "partial_static"))
    ]}), encoding="utf-8")

    assert _cmd_aggregate(_agg_args(results, manifest=manifest)) == 0
    md = (results / "agg.md").read_text(encoding="utf-8")
    assert "mean +/- 90% CI" in md and "ci_90%" in md
    assert "old.json" in md and "LLMError: 400" in md
    assert "| M | partial_static | 2 | 0 | 1 | 1 |" in md
    assert "likely an old-schema" not in md

    _atomic_write_json(results / "J5.json", _agg_result(
        "J5", "M", "full_dynamic", "E1", inputs={"market_csv": "/other.csv"}))
    with pytest.raises(SystemExit, match="different inputs"):
        _cmd_aggregate(_agg_args(results))


def test_calibrate_margin_requires_frozen_risk_budget():
    from stochopia.benchmark_cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "calibrate-margin", "m.csv", "--episodes", "E1", "--client-json", "c.json",
            "--report-output", "r.json", "--policy-output", "p.json",
        ])


def test_aggregate_writes_coverage_report_when_no_complete_rows(tmp_path):
    from stochopia.benchmark_cli import _cmd_aggregate

    results = tmp_path / "res"
    results.mkdir()
    _atomic_write_json(results / "J1.json", {
        "schema_version": RUN_OUTPUT_SCHEMA_VERSION, "complete": False, "status": "failed",
        "error": "LLMError: 400", "job": {"job_id": "J1", "model": "M", "condition": "full_static"},
        "inputs": {}, "trace": None,
    })
    with pytest.raises(SystemExit, match="no result json"):
        _cmd_aggregate(_agg_args(results))
    assert "LLMError: 400" in (results / "agg.md").read_text(encoding="utf-8")


def test_run_episode_writes_failed_record_for_any_exception(tmp_path, monkeypatch):
    import stochopia.benchmark_cli as cli

    from stochopia.pricing import PricingError

    async def broken_run_episode(env, llm, *, strategy, seed):
        raise PricingError("pricing bug")

    monkeypatch.setattr(cli, "run_episode", broken_run_episode)
    output = tmp_path / "episode.json"
    with pytest.raises(PricingError):
        cli.main([
            "run-episode", str(ROOT / "scenarios" / "stochopia_csi" / "market_snapshots.example.csv"),
            "--episode", "SYNTHETIC_CSI500_DEMO",
            "--client-json", str(ROOT / "scenarios" / "stochopia_csi" / "client.example.json"),
            "--risk-budget-json", str(ROOT / "scenarios" / "stochopia_csi" / "risk_budget.example.json"),
            "--model", "mock", "--output", str(output),
        ])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["complete"] is False and payload["status"] == "failed"
    assert "pricing bug" in payload["error"] and payload["model"] == "mock"
