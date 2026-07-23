"""CLI-level wiring: calibrate-margin end-to-end, --quote-policy-json plumbing,
and judge-runs reading its judge-model/repeats defaults from
config/benchmark_roles.yaml's judges: block (REDESIGN_PLAN.md items 1 and 2
of the pre-freeze cleanup)."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pytest

from mirage.benchmark import (
    BenchmarkCondition,
    LongHorizonEnvironment,
    MarketSnapshot,
    PortfolioState,
    ProductDomainSpec,
    RiskBudget,
    calibrate_risk_budget,
    enumerate_domain,
    oracle_candidate_grid,
)
from mirage.benchmark_cli import (
    RUN_OUTPUT_SCHEMA_VERSION,
    _atomic_write_json,
    _cmd_calibrate_margin,
    _cmd_aggregate,
    _cmd_judge_runs,
    _completed_result_matches,
    _load_quote_policy,
)
from mirage.pricing import QuotePolicy
from mirage.products import ClientProfile, ProductSpec

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
    base = dict(
        csv=tmp_path / "unused.csv",
        episodes=["E1"],
        client_json=client_json,
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
    import mirage.benchmark_cli as cli_mod

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
    import mirage.benchmark_cli as cli_mod

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
    import mirage.benchmark_cli as cli_mod

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


def test_completed_result_is_skippable_only_for_exact_fingerprints(tmp_path):
    path = tmp_path / "job.json"
    payload = {
        "schema_version": RUN_OUTPUT_SCHEMA_VERSION,
        "complete": True,
        "run_fingerprint": "run-a",
        "job_fingerprint": "job-a",
        "job": {"job_id": "J"},
        "trace": {},
        "metrics": {},
    }
    _atomic_write_json(path, payload)

    assert _completed_result_matches(
        path,
        run_fingerprint="run-a",
        job_fingerprint="job-a",
        job_id="J",
    )
    assert not _completed_result_matches(
        path,
        run_fingerprint="run-b",
        job_fingerprint="job-a",
        job_id="J",
    )

    payload["complete"] = False
    _atomic_write_json(path, payload)
    assert not _completed_result_matches(
        path,
        run_fingerprint="run-a",
        job_fingerprint="job-a",
        job_id="J",
    )


def test_aggregate_rejects_mixed_run_fingerprints(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for index, fingerprint in enumerate(("run-a", "run-b")):
        _atomic_write_json(
            results / f"{index}.json",
            {
                "schema_version": RUN_OUTPUT_SCHEMA_VERSION,
                "complete": True,
                "run_fingerprint": fingerprint,
                "job_fingerprint": f"job-{index}",
                "job": {"job_id": str(index), "model": "m", "strategy": "s"},
                "trace": {},
                "metrics": {"episode_id": "E", "condition": "full_static"},
            },
        )
    args = argparse.Namespace(
        results_dir=results,
        output_csv=tmp_path / "aggregate.csv",
        output_md=None,
        bootstrap_resamples=10,
        n_permutations=10,
        alpha=0.05,
        seed=1,
    )

    with pytest.raises(SystemExit, match="mixed run fingerprints"):
        _cmd_aggregate(args)


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
