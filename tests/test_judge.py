"""Soft judge parsing and reliability metrics, plus the offline judge-runs pipeline."""

from __future__ import annotations

import json

import pytest

from stochopia.benchmark_cli import (
    _collect_voluntary_samples,
    _run_judge_runs,
    _stratified_sample,
)
from stochopia.judge import (
    DIMENSIONS,
    DimensionScore,
    JudgeError,
    JudgeResult,
    aggregate_judge_bundle,
    build_judge_input,
    parse_judge_result,
    reliability_summary,
    spearman_correlation,
    validate_evidence_spans,
    weighted_cohens_kappa,
    score_flip_rate,
)
from stochopia.llm import MockLLMClient


def payload(scores: list[int]) -> str:
    return json.dumps({
        "dimensions": {
            name: {"score": score, "evidence": f"evidence-{name}", "reason": "reason"}
            for name, score in zip(DIMENSIONS, scores)
        }
    })


def test_parse_judge_result_requires_evidence_and_exact_dimensions():
    result = parse_judge_result(payload([0, 1, 2, 3, 4, 4]), model="judge-a", repeat=1)
    assert result.total == 14
    assert result.model == "judge-a"
    with pytest.raises(JudgeError, match="dimensions"):
        parse_judge_result('{"dimensions": {}}')


def test_reliability_metrics_perfect_agreement():
    assert weighted_cohens_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    left = [parse_judge_result(payload([i] * 6)) for i in (0, 1, 2)]
    right = [parse_judge_result(payload([i] * 6)) for i in (0, 1, 2)]
    report = reliability_summary(left, right)
    assert report["exact_total_agreement"] == 1.0
    assert report["spearman_total"] == pytest.approx(1.0)
    assert "not expert-grounded" in report["claim"]
    assert score_flip_rate(left, right) == 0.0


def test_metric_input_validation():
    with pytest.raises(JudgeError):
        weighted_cohens_kappa([], [])
    with pytest.raises(JudgeError):
        spearman_correlation([1], [1])


# ---------------------------------------------------------------------------
# build_judge_input: anti-leakage whitelist + blind_id
# ---------------------------------------------------------------------------

LEAK_MODEL = "LEAK_MODEL_gpt4_should_not_appear"
LEAK_MARGIN = "LEAK_MARGIN_9999"
LEAK_ORACLE = "LEAK_ORACLE_8888"
LEAK_VENDOR = "LEAK_VENDOR_openai"
LEAK_STRATEGY = "LEAK_STRATEGY_quote_and_revise"
LEAK_PASS = "LEAK_PASS_CLIENT_LOSS_BUDGET_V2"


def _leaky_round_dict(submission_id: str = "job-x::round1") -> dict:
    """A round record with leakage attempts at every level: top-level fields
    (model/vendor/strategy/margin), and fields nested *inside*
    client_brief_snapshot/submitted_product that a naive blacklist would miss
    but the whitelist approach must still drop."""
    return {
        "submission_id": submission_id,
        "model": LEAK_MODEL,
        "vendor": LEAK_VENDOR,
        "strategy": LEAK_STRATEGY,
        "dealer_margin": LEAK_MARGIN,
        "oracle_margin": LEAK_ORACLE,
        "hard_failures": [LEAK_PASS],
        "submission_origin": "voluntary",
        "client_brief_snapshot": {
            "episode_id": "E1", "round": 1, "underlying": "CSI300", "spot": 4000.0,
            "condition": "full_static", "quote_budget": 2,
            "model": LEAK_MODEL,
            "margin_hint": LEAK_MARGIN,
        },
        "submitted_product": {
            "product_type": "vanilla_call", "notional": 1_000_000, "maturity_months": 6,
            "strike_pct": 1.0, "barrier_pct": None, "barrier_type": None, "coupon_rate": None,
            "participation_rate": 1.0, "principal_protected": False, "target_client": "client_a",
            "pitch": "structured note", "hedging_plan": "delta hedge",
            "margin": LEAK_MARGIN, "oracle": LEAK_ORACLE, "vendor": LEAK_VENDOR,
        },
        "submitted_explanation": "This note offers upside participation with defined risk.",
    }


def test_build_judge_input_strips_leaking_fields():
    judge_input = build_judge_input(_leaky_round_dict(), salt="salt-1")
    assert judge_input.submission_id == "job-x::round1"
    serialized = json.dumps(judge_input.public_fact_sheet, ensure_ascii=False)
    for leak in (LEAK_MODEL, LEAK_MARGIN, LEAK_ORACLE, LEAK_VENDOR, LEAK_STRATEGY, LEAK_PASS):
        assert leak not in serialized
    assert "model" not in judge_input.public_fact_sheet["client_brief"]
    assert "margin_hint" not in judge_input.public_fact_sheet["client_brief"]
    assert "margin" not in judge_input.public_fact_sheet["product"]
    assert "oracle" not in judge_input.public_fact_sheet["product"]
    assert "vendor" not in judge_input.public_fact_sheet["product"]
    # legitimate client-brief/product fields survive the whitelist
    assert judge_input.public_fact_sheet["client_brief"]["underlying"] == "CSI300"
    assert judge_input.public_fact_sheet["product"]["product_type"] == "vanilla_call"


def test_build_judge_input_requires_submission_id():
    round_dict = _leaky_round_dict()
    del round_dict["submission_id"]
    with pytest.raises(JudgeError):
        build_judge_input(round_dict, salt="salt-1")


def test_blind_id_stable_and_not_reversible():
    round_dict = _leaky_round_dict("secret-submission-id-123")
    a = build_judge_input(round_dict, salt="salt-1")
    b = build_judge_input(round_dict, salt="salt-1")
    assert a.blind_id == b.blind_id
    c = build_judge_input(round_dict, salt="salt-2")
    assert a.blind_id != c.blind_id
    assert len(a.blind_id) == 12
    assert "secret-submission-id-123" not in a.blind_id
    assert a.blind_id != round_dict["submission_id"]


# ---------------------------------------------------------------------------
# evidence span validation
# ---------------------------------------------------------------------------

def _all_scored(score: int, evidence: str) -> dict[str, DimensionScore]:
    return {name: DimensionScore(score=score, evidence=evidence, reason="ok") for name in DIMENSIONS}


def test_validate_evidence_spans_marks_fabricated_evidence_missing():
    explanation = "This note offers upside participation with capped downside risk."
    dims = _all_scored(3, "upside participation")
    dims["non_misleading"] = DimensionScore(score=4, evidence="guaranteed to never lose money", reason="fabricated")
    result = JudgeResult(dims, model="judge-a", repeat=0)

    validated = validate_evidence_spans(result, explanation)

    assert validated.dimensions["non_misleading"].score is None
    for name in DIMENSIONS:
        if name != "non_misleading":
            assert validated.dimensions[name].score == 3
    assert validated.total == 3 * 5


def test_validate_evidence_spans_tolerates_whitespace_normalization():
    explanation = "Line one.\n  Line   two has   extra   spaces."
    dims = _all_scored(2, "Line\ntwo   has  extra spaces.")
    result = JudgeResult(dims, model="judge-a", repeat=0)

    validated = validate_evidence_spans(result, explanation)

    assert all(dim.score == 2 for dim in validated.dimensions.values())


# ---------------------------------------------------------------------------
# aggregate_judge_bundle: median / IQR / missing-rate, no composite score
# ---------------------------------------------------------------------------

def test_aggregate_judge_bundle_median_iqr_missing_golden():
    results = [
        JudgeResult(_all_scored(score, f"note-{score}"), model="j", repeat=i)
        for i, score in enumerate([1, 2, 3, 4])
    ]
    corrupted = dict(results[0].dimensions)
    corrupted["hedging_rationale"] = DimensionScore(score=None, evidence="x", reason="missing")
    results[0] = JudgeResult(corrupted, model="j", repeat=0)

    bundle = aggregate_judge_bundle(results)

    assert bundle["client_understanding"]["median"] == pytest.approx(2.5)
    assert bundle["client_understanding"]["iqr"] == pytest.approx(1.5)
    assert bundle["client_understanding"]["missing_rate"] == 0.0
    assert bundle["client_understanding"]["n_scored"] == 4

    assert bundle["hedging_rationale"]["n_scored"] == 3
    assert bundle["hedging_rationale"]["n_total"] == 4
    assert bundle["hedging_rationale"]["missing_rate"] == pytest.approx(0.25)
    assert bundle["hedging_rationale"]["median"] == pytest.approx(3.0)
    assert bundle["hedging_rationale"]["iqr"] == pytest.approx(1.0)

    assert "total" not in bundle
    assert set(bundle) == set(DIMENSIONS)


def test_aggregate_judge_bundle_rejects_empty_input():
    with pytest.raises(JudgeError):
        aggregate_judge_bundle([])


# ---------------------------------------------------------------------------
# judge-runs CLI pipeline: stratified sampling, self-judge exclusion,
# judges.json structure, original results untouched.
# ---------------------------------------------------------------------------

def _round(round_num: int, origin: str) -> dict:
    submitted = origin in ("voluntary", "forced_prompt")
    return {
        "round_num": round_num,
        "actions": [],
        "submitted": submitted,
        "accepted": submitted,
        "submission_origin": origin,
        "dealer_margin": 1234.5 if submitted else 0.0,
        "hard_failures": [],
        "all_quote_failures": [],
        "quote_failures_before_success": 0,
        "oracle_margin": 1500.0,
        "imputed_counterfactual_margin": None,
        "submitted_product": ({
            "product_type": "vanilla_call", "notional": 1_000_000, "maturity_months": 6,
            "strike_pct": 1.0, "barrier_pct": None, "barrier_type": None, "coupon_rate": None,
            "participation_rate": 1.0, "principal_protected": False, "target_client": "client_a",
            "pitch": "structured upside note", "hedging_plan": "delta hedge with listed options",
        } if submitted else None),
        "submitted_explanation": (
            "This note offers upside participation with capped downside risk." if submitted else ""
        ),
        "client_brief_snapshot": ({
            "episode_id": "E1", "round": round_num, "underlying": "CSI300", "spot": 4000.0,
            "condition": "full_static",
        } if submitted else None),
    }


def _result_payload(job_id: str, model: str, condition: str, episode_id: str, rounds: list[dict]) -> dict:
    return {
        "job": {
            "job_id": job_id, "episode_id": episode_id, "model": model,
            "strategy": "quote_and_revise", "condition": condition, "replicate": 0, "seed": 1,
        },
        "trace": {
            "episode_id": episode_id, "condition": condition, "strategy": "quote_and_revise",
            "rounds": rounds, "usage": {}, "seed": 1,
        },
        "metrics": {"episode_id": episode_id, "condition": condition, "strategy": "quote_and_revise"},
    }


def _judge_payload_json(score: int = 3, evidence: str = "upside participation") -> str:
    return json.dumps({
        "dimensions": {
            name: {"score": score, "evidence": evidence, "reason": "concise and accurate"}
            for name in DIMENSIONS
        }
    })


def test_stratified_sample_is_deterministic_and_balanced():
    samples = [
        {"submission_id": f"a::round{i}", "condition": "full_static", "model": "model_a"} for i in range(5)
    ] + [
        {"submission_id": f"b::round{i}", "condition": "partial_dynamic", "model": "model_b"} for i in range(5)
    ]
    picked_1 = _stratified_sample(samples, 4, seed=123)
    picked_2 = _stratified_sample(samples, 4, seed=123)
    assert [item["submission_id"] for item in picked_1] == [item["submission_id"] for item in picked_2]

    counts: dict[str, int] = {}
    for item in picked_1:
        counts[item["condition"]] = counts.get(item["condition"], 0) + 1
    assert counts == {"full_static": 2, "partial_dynamic": 2}

    picked_all = _stratified_sample(samples, 100, seed=123)
    assert len(picked_all) == len(samples)


def test_collect_voluntary_samples_only_takes_voluntary_with_product(tmp_path):
    rounds = [_round(1, "voluntary"), _round(2, "forced_prompt"), _round(3, "none")]
    payload_dict = _result_payload("job1", "model_a", "full_static", "E1", rounds)
    (tmp_path / "run1.json").write_text(json.dumps(payload_dict, ensure_ascii=False), encoding="utf-8")

    samples = _collect_voluntary_samples(tmp_path)

    assert len(samples) == 1
    assert samples[0]["submission_id"] == "job1::round1"
    assert samples[0]["model"] == "model_a"
    assert samples[0]["condition"] == "full_static"


async def test_judge_runs_end_to_end(tmp_path):
    rounds_a = [_round(1, "voluntary"), _round(2, "voluntary"), _round(3, "voluntary"), _round(4, "none")]
    rounds_b = [_round(1, "voluntary"), _round(2, "voluntary"), _round(3, "voluntary")]
    payload_a = _result_payload(
        "E1__model_a__quote_and_revise__full_static__r0", "model_a", "full_static", "E1", rounds_a,
    )
    payload_b = _result_payload(
        "E1__model_b__quote_and_revise__partial_dynamic__r0", "model_b", "partial_dynamic", "E1", rounds_b,
    )
    path_a = tmp_path / "run_model_a.json"
    path_b = tmp_path / "run_model_b.json"
    path_a.write_text(json.dumps(payload_a, ensure_ascii=False, indent=2), encoding="utf-8")
    path_b.write_text(json.dumps(payload_b, ensure_ascii=False, indent=2), encoding="utf-8")
    original_a_bytes = path_a.read_bytes()
    original_b_bytes = path_b.read_bytes()

    clients = {
        "model_a": MockLLMClient([_judge_payload_json()]),
        "judge_x": MockLLMClient([_judge_payload_json()]),
    }

    summary = await _run_judge_runs(
        results_dir=tmp_path,
        judge_models=["model_a", "judge_x"],
        clients=clients,
        repeats=2,
        sample_size=4,
        salt="test-salt",
        seed=42,
    )

    # Original result files must be byte-for-byte untouched.
    assert path_a.read_bytes() == original_a_bytes
    assert path_b.read_bytes() == original_b_bytes

    assert summary["candidates_total"] == 6
    assert summary["sample_size_selected"] == 4
    # 2 of the 4 selected samples belong to model_a; each skips exactly the
    # "model_a" judge pairing (self-judge), never the "judge_x" pairing.
    assert summary["self_judge_skips"] == 2
    assert summary["judge_calls"] == 12
    assert summary["judge_calls_missing"] == 0

    manifest = json.loads((tmp_path / "judge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["salt"] == "test-salt"
    assert manifest["sample_size_selected"] == 4
    assert len(manifest["selected"]) == 4
    assert len(manifest["self_judge_skips"]) == 2
    assert all(skip["judge_model"] == "model_a" for skip in manifest["self_judge_skips"])
    assert all(skip["model"] == "model_a" for skip in manifest["self_judge_skips"])

    judges_path_a = tmp_path / "run_model_a.judges.json"
    judges_path_b = tmp_path / "run_model_b.judges.json"
    assert judges_path_a.exists()
    assert judges_path_b.exists()

    for judges_path in (judges_path_a, judges_path_b):
        judges_payload = json.loads(judges_path.read_text(encoding="utf-8"))
        assert judges_payload["salt"] == "test-salt"
        assert judges_payload["judge_models"] == ["model_a", "judge_x"]
        assert judges_payload["repeats"] == 2
        for entry in judges_payload["entries"]:
            assert entry["judge_model"] in ("model_a", "judge_x")
            assert entry["status"] in ("ok", "missing", "skipped_self_judge")
            if entry["status"] == "skipped_self_judge":
                assert entry["judge_model"] == "model_a"
                assert entry["judge_result"] is None
            if entry["status"] == "ok":
                assert entry["judge_result"]["model"] == entry["judge_model"]
                assert set(entry["judge_result"]["dimensions"]) == set(DIMENSIONS)
                assert entry["repeat"] in (0, 1)
                assert isinstance(entry["seed"], int)


async def test_judge_runs_retries_then_marks_missing_without_aborting_batch(tmp_path):
    rounds = [_round(1, "voluntary")]
    payload_only = _result_payload("job1", "model_a", "full_static", "E1", rounds)
    (tmp_path / "run1.json").write_text(json.dumps(payload_only, ensure_ascii=False), encoding="utf-8")

    class AlwaysFailsClient(MockLLMClient):
        async def chat(self, messages, temperature=None, max_tokens=None, seed=None) -> str:
            self.total_usage["calls"] += 1
            raise RuntimeError("simulated judge API failure")

    clients = {"judge_x": AlwaysFailsClient(), "judge_y": MockLLMClient([_judge_payload_json()])}

    summary = await _run_judge_runs(
        results_dir=tmp_path,
        judge_models=["judge_x", "judge_y"],
        clients=clients,
        repeats=1,
        sample_size=5,
        salt="s",
        seed=1,
    )

    assert summary["judge_calls_missing"] == 1
    judges_payload = json.loads((tmp_path / "run1.judges.json").read_text(encoding="utf-8"))
    failing_entries = [e for e in judges_payload["entries"] if e["judge_model"] == "judge_x"]
    assert len(failing_entries) == 1
    assert failing_entries[0]["status"] == "missing"
    assert failing_entries[0]["error"] is not None
    assert failing_entries[0]["judge_result"] is None
    # 1 initial attempt + 2 retries = 3 calls to the always-failing client.
    assert clients["judge_x"].total_usage["calls"] == 3
