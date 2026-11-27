"""Report rendering from a fabricated complete evidence fixture (no model fits)."""

from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from experiments.synthetic_tau_z.design import (
    CF, EVAL_DAYS, REPORT_METRICS, TRUTH_K, W1, W3, all_effects, freeze_laws, predeclared_protocol, roster,
)
from experiments.synthetic_tau_z.report import PROPERTIES, WORLDS, render_report
from stochopia.sim.queue_reactive_world import standard_worlds


def _json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _fixture(artifact: Path) -> None:
    """Prepare deterministic effects: H1 supports, H2 refutes, H3 inconclusive."""
    jobs = roster()
    registration = predeclared_protocol()
    registration.update(
        code_commit="fixture-sha", worlds={name: asdict(spec) for name, spec in standard_worlds().items()},
        parameter_counts={job.model: 25000 for job in jobs}, state_scalars=60,
        child_times_ms=[100, 200], child_shares=100, grid_ms=[0, 50, 100],
        generation_calibration={"initial_generation_only": "fixture-only generation",
                                "final_train_anchor_calibration": "fixture train anchors",
                                "formal_model_outcomes_seen": False,
                                "truth_condition": "fixture truth condition"},
    )
    check = {"memory": True, "thin": True, "burst": True, "valid_book": True}
    properties = {}
    receipts = {}
    tier0 = {}
    for index, world in enumerate(WORLDS):
        properties[world] = {}
        for dimension, names in PROPERTIES.items():
            properties[world][dimension] = {name: {"median": index + 0.123456, "intraday_iqr": 0.5} for name in names}
        receipts[world] = {"eval_days": list(range(EVAL_DAYS)), "truth_k": TRUTH_K,
                           "truth_support_min": TRUTH_K, "truth_support_mean_fraction": 1.0,
                           "truth_peak_abs_mean_ticks": 0.2}
    for label in ("short", "long"):
        tier0[label] = {"flow": {"n_slots": 1, "alpha": 1.1, "max_len": 1},
                        "sign_acf_sum_1_32": {"median": 1.5, "intraday_iqr": 2.5},
                        "fixed_kernel_truth": [0.0, 1.0, 0.8],
                        "independent_kernel_tau_max": 16.0,
                        "independent_kernel_truth": [0.0, 1.0, 0.2]}
    manipulation = {"properties": properties, "world_receipts": receipts, "tier0": tier0,
                    "training_invalid_spread_rate": dict.fromkeys(WORLDS, 0.0), "checks": check,
                    "protocol": dict.fromkeys(check, "fixture-rule"),
                    "property_drift": {"W1->W2": {"z_mem.flow_vr_32_4": 2.0, "total_rms": 1.0}}}
    records = {}
    rows = []
    for job in jobs:
        metrics = dict.fromkeys(REPORT_METRICS, 1.0)
        if job.experiment == "E1" and job.model.endswith(("multiscale", "surprise")):
            metrics[CF] += 0.2 if job.world == W1 else -0.5
        if job.experiment == "E2" and job.model == "matrix-advect":
            metrics[CF] += 0.0 if job.world == W1 else 0.3
        records[job.key] = {"daily": [{"trading_date": date, **metrics} for date in range(EVAL_DAYS)]}
        rows.append({"job_id": job.key, "experiment": job.experiment, "world": job.world, "model": job.model,
                     "train_seed": job.seed, "parameters": 25000, "state_scalars": 60, "model_k": 4, **metrics})
    effects = all_effects(records)
    laws = freeze_laws(effects, check)
    _json(artifact / "registration.json", registration)
    _json(artifact / "manipulation_check.json", manipulation)
    _json(artifact / "execution_summary.json", {"completed": 78, "counts": dict(Counter(job.experiment for job in jobs)),
                                               "api_calls": 0, "wall_seconds": 120, "total_fit_seconds": 30,
                                               "total_eval_seconds": 50, "completed_utc": "fixture-date"})
    _json(artifact / "daily_results.json", {key: value["daily"] for key, value in records.items()})
    _json(artifact / "paired_effects.json", effects)
    _json(artifact / "frozen_synthetic_laws.json", laws)
    with (artifact / "experiment_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_report_renders_five_tables_and_negative_results(tmp_path: Path) -> None:
    _fixture(tmp_path)
    report = render_report(tmp_path)
    assert report.count("## 表 ") == 5
    for names in PROPERTIES.values():
        for name in names:
            assert name in report
    assert "0.123456 / 0.5" in report
    assert "| W1_thick_short | gru-base |" in report
    assert "| W3_thin_short | matrix-advect | 25000 | 60 |" in report
    assert "| W6_thin_hawkes | gru-hawkes |" in report
    assert "| H1 |" in report and "SUPPORTED_WITH_BOUNDARY" in report
    assert "| H2 |" in report and "REFUTED" in report
    assert "| H3 |" in report and "INCONCLUSIVE" in report
    assert "-0.7 [-0.7, -0.7]" in report
    assert "0.3 [0.3, 0.3]" in report
    assert "泊松等效（±0.02 nats）：支持" in report
    assert "12/12 日; 2000/2000 抽样" in report
    assert "78 次独立固定预算拟合" in report
    assert "模型 K=4、已知生成器 K=16" in report
    assert "校准时是否看过正式模型结果：否" in report
    assert "未建模原始订单队列的逐单账龄" in report
    assert "0/936 个拟合×评估日读数不可用" in report
    assert "`fixture-sha`" in report
    assert 'worktree add --detach "$study_tmp/locked-src" "$study_commit"' in report
    assert "-m experiments.synthetic_tau_z.run_study" in report
    assert "-m experiments.synthetic_tau_z.report" in report
    assert '--artifact-root "$study_tmp/evidence" --work-root "$study_tmp/cache"' in report


def test_report_rejects_incomplete_roster(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "experiment_results.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="78-fit roster"):
        render_report(tmp_path)


def test_report_rejects_permuted_day_indices(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "daily_results.json"
    data = json.loads(path.read_text())
    first = data[roster()[0].key]
    first[:2] = first[1::-1]
    _json(path, data)
    with pytest.raises(ValueError, match="registered day axis"):
        render_report(tmp_path)
