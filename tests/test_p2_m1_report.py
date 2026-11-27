"""The M1 report preserves optional conditional scores and day clusters."""

from copy import deepcopy

import pytest

from experiments.p2_m1_report import CONTINUOUS, HEADS, LABELS, render


def _summary(estimate: float | None = 0.0, interval: list[float] | None = None) -> dict:
    return {"estimate": estimate, "mean": estimate, "interval": interval,
            "candidate_bootstrap_mean": 0.25, "bootstrap_valid": 0}


def _fixture() -> dict:
    measurements = {}
    pairs = {}
    for k in (1, 4, 8):
        for label in LABELS if k == 1 else ("M0", "M3"):
            key = f"{label}-k{k}"
            measurements[key] = {
                "n_days": 20,
                "timing": {"wall_seconds": 10.0},
                "evidence": {
                    "impact_identifiability_state": "NO_RESPONSE",
                    "impact_curve_mae": 0.1,
                    "post_stop_relaxation_delta": 0.0,
                    "visible_minus_blind_auc": 0.2,
                    "lob_impact_score": None,
                    "paired_seed_windows": 120,
                    "val_nll_total": 1.0,
                    "stateless_nll_total": 1.25,
                    "stateless_nll_gap": 0.25,
                    "stateless_nll_gap_by_stage": dict.fromkeys(HEADS, 0.0625),
                },
            }
            pair = {metric: _summary(0.0, [-0.01, 0.02]) for metric in CONTINUOUS}
            pair["delta_c2_pass_prob"] = _summary(0.0, [0.0, 0.0])
            pair["delta_lob_impact_score"] = _summary(None)
            pair["lob_impact_score"] = _summary(None)
            pair["joint_c2_pass_share"] = 0.0
            pairs[key] = pair
    return {
        "registration": {"schema": "p2-m1", "test_dates": list(range(20)), "rollout_seed": 9127,
                         "code": {"stochopia": "stub-a", "openhyra": "stub-b"}},
        "measurements": measurements,
        "pairs": pairs,
    }


def test_three_report_tables_and_undefined_conditional_score() -> None:
    doc = render(_fixture())
    assert all(f"## 表 {i}" in doc for i in range(1, 4))
    assert "有限配对区间 12/12" in doc
    assert "不可判定" in doc
    assert "20/20" in doc
    assert "不是新的留出测试" in doc


def test_rejects_replica_pseudoclusters() -> None:
    report = deepcopy(_fixture())
    report["measurements"]["M0-k4"]["n_days"] = 80
    with pytest.raises(ValueError, match="independent days"):
        render(report)
