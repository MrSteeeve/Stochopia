"""Formal roster and day-cluster hypothesis adjudication, no training."""

from collections import Counter

import numpy as np

from experiments.p2_m2_design import (
    CF, EVAL_DAYS, W1, W2, W3, W6, all_effects,
    freeze_laws, interaction, predeclared_protocol, roster, summarize_days,
)


def test_registered_factorial_budget_is_exactly_78() -> None:
    jobs = roster()
    assert len({job.key for job in jobs}) == 78
    assert Counter(job.experiment for job in jobs) == {"E1": 36, "E2": 24, "E3": 18}
    config = predeclared_protocol()
    assert config["model_k"] == 4
    assert config["truth_k"] == 16
    assert config["fit_protocol"]["steps"] == 96


def test_bootstrap_does_not_add_replica_or_seed_clusters() -> None:
    result = summarize_days([float(index) for index in range(EVAL_DAYS)])
    assert result["registered_days"] == result["paired_days"] == 12
    assert result["bootstrap_valid"] == 2000
    assert result["estimate"] == 5.5
    assert result["interval"][0] < 5.5 < result["interval"][1]
    missing = summarize_days([1.0] * 5 + [None] * 7)
    assert missing["interval"] is None


def _records() -> dict:
    rows = {}
    for job in roster():
        impact = 1.0
        dt = 2.0
        if job.experiment == "E1" and job.model.endswith(("multiscale", "surprise")):
            impact += 0.2 if job.world == W1 else -0.5
        if job.experiment == "E2" and job.model == "matrix-advect":
            impact += 0.0 if job.world == W1 else 0.3
        if job.experiment == "E3" and job.model == "gru-hawkes":
            dt += 0.0 if job.world == W3 else -0.1
        rows[job.key] = {"daily": [
            {"trading_date": index, CF: impact, "dt_nll": dt,
             "direction_acf_mae": 0.1, "memory_gap_64": 0.1,
             "rel_ticks_nll": 0.2, "zero_wait_rate_error": 0.1, "interarrival_cv2_error": 0.1}
            for index in range(EVAL_DAYS)
        ]}
    return rows


def test_signed_interaction_and_all_law_statuses() -> None:
    effects = all_effects(_records())
    checks = {"memory": True, "thin": True, "burst": True}
    laws = freeze_laws(effects, checks)
    assert laws["H1"]["status"] == "SUPPORTED_WITH_BOUNDARY"
    assert np.isclose(laws["H1"]["primary"]["estimate"], -0.7)
    assert laws["H2"]["status"] == "REFUTED"
    assert laws["H3"]["status"] == "SUPPORTED_WITH_BOUNDARY"
    assert laws["H3"]["poisson_equivalence_supported"]
    checks["burst"] = False
    assert freeze_laws(effects, checks)["H3"]["status"] == "INCONCLUSIVE"
    direct = interaction(_records(), "E3", W3, W6, "gru-hawkes", "gru-base", "dt_nll")
    assert direct["paired_days"] == 12
    assert np.isclose(direct["estimate"], -0.1)
