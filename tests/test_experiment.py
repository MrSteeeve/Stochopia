"""Frozen factorial manifest and matched contrasts."""

import pytest

from mirage.experiment import (
    DEFAULT_REPLICATES_BY_CONDITION,
    build_experiment_manifest,
    paired_condition_contrasts,
)


def test_manifest_every_condition_has_at_least_three_replicates():
    jobs = build_experiment_manifest(["E1"], ["M1"], strategies=("one_shot",))
    # 3 + 3 + 3 + 5 = 14 for the default replicate schedule.
    assert len(jobs) == 14
    by_condition: dict[str, int] = {}
    for job in jobs:
        by_condition[job.condition] = by_condition.get(job.condition, 0) + 1
    assert by_condition == {
        "full_static": 3, "full_dynamic": 3, "partial_static": 3, "partial_dynamic": 5,
    }
    assert all(count >= 3 for count in by_condition.values())
    assert by_condition["partial_dynamic"] == 5


def test_manifest_replicate_seeds_are_deterministic_and_distinct():
    jobs_a = build_experiment_manifest(["E1"], ["M1"], strategies=("one_shot",))
    jobs_b = build_experiment_manifest(["E1"], ["M1"], strategies=("one_shot",))
    assert [job.seed for job in jobs_a] == [job.seed for job in jobs_b]
    # Every job in the manifest gets its own seed (no accidental collisions).
    assert len({job.seed for job in jobs_a}) == len(jobs_a)


def test_manifest_job_ids_unique_and_include_replicate_index():
    jobs = build_experiment_manifest(["E1", "E2"], ["M1"], strategies=("one_shot",))
    job_ids = [job.job_id for job in jobs]
    assert len(job_ids) == len(set(job_ids))
    assert any(job_id.endswith("__r4") for job_id in job_ids)  # partial_dynamic 5th replicate


def test_manifest_custom_replicates_by_condition():
    jobs = build_experiment_manifest(
        ["E1"], ["M1"], strategies=("one_shot",),
        replicates_by_condition={"full_static": 4, "partial_dynamic": 8},
    )
    by_condition: dict[str, int] = {}
    for job in jobs:
        by_condition[job.condition] = by_condition.get(job.condition, 0) + 1
    assert by_condition["full_static"] == 4
    assert by_condition["partial_dynamic"] == 8
    # Untouched conditions keep the default.
    assert by_condition["full_dynamic"] == DEFAULT_REPLICATES_BY_CONDITION["full_dynamic"]


def test_manifest_rejects_replicates_below_three():
    with pytest.raises(Exception):
        build_experiment_manifest(
            ["E1"], ["M1"], strategies=("one_shot",),
            replicates_by_condition={"full_static": 1},
        )


def test_paired_condition_contrasts_means_backward_compatible():
    rows = [
        {"episode_id": "E", "model": "M", "strategy": "S", "condition": "full_static", "score": 0.9},
        {"episode_id": "E", "model": "M", "strategy": "S", "condition": "full_dynamic", "score": 0.7},
        {"episode_id": "E", "model": "M", "strategy": "S", "condition": "partial_static", "score": 0.8},
        {"episode_id": "E", "model": "M", "strategy": "S", "condition": "partial_dynamic", "score": 0.5},
    ]
    result = paired_condition_contrasts(rows, "score")
    assert result["dynamic_degradation_mean"] == pytest.approx(0.25)
    assert result["partial_observability_degradation_mean"] == pytest.approx(0.15)
    assert result["dynamic_pairs"] == 2
    assert result["partial_pairs"] == 2


def test_paired_condition_contrasts_adds_ci_and_significance():
    # Three episodes so Wilcoxon/bootstrap/permutation have non-trivial n.
    rows = []
    for episode, (fs, fd, ps, pd) in {
        "E1": (0.9, 0.7, 0.8, 0.5),
        "E2": (0.85, 0.6, 0.75, 0.4),
        "E3": (0.95, 0.8, 0.85, 0.6),
    }.items():
        rows += [
            {"episode_id": episode, "model": "M", "strategy": "S", "condition": "full_static", "score": fs},
            {"episode_id": episode, "model": "M", "strategy": "S", "condition": "full_dynamic", "score": fd},
            {"episode_id": episode, "model": "M", "strategy": "S", "condition": "partial_static", "score": ps},
            {"episode_id": episode, "model": "M", "strategy": "S", "condition": "partial_dynamic", "score": pd},
        ]
    result = paired_condition_contrasts(rows, "score", n_permutations=2000, n_resamples=2000)
    dynamic = result["dynamic_degradation"]
    partial = result["partial_observability_degradation"]
    assert dynamic["n_pairs"] == 6  # 2 drops per episode x 3 episodes
    assert partial["n_pairs"] == 6
    assert dynamic["mean"] > 0  # full/partial always degrade under dynamic in this fixture
    lo, hi = dynamic["ci"]
    assert lo <= dynamic["mean"] <= hi
    assert 0.0 <= dynamic["wilcoxon_p"] <= 1.0
    assert 0.0 <= dynamic["permutation_p"] <= 1.0
    assert set(result["holm_adjusted_wilcoxon_p"]) == {
        "dynamic_degradation", "partial_observability_degradation",
    }
    for adjusted_p in result["holm_adjusted_wilcoxon_p"].values():
        assert 0.0 <= adjusted_p <= 1.0


def test_paired_condition_contrasts_reproducible_with_same_seed():
    rows = [
        {"episode_id": "E1", "model": "M", "strategy": "S", "condition": "full_static", "score": 0.9},
        {"episode_id": "E1", "model": "M", "strategy": "S", "condition": "full_dynamic", "score": 0.5},
        {"episode_id": "E2", "model": "M", "strategy": "S", "condition": "full_static", "score": 0.8},
        {"episode_id": "E2", "model": "M", "strategy": "S", "condition": "full_dynamic", "score": 0.3},
    ]
    a = paired_condition_contrasts(rows, "score", n_permutations=2000, n_resamples=2000, seed=7)
    b = paired_condition_contrasts(rows, "score", n_permutations=2000, n_resamples=2000, seed=7)
    assert a == b


def test_paired_condition_contrasts_missing_field_raises():
    with pytest.raises(Exception):
        paired_condition_contrasts([{"episode_id": "E"}], "score")
