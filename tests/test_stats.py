"""Golden-value tests for stochopia/stats.py: seeding, bootstrap CIs, paired
significance tests, and Holm correction.

The Wilcoxon exact-method tests cross-check stochopia.stats against an
independently written brute-force reference (enumerating every sign-flip
pattern directly) rather than a memorized textbook number, so the "golden"
value is derived from first principles inside the test itself and does not
share code with the implementation under test.
"""

from __future__ import annotations

import itertools
import random

import pytest

from stochopia.experiment import SEED_NAMESPACE as EXPERIMENT_SEED_NAMESPACE
from stochopia.stats import (
    StatsError,
    bootstrap_ci,
    cluster_bootstrap_ci,
    derive_seed,
    holm_adjust,
    paired_permutation_test,
    wilcoxon_signed_rank,
)


# --------------------------------------------------------------------------
# derive_seed
# --------------------------------------------------------------------------

def test_derive_seed_deterministic_for_same_input():
    parts = ("E01", "gpt-5", "ledger_archive", "full_static", 0)
    a = derive_seed(EXPERIMENT_SEED_NAMESPACE, *parts)
    b = derive_seed(EXPERIMENT_SEED_NAMESPACE, *parts)
    assert a == b == 4_233_541_484


def test_derive_seed_varies_with_namespace():
    a = derive_seed("namespace-a", "E01", "gpt-5")
    b = derive_seed("namespace-b", "E01", "gpt-5")
    assert a != b


def test_derive_seed_varies_with_parts_and_is_uint32():
    a = derive_seed("ns", "E01", "modelA")
    b = derive_seed("ns", "E01", "modelB")
    c = derive_seed("ns", "E02", "modelA")
    assert len({a, b, c}) == 3
    for value in (a, b, c):
        assert 0 <= value < 2 ** 32


def test_derive_seed_distinguishes_concatenation_boundary():
    # "ab","c" and "a","bc" must not collide even though naive '+'-join would.
    a = derive_seed("ns", "ab", "c")
    b = derive_seed("ns", "a", "bc")
    assert a != b


# --------------------------------------------------------------------------
# bootstrap_ci
# --------------------------------------------------------------------------

def test_bootstrap_ci_reproducible_with_fixed_seed():
    values = [float(v) for v in range(1, 11)]
    result_a = bootstrap_ci(values, n_resamples=2000, seed=42)
    result_b = bootstrap_ci(values, n_resamples=2000, seed=42)
    assert result_a == result_b


def test_bootstrap_ci_covers_mean_and_is_ordered():
    values = [10.0, 12.0, 9.0, 11.0, 10.5, 9.5, 13.0, 8.0]
    mean, lo, hi = bootstrap_ci(values, n_resamples=5000, seed=7)
    assert mean == pytest.approx(sum(values) / len(values))
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_ci_single_value_degenerate():
    mean, lo, hi = bootstrap_ci([5.0], seed=1)
    assert mean == lo == hi == 5.0


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(StatsError):
        bootstrap_ci([], seed=1)


# --------------------------------------------------------------------------
# cluster_bootstrap_ci
# --------------------------------------------------------------------------

def test_cluster_bootstrap_ci_reproducible_and_groups_by_cluster():
    rows = [
        {"episode_id": "E1", "hard_feasibility_rate": 0.8},
        {"episode_id": "E1", "hard_feasibility_rate": 0.6},
        {"episode_id": "E2", "hard_feasibility_rate": 0.4},
        {"episode_id": "E2", "hard_feasibility_rate": 0.5},
        {"episode_id": "E3", "hard_feasibility_rate": 0.9},
    ]
    a = cluster_bootstrap_ci(rows, "hard_feasibility_rate", seed=99, n_resamples=2000)
    b = cluster_bootstrap_ci(rows, "hard_feasibility_rate", seed=99, n_resamples=2000)
    assert a == b
    mean, lo, hi = a
    all_values = [row["hard_feasibility_rate"] for row in rows]
    assert mean == pytest.approx(sum(all_values) / len(all_values))
    assert lo <= mean <= hi


def test_cluster_bootstrap_ci_skips_missing_fields():
    rows = [
        {"episode_id": "E1", "total_dealer_margin": 100.0},
        {"episode_id": "E1"},  # missing metric -> skipped
        {"total_dealer_margin": 50.0},  # missing cluster key -> skipped
        {"episode_id": "E2", "total_dealer_margin": 200.0},
    ]
    mean, lo, hi = cluster_bootstrap_ci(rows, "total_dealer_margin", seed=1, n_resamples=500)
    assert mean == pytest.approx(150.0)


def test_cluster_bootstrap_ci_rejects_all_missing():
    with pytest.raises(StatsError):
        cluster_bootstrap_ci([{"episode_id": "E1"}], "total_dealer_margin", seed=1)


# --------------------------------------------------------------------------
# wilcoxon_signed_rank: independent brute-force reference
# --------------------------------------------------------------------------

def _brute_force_wilcoxon(diffs):
    """Independent reference: enumerate all 2**n sign patterns directly."""
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0
    abs_d = [abs(d) for d in nonzero]
    order = sorted(range(n), key=lambda i: abs_d[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs_d[order[j]] == abs_d[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in order[i:j]:
            ranks[k] = avg_rank
        i = j
    observed_w_pos = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    observed_w_neg = sum(ranks) - observed_w_pos
    statistic = min(observed_w_pos, observed_w_neg)
    total = 0
    le = 0
    ge = 0
    for signs in itertools.product((1, -1), repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s > 0)
        total += 1
        if w <= observed_w_pos + 1e-9:
            le += 1
        if w >= observed_w_pos - 1e-9:
            ge += 1
    p_value = min(1.0, 2.0 * min(le, ge) / total)
    return statistic, p_value


@pytest.mark.parametrize(
    "diffs",
    [
        [1.2, -0.3, 2.5, 0.8, -1.1, 1.9],  # n=6, no ties
        [3.0, -1.0, 2.0, -2.0, 1.5, -0.5, 4.0, -3.0],  # n=8, no ties
        [1.0, 1.0, -2.0, 3.0, -1.0, 2.0],  # n=6, with ties in |diff|
        [0.0, 2.0, -1.0, 3.0, -2.0, 1.0, 0.0],  # includes zero diffs
        [5.0, 4.0, 3.0, 2.0, 1.0],  # n=5, strictly one-sided (small exact p)
    ],
)
def test_wilcoxon_exact_matches_brute_force_reference(diffs):
    expected_stat, expected_p = _brute_force_wilcoxon(diffs)
    stat, p = wilcoxon_signed_rank(diffs)
    assert stat == pytest.approx(expected_stat)
    assert p == pytest.approx(expected_p, abs=1e-9)


def test_wilcoxon_all_zero_diffs_returns_no_evidence():
    stat, p = wilcoxon_signed_rank([0.0, 0.0, 0.0])
    assert stat == 0.0
    assert p == 1.0


def test_wilcoxon_drops_zero_diffs_before_testing():
    with_zero = [0.0, 1.2, -0.3, 2.5, 0.8, -1.1, 1.9]
    without_zero = [1.2, -0.3, 2.5, 0.8, -1.1, 1.9]
    assert wilcoxon_signed_rank(with_zero) == wilcoxon_signed_rank(without_zero)


def test_wilcoxon_large_sample_normal_approx_close_to_exact():
    rng = random.Random(2026)
    diffs = [rng.gauss(0.25, 1.0) for _ in range(30)]
    stat_exact, p_exact = wilcoxon_signed_rank(diffs, method="exact")
    stat_normal, p_normal = wilcoxon_signed_rank(diffs, method="normal")
    assert stat_exact == stat_normal
    assert abs(p_exact - p_normal) < 0.1
    # auto-selection at n=30 (>25) must take the normal branch.
    stat_auto, p_auto = wilcoxon_signed_rank(diffs)
    assert (stat_auto, p_auto) == (stat_normal, p_normal)


def test_wilcoxon_auto_uses_exact_at_boundary_n25():
    rng = random.Random(11)
    diffs = [rng.gauss(0.0, 1.0) for _ in range(25)]
    assert wilcoxon_signed_rank(diffs) == wilcoxon_signed_rank(diffs, method="exact")


# --------------------------------------------------------------------------
# paired_permutation_test
# --------------------------------------------------------------------------

def test_paired_permutation_test_deterministic_with_seed():
    diffs = [1.0, 2.0, -0.5, 1.5, 0.8, -0.2]
    a = paired_permutation_test(diffs, n_permutations=5000, seed=123)
    b = paired_permutation_test(diffs, n_permutations=5000, seed=123)
    assert a == b


def test_paired_permutation_test_different_seed_may_differ_but_close():
    diffs = [1.0, 2.0, -0.5, 1.5, 0.8, -0.2]
    a = paired_permutation_test(diffs, n_permutations=20000, seed=1)
    b = paired_permutation_test(diffs, n_permutations=20000, seed=2)
    assert abs(a - b) < 0.05


def test_paired_permutation_test_one_sided_signal_gives_small_p():
    diffs = [2.0, 2.1, 1.9, 2.2, 2.05, 1.95, 2.3, 1.85]
    p = paired_permutation_test(diffs, n_permutations=20000, seed=5)
    assert p < 0.05


def test_paired_permutation_test_symmetric_noise_gives_large_p():
    diffs = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    p = paired_permutation_test(diffs, n_permutations=20000, seed=5)
    assert p > 0.5


def test_paired_permutation_test_rejects_empty():
    with pytest.raises(StatsError):
        paired_permutation_test([], seed=1)


# --------------------------------------------------------------------------
# holm_adjust
# --------------------------------------------------------------------------

def test_holm_adjust_known_example():
    # sorted ascending: d=.005, a=.01, c=.03, b=.04 ; m=4
    # d: 4*.005=.02 ; a: 3*.01=.03 ; c: 2*.03=.06 ; b: 1*.04=.04 -> cummax .06
    p_values = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005}
    adjusted = holm_adjust(p_values)
    assert adjusted["d"] == pytest.approx(0.02)
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)


def test_holm_adjust_monotone_nondecreasing_in_sorted_order():
    p_values = {"x": 0.2, "y": 0.01, "z": 0.15, "w": 0.001}
    adjusted = holm_adjust(p_values)
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    values_in_order = [adjusted[name] for name, _ in ordered]
    assert values_in_order == sorted(values_in_order)


def test_holm_adjust_caps_at_one():
    adjusted = holm_adjust({"a": 0.9, "b": 0.8, "c": 0.7})
    assert all(v <= 1.0 for v in adjusted.values())
    assert adjusted["a"] == 1.0


def test_holm_adjust_empty():
    assert holm_adjust({}) == {}


def test_holm_adjust_single_value_unchanged():
    adjusted = holm_adjust({"only": 0.03})
    assert adjusted["only"] == pytest.approx(0.03)
