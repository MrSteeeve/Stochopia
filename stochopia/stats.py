"""Dependency-free statistics for Stochopia v2: seeding, bootstrap CIs, paired
significance tests, and multiple-comparison correction.

Every routine here is deterministic given an explicit seed and uses only the
standard library (see judge.py for the precedent of self-written Spearman /
weighted-kappa rather than adding scipy/numpy). Nothing in this module calls
Python's built-in ``hash()``, which is randomized per-process by default and
therefore unsuitable for reproducible experiment seeding.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Literal, Mapping, Sequence


class StatsError(Exception):
    """Inputs to a statistics routine are empty, misaligned, or malformed."""


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------

def derive_seed(namespace: str, *parts: str | int | float) -> int:
    """Deterministically derive a uint32 seed from a namespace and parts.

    Uses SHA256 over a canonical, unambiguous encoding of ``(namespace,
    *parts)`` (each part stringified and separated by the ASCII unit
    separator 0x1F, which is vanishingly unlikely to appear in identifiers)
    and takes the first 4 bytes as a big-endian unsigned integer. This is
    stable across processes and Python versions, unlike the builtin
    ``hash()``, which is salted per-process for strings.
    """
    canonical = "\x1f".join((str(namespace), *(str(part) for part in parts)))
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# --------------------------------------------------------------------------
# Bootstrap confidence intervals
# --------------------------------------------------------------------------

def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence, q in [0,1]."""
    n = len(sorted_values)
    if n == 0:
        raise StatsError("percentile of empty sequence")
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[int(pos)]
    frac = pos - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(mean, lo, hi)`` where ``mean`` is the observed sample mean and
    ``(lo, hi)`` is the ``1 - alpha`` percentile bootstrap interval computed
    from ``n_resamples`` resamples-with-replacement, using a seeded RNG so
    the result is exactly reproducible.
    """
    values = [float(v) for v in values]
    if not values:
        raise StatsError("bootstrap_ci requires at least one value")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = _percentile(means, alpha / 2)
    hi = _percentile(means, 1 - alpha / 2)
    return mean, lo, hi


def cluster_bootstrap_ci(
    rows: Sequence[Mapping],
    metric: str,
    *,
    cluster_key: str = "episode_id",
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int,
) -> tuple[float, float, float]:
    """Cluster (two-stage) percentile bootstrap CI for the mean of ``metric``.

    Rows are grouped by ``cluster_key`` (episode by default, per the
    redesign's "episode is the primary replication unit" decision). Each
    resample draws clusters with replacement (same count as the number of
    observed clusters) and pools every row belonging to the drawn clusters,
    which respects within-cluster correlation instead of treating every row
    as an independent Bernoulli/observation.

    Rows missing ``cluster_key`` or ``metric`` (or with a non-numeric /
    ``None`` metric value) are silently skipped so partially-populated
    result directories degrade gracefully instead of raising.
    """
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if cluster_key not in row or metric not in row:
            continue
        value = row[metric]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        grouped.setdefault(str(row[cluster_key]), []).append(float(value))
    if not grouped:
        raise StatsError(f"cluster_bootstrap_ci found no rows with {cluster_key!r} and {metric!r}")
    clusters = list(grouped.values())
    all_values = [v for values in clusters for v in values]
    point_mean = sum(all_values) / len(all_values)
    n_clusters = len(clusters)
    if n_clusters == 1:
        return point_mean, point_mean, point_mean
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        pooled: list[float] = []
        for _ in range(n_clusters):
            pooled.extend(clusters[rng.randrange(n_clusters)])
        means.append(sum(pooled) / len(pooled))
    means.sort()
    lo = _percentile(means, alpha / 2)
    hi = _percentile(means, 1 - alpha / 2)
    return point_mean, lo, hi


# --------------------------------------------------------------------------
# Wilcoxon signed-rank test (self-written, zero-dependency)
# --------------------------------------------------------------------------

def _signed_ranks(abs_diffs: Sequence[float]) -> list[float]:
    """Average ranks (1-based) of |diffs|, doubled so ties (.5 ranks) are integers."""
    order = sorted(range(len(abs_diffs)), key=lambda i: abs_diffs[i])
    ranks = [0.0] * len(abs_diffs)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and abs_diffs[order[end]] == abs_diffs[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _exact_signed_rank_distribution(doubled_ranks: Sequence[int]) -> list[int]:
    """dp[s] = number of the 2**n sign assignments whose positive-rank sum is s/2.

    ``doubled_ranks`` are the true ranks multiplied by 2 (always integers,
    since average ranks only ever have a .0 or .5 fractional part). This is
    a standard 0/1 subset-sum counting DP: each rank independently lands in
    the "positive" or "negative" bucket, and we count how many of the
    2**n equally likely sign patterns produce each achievable positive-sum.
    """
    max_sum = sum(doubled_ranks)
    dp = [0] * (max_sum + 1)
    dp[0] = 1
    for rank in doubled_ranks:
        for s in range(max_sum, rank - 1, -1):
            if dp[s - rank]:
                dp[s] += dp[s - rank]
    return dp


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank(
    diffs: Sequence[float],
    *,
    method: Literal["auto", "exact", "normal"] = "auto",
    exact_max_n: int = 25,
) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Zero differences are dropped first (standard convention: they carry no
    directional information). Returns ``(statistic, p_value)`` where
    ``statistic`` is the classic Wilcoxon T = min(W+, W-), the smaller of the
    rank sums for positive vs. negative differences.

    ``method="auto"`` (the default) follows the redesign spec: for n <= 25
    non-zero differences the exact null distribution of the signed-rank sum
    is enumerated by dynamic programming (conditional on the observed,
    possibly tie-averaged, ranks); for n > 25 a normal approximation with
    continuity correction and a tie-variance correction is used. ``method``
    can be forced to "exact" or "normal" (e.g. to check they agree near the
    boundary).

    If every difference is zero (no evidence either way), returns
    ``(0.0, 1.0)``.
    """
    nonzero = [float(d) for d in diffs if float(d) != 0.0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0

    abs_diffs = [abs(d) for d in nonzero]
    ranks = _signed_ranks(abs_diffs)
    w_pos = sum(rank for rank, d in zip(ranks, nonzero) if d > 0)
    w_neg = sum(rank for rank, d in zip(ranks, nonzero) if d < 0)
    statistic = min(w_pos, w_neg)

    use_exact = method == "exact" or (method == "auto" and n <= exact_max_n)
    if method == "normal":
        use_exact = False

    if use_exact:
        doubled = [int(round(rank * 2)) for rank in ranks]
        dp = _exact_signed_rank_distribution(doubled)
        total = 1 << n
        doubled_w_pos = int(round(w_pos * 2))
        # P(W+ <= observed) and P(W+ >= observed) under the null (each sign
        # pattern equally likely); two-sided p is twice the smaller tail,
        # capped at 1.
        le = sum(dp[s] for s in range(0, doubled_w_pos + 1)) / total
        ge = sum(dp[s] for s in range(doubled_w_pos, len(dp))) / total
        p_value = min(1.0, 2.0 * min(le, ge))
        return statistic, p_value

    # Normal approximation with continuity correction and tie correction.
    mean = n * (n + 1) / 4.0
    # Tie groups among |diffs|.
    tie_sizes: dict[float, int] = {}
    for value in abs_diffs:
        tie_sizes[value] = tie_sizes.get(value, 0) + 1
    tie_correction = sum(t ** 3 - t for t in tie_sizes.values())
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0
    if variance <= 0:
        return statistic, 1.0
    sd = math.sqrt(variance)
    if w_pos > mean:
        z = (w_pos - mean - 0.5) / sd
    else:
        z = (w_pos - mean + 0.5) / sd
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return statistic, min(1.0, max(0.0, p_value))


# --------------------------------------------------------------------------
# Paired sign-flip permutation test
# --------------------------------------------------------------------------

def paired_permutation_test(
    diffs: Sequence[float],
    *,
    n_permutations: int = 100_000,
    seed: int,
) -> float:
    """Two-sided paired permutation (sign-flip) test on paired differences.

    Under the null hypothesis of no systematic difference, each paired
    difference is equally likely to have its sign flipped. We compare the
    observed mean difference against the distribution of mean differences
    under ``n_permutations`` random sign flips, using add-one (Laplace)
    smoothing so p is never exactly zero.
    """
    values = [float(d) for d in diffs]
    if not values:
        raise StatsError("paired_permutation_test requires at least one difference")
    n = len(values)
    observed = abs(sum(values) / n)
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        total = 0.0
        for value in values:
            total += value if rng.random() < 0.5 else -value
        if abs(total / n) >= observed - 1e-12:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (n_permutations + 1)


# --------------------------------------------------------------------------
# Holm-Bonferroni step-down correction
# --------------------------------------------------------------------------

def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, keyed the same as the input.

    Standard Holm-Bonferroni: sort ascending, adjust the k-th smallest
    (1-indexed) by (m - k + 1), enforce monotonicity via a running max, and
    cap at 1.0. Controls family-wise error rate without assuming
    independence, uniformly more powerful than plain Bonferroni.
    """
    if not p_values:
        return {}
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    running_max = 0.0
    adjusted: dict[str, float] = {}
    for index, (name, p) in enumerate(items):
        candidate = (m - index) * p
        running_max = max(running_max, candidate)
        adjusted[name] = min(1.0, running_max)
    return adjusted
