"""Frozen experiment-matrix generation and paired condition contrasts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import product
from typing import Mapping

from .benchmark import BenchmarkCondition, BenchmarkError
from .benchmark_runner import STRATEGIES
from .stats import bootstrap_ci, derive_seed, holm_adjust, paired_permutation_test, wilcoxon_signed_rank


CONDITIONS = (
    BenchmarkCondition(True, False),
    BenchmarkCondition(True, True),
    BenchmarkCondition(False, False),
    BenchmarkCondition(False, True),
)

# All primary cells get n>=3 replicates; partial_dynamic (full observability
# loss + market drift, the hardest cell) is over-sampled to n=5 to better
# resolve its residual variance. See REDESIGN_PLAN.md §4 / draft_codex.md §8.2.
DEFAULT_REPLICATES_BY_CONDITION: dict[str, int] = {
    "full_static": 3,
    "full_dynamic": 3,
    "partial_static": 3,
    "partial_dynamic": 5,
}

SEED_NAMESPACE = "mirage.experiment.v2"


@dataclass(frozen=True)
class ExperimentJob:
    job_id: str
    episode_id: str
    model: str
    strategy: str
    condition: str
    replicate: int
    seed: int


def build_experiment_manifest(
    episode_ids: list[str],
    models: list[str],
    *,
    strategies: tuple[str, ...] = STRATEGIES,
    replicates_by_condition: Mapping[str, int] | None = None,
) -> list[ExperimentJob]:
    """Generate the pre-registered matrix: every cell gets n>=3, partial_dynamic n=5.

    Seeds are derived deterministically from (episode, model, strategy,
    condition, replicate) via ``stats.derive_seed`` rather than a hard-coded
    seed list, so the manifest is reproducible without hand-maintained seed
    tables and every model/strategy/condition combination is independently
    seeded (no accidental seed reuse across cells).
    """
    if not episode_ids or not models:
        raise BenchmarkError("experiment manifest needs episodes and models")
    if len(episode_ids) != len(set(episode_ids)) or len(models) != len(set(models)):
        raise BenchmarkError("episode and model identifiers must be unique")
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise BenchmarkError(f"unknown strategies: {sorted(unknown)}")

    replicates = dict(DEFAULT_REPLICATES_BY_CONDITION)
    if replicates_by_condition is not None:
        unknown_conditions = set(replicates_by_condition) - {c.id for c in CONDITIONS}
        if unknown_conditions:
            raise BenchmarkError(f"unknown conditions in replicates_by_condition: {sorted(unknown_conditions)}")
        replicates.update(replicates_by_condition)
    if any(count < 3 for count in replicates.values()):
        raise BenchmarkError("every condition must have at least 3 replicates")

    jobs: list[ExperimentJob] = []
    for episode_id, model, strategy, condition in product(episode_ids, models, strategies, CONDITIONS):
        n_replicates = replicates[condition.id]
        for replicate in range(n_replicates):
            seed = derive_seed(SEED_NAMESPACE, episode_id, model, strategy, condition.id, replicate)
            job_id = f"{episode_id}__{model}__{strategy}__{condition.id}__r{replicate}"
            jobs.append(ExperimentJob(job_id, episode_id, model, strategy, condition.id, replicate, seed))
    return jobs


def manifest_payload(jobs: list[ExperimentJob]) -> dict:
    return {
        "jobs": [asdict(job) for job in jobs],
        "count": len(jobs),
        "protocol": {
            "conditions": [condition.id for condition in CONDITIONS],
            "replicates_by_condition": DEFAULT_REPLICATES_BY_CONDITION,
            "hard_condition_oversampled": "partial_dynamic",
            "seed_namespace": SEED_NAMESPACE,
            # Pre-registered primary outcomes (Holm-corrected family); every
            # other metric is exploratory. Names are the v2 canonical fields
            # (see REDESIGN_PLAN.md §4): hard_executable feasibility plus the
            # margin-family economic outcome.
            "primary_outcomes": [
                "hard_execution_rate",
                "settlement_acceptance_rate",
                "total_dealer_margin",
            ],
        },
    }


def paired_condition_contrasts(
    rows: list[dict],
    metric: str,
    *,
    seed: int = 20260802,
    n_resamples: int = 10_000,
    n_permutations: int = 100_000,
) -> dict:
    """Compute matched degradation with CI + paired significance tests.

    Episode is the primary replication/clustering unit (temperature=0 makes
    within-cell replicate seeds near-redundant; see REDESIGN_PLAN.md §4).
    Rows are first collapsed to one mean per (episode, model, strategy,
    condition) cell (averaging across replicates), then matched within each
    (episode, model, strategy).  All model/strategy/information-level contrasts
    sharing an episode are averaged again, leaving exactly one independent
    value per episode for bootstrap/Wilcoxon/permutation tests. The two named
    contrasts (dynamic degradation, partial
    observability degradation) form the pre-registered family that gets
    Holm correction; anything else computed from this function's output is
    exploratory.
    """
    keyed: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        required = {"episode_id", "model", "strategy", "condition", metric}
        if not required <= set(row):
            raise BenchmarkError(f"result row missing fields: {sorted(required - set(row))}")
        key = (row["episode_id"], row["model"], row["strategy"], row["condition"])
        value = row[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BenchmarkError(
                f"result row metric {metric!r} must be a finite numeric value; "
                f"got {value!r}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise BenchmarkError(
                f"result row metric {metric!r} must be finite; got {value!r}"
            )
        keyed.setdefault(key, []).append(numeric)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    # The episode, not each model/strategy contrast inside an episode, is the
    # replication unit.  Accumulate all within-episode matched contrasts first
    # and collapse them to one value per episode before any inference.  The old
    # implementation appended every strategy contrast directly and therefore
    # treated correlated observations from the same market path as independent.
    dynamic_by_episode: dict[str, list[float]] = {}
    partial_by_episode: dict[str, list[float]] = {}
    bases = {(e, m, s) for e, m, s, _ in keyed}
    for episode, model, strategy in sorted(bases):
        cells = {
            condition: mean(keyed[(episode, model, strategy, condition)])
            for condition in (item.id for item in CONDITIONS)
            if (episode, model, strategy, condition) in keyed
        }
        if {"full_static", "full_dynamic"} <= cells.keys():
            dynamic_by_episode.setdefault(episode, []).append(
                cells["full_static"] - cells["full_dynamic"]
            )
        if {"partial_static", "partial_dynamic"} <= cells.keys():
            dynamic_by_episode.setdefault(episode, []).append(
                cells["partial_static"] - cells["partial_dynamic"]
            )
        if {"full_static", "partial_static"} <= cells.keys():
            partial_by_episode.setdefault(episode, []).append(
                cells["full_static"] - cells["partial_static"]
            )
        if {"full_dynamic", "partial_dynamic"} <= cells.keys():
            partial_by_episode.setdefault(episode, []).append(
                cells["full_dynamic"] - cells["partial_dynamic"]
            )

    dynamic_drops = [mean(values) for _, values in sorted(dynamic_by_episode.items())]
    partial_drops = [mean(values) for _, values in sorted(partial_by_episode.items())]

    def _contrast_stats(name: str, diffs: list[float]) -> dict:
        if not diffs:
            return {
                "mean": None, "n_pairs": 0, "ci": None,
                "wilcoxon_statistic": None, "wilcoxon_p": None,
                "permutation_p": None,
            }
        ci_seed = derive_seed(SEED_NAMESPACE, "contrast_ci", metric, name, seed)
        perm_seed = derive_seed(SEED_NAMESPACE, "contrast_perm", metric, name, seed)
        _, lo, hi = bootstrap_ci(diffs, n_resamples=n_resamples, seed=ci_seed)
        stat, wilcoxon_p = wilcoxon_signed_rank(diffs)
        permutation_p = paired_permutation_test(diffs, n_permutations=n_permutations, seed=perm_seed)
        return {
            "mean": mean(diffs),
            "n_pairs": len(diffs),
            "ci": (lo, hi),
            "wilcoxon_statistic": stat,
            "wilcoxon_p": wilcoxon_p,
            "permutation_p": permutation_p,
        }

    dynamic_stats = _contrast_stats("dynamic_degradation", dynamic_drops)
    partial_stats = _contrast_stats("partial_observability_degradation", partial_drops)

    family_p = {
        name: stats["wilcoxon_p"]
        for name, stats in (
            ("dynamic_degradation", dynamic_stats),
            ("partial_observability_degradation", partial_stats),
        )
        if stats["wilcoxon_p"] is not None
    }
    holm_p = holm_adjust(family_p)

    return {
        "metric": metric,
        # Backward-compatible flat keys.
        "dynamic_degradation_mean": dynamic_stats["mean"],
        "partial_observability_degradation_mean": partial_stats["mean"],
        "dynamic_pairs": dynamic_stats["n_pairs"],
        "partial_pairs": partial_stats["n_pairs"],
        # New CI / significance detail.
        "dynamic_degradation": dynamic_stats,
        "partial_observability_degradation": partial_stats,
        "holm_adjusted_wilcoxon_p": holm_p,
    }
