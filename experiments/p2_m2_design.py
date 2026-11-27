"""Predeclared 78-cell synthetic study and day-clustered interaction estimands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from collections.abc import Sequence
from typing import Any

import numpy as np

from stochopia.neural.lob_training import FitProtocol

Json = dict[str, Any]
SEEDS = (103, 137, 211)
DATA_SEED = 20261127
TRUTH_SEED = 76193
ROLLOUT_SEED = 9127
BOOTSTRAP_SEED = 41397
N_BOOTSTRAP = 2000
K = 4
TRUTH_K = 16
BATCH_WINDOWS = 2
TRAIN_DAYS = 16
EVAL_DAYS = 12
EVENTS_PER_SESSION = 256
CONTEXT = 128
FIT_PROTOCOL = FitProtocol(subseq_len=128, burn_in=32, batch_size=8, steps=96,
                           max_train_seconds=120.0)
W1 = "W1_thick_short"
W2 = "W2_thick_long"
W3 = "W3_thin_short"
W4 = "W4_thin_long"
W6 = "W6_thin_hawkes"
MEM_MODELS = ("gru-base", "gru-multiscale", "gru-surprise",
              "lru-base", "lru-multiscale", "lru-surprise")
THIN_MODELS = ("flat-gru", "matrix-noshift", "matrix-advect", "flat-readoutgate")
BURST_MODELS = ("gru-base", "selective-ssm", "gru-hawkes")
CF = "counterfactual_impact_mae"
REPORT_METRICS = ("val_nll_total", "dt_nll", "rel_ticks_nll", "action_nll", "volume_nll",
                  "memory_gap_1", "memory_gap_4", "memory_gap_16", "memory_gap_64",
                  "direction_acf_mae", "zero_wait_rate_error", "interarrival_cv2_error",
                  "exhausted_replicate_rate", CF)


@dataclass(frozen=True)
class Job:
    experiment: str
    world: str
    model: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.experiment}-{self.world}-{self.model}-{self.seed}"


def roster() -> tuple[Job, ...]:
    """Exactly 36+24+18 independent, fixed-budget fits (shared baselines repeated)."""
    grids = (("E1", (W1, W2), MEM_MODELS),
             ("E2", (W1, W3), THIN_MODELS),
             ("E3", (W3, W6), BURST_MODELS))
    jobs = []
    for experiment, worlds, models in grids:
        for world in worlds:
            for model in models:
                for seed in SEEDS:
                    jobs.append(Job(experiment, world, model, seed))
    assert len(jobs) == 78
    return tuple(jobs)


def predeclared_protocol() -> Json:
    return {
        "schema": "p2-m2-synthetic-v1", "fit_protocol": asdict(FIT_PROTOCOL),
        "train_days": TRAIN_DAYS, "eval_days": EVAL_DAYS, "sessions_per_day": 2,
        "events_per_session": EVENTS_PER_SESSION, "context_len": CONTEXT,
        "data_seed": DATA_SEED, "truth_seed": TRUTH_SEED, "rollout_seed": ROLLOUT_SEED,
        "model_k": K, "truth_k": TRUTH_K, "batch_windows": BATCH_WINDOWS,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": N_BOOTSTRAP,
        "jobs": [dict(asdict(job), job_id=job.key) for job in roster()],
        "hypotheses": {
            "H1": "mean of four (GRU/LRU x multiscale/surprise) impact MAE interactions long-short",
            "H2": "advect minus no-shift impact MAE interaction thin-thick",
            "H3": "GRU-Hawkes minus GRU-base dt NLL interaction hawkes-poisson",
        },
        "decision": "95% paired day-bootstrap interval wholly below 0 supports; wholly above 0 refutes; otherwise inconclusive; require target manipulation and common support",
        "poisson_dt_equivalence_band_nats": [-0.02, 0.02],
        "api_calls": 0,
    }


def _finite(value: object) -> float | None:
    return float(value) if isinstance(value, (float, int)) and isfinite(value) else None


def summarize_days(values: Sequence[float | None]) -> Json:
    """12 independent day indices, 2000 common draws; no seed/replica inflation.

    The mean and percentile interval use available paired day estimates. At
    least half of the 12 registered days and 1800 finite draws are required;
    no missing measurement is imputed. All contrasts share draw indices.
    """
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    result: Json = {"estimate": None, "interval": None, "paired_days": int(valid.sum()),
                    "registered_days": len(x), "bootstrap_valid": 0, "bootstrap_total": N_BOOTSTRAP}
    if valid.any():
        result["estimate"] = float(x[valid].mean())
    if len(x) == 0:
        return result
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(x), size=(N_BOOTSTRAP, len(x)))
    draws = x[indices]
    counts = np.isfinite(draws).sum(axis=1)
    totals = np.nansum(draws, axis=1)
    sample_means = np.divide(totals, counts, out=np.full(N_BOOTSTRAP, np.nan), where=counts > 0)
    complete = sample_means[np.isfinite(sample_means)]
    result["bootstrap_valid"] = len(complete)
    if valid.sum() >= 6 and len(complete) >= 1800:
        result["interval"] = np.quantile(complete, (0.025, 0.975)).tolist()
    return result


def _panel(records: dict[str, Json], experiment: str, world: str, model: str, metric: str) -> np.ndarray:
    """[seed, day] with day order fixed across the simulator interventions."""
    rows = []
    for seed in SEEDS:
        record = records[Job(experiment, world, model, seed).key]
        if len(record["daily"]) != EVAL_DAYS:
            raise ValueError("incomplete evaluation day panel")
        rows.append([_finite(day.get(metric)) for day in record["daily"]])
    return np.asarray(rows, dtype=float)


def _seed_average(values: np.ndarray) -> list[float | None]:
    """Average the matched training seeds within each independent date."""
    result = []
    for column in values.T:
        result.append(float(column.mean()) if np.isfinite(column).all() else None)
    return result


def contrast(records: dict[str, Json], experiment: str, world: str, candidate: str,
             parent: str, metric: str) -> Json:
    effect = _panel(records, experiment, world, candidate, metric) - _panel(records, experiment, world, parent, metric)
    daily = _seed_average(effect)
    return {**summarize_days(daily), "daily": daily, "metric": metric, "world": world,
            "candidate": candidate, "parent": parent}


def interaction(records: dict[str, Json], experiment: str, low: str, high: str,
                candidate: str, parent: str, metric: str) -> Json:
    """Matched day/seed difference-in-difference, high-minus-low intervention."""
    before = _panel(records, experiment, low, candidate, metric) - _panel(records, experiment, low, parent, metric)
    after = _panel(records, experiment, high, candidate, metric) - _panel(records, experiment, high, parent, metric)
    daily = _seed_average(after - before)
    return {**summarize_days(daily), "daily": daily, "metric": metric, "world_low": low,
            "world_high": high, "candidate": candidate, "parent": parent}


def all_effects(records: dict[str, Json]) -> Json:
    """Predeclared model contrasts and conditional interactions."""
    definitions = []
    for backbone in ("gru", "lru"):
        for operator in ("multiscale", "surprise"):
            definitions.append(("E1", W1, W2, f"{backbone}-{operator}", f"{backbone}-base",
                                (CF, "direction_acf_mae", "memory_gap_64")))
    definitions.extend((
        ("E2", W1, W3, "matrix-noshift", "flat-gru", (CF, "rel_ticks_nll")),
        ("E2", W1, W3, "matrix-advect", "matrix-noshift", (CF, "rel_ticks_nll")),
        ("E2", W1, W3, "flat-readoutgate", "flat-gru", (CF, "rel_ticks_nll")),
        ("E3", W3, W6, "selective-ssm", "gru-base", (CF, "dt_nll", "zero_wait_rate_error", "interarrival_cv2_error")),
        ("E3", W3, W6, "gru-hawkes", "gru-base", (CF, "dt_nll", "zero_wait_rate_error", "interarrival_cv2_error")),
    ))
    comparisons = {}
    interactions = {}
    for experiment, low, high, candidate, parent, metrics in definitions:
        for metric in metrics:
            name = f"{experiment}/{candidate}-vs-{parent}/{metric}"
            comparisons[name] = {
                world: contrast(records, experiment, world, candidate, parent, metric)
                for world in (low, high)
            }
            interactions[name] = interaction(records, experiment, low, high, candidate, parent, metric)
    return {"comparisons": comparisons, "interactions": interactions}


def _status(summary: Json, manipulation_passed: bool) -> str:
    interval = summary["interval"]
    if not manipulation_passed or interval is None:
        return "INCONCLUSIVE"
    if interval[1] < 0:
        return "SUPPORTED_WITH_BOUNDARY"
    if interval[0] > 0:
        return "REFUTED"
    return "INCONCLUSIVE"


def freeze_laws(effects: Json, checks: Json) -> Json:
    """Freeze all three hypotheses, including negative/undefined results."""
    interactions = effects["interactions"]
    h1_rows = [row["daily"] for key, row in interactions.items()
               if key.startswith("E1/") and key.endswith("/" + CF)]
    h1 = np.asarray(h1_rows, dtype=float)
    daily_h1 = [float(column.mean()) if np.isfinite(column).all() else None for column in h1.T]
    h1_summary = {**summarize_days(daily_h1), "daily": daily_h1}
    h2 = interactions[f"E2/matrix-advect-vs-matrix-noshift/{CF}"]
    h3 = interactions["E3/gru-hawkes-vs-gru-base/dt_nll"]
    poisson = effects["comparisons"]["E3/gru-hawkes-vs-gru-base/dt_nll"][W3]
    eq = poisson["interval"] is not None and -0.02 <= poisson["interval"][0] <= poisson["interval"][1] <= 0.02
    hypotheses = (
        ("H1", h1_summary, checks["memory"], "equal-weight impact MAE interaction over four backbone/operator comparisons",
         "individual operator and backbone results reported; not universal"),
        ("H2", h2, checks["thin"], "advect vs no-shift impact MAE interaction; identical parameter count and state size",
         "matrix vs flat and readout gate are secondary controls"),
        ("H3", h3, checks["burst"], "Hawkes vs GRU dt NLL interaction",
         "impact and burst residuals reported separately; no-gain equivalence requires the Poisson interval within ±0.02 nats"),
    )
    result = {}
    for key, primary, passed, estimand, boundary in hypotheses:
        result[key] = {"status": _status(primary, passed), "primary": primary,
                       "manipulation_passed": passed, "estimand": estimand, "boundary": boundary}
    result["H3"]["poisson_equivalence_supported"] = eq
    result["H3"]["poisson_contrast"] = poisson
    return result
