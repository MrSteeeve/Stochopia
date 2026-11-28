"""Fixed twelve-fit ablation of the original M3 beta winner (no provider use).

The registered A00 baseline changes the recurrent topology as well as the
two named mechanisms; the 2x2 contrast is therefore a model-configuration
interaction, not an isolated causal decomposition of just two operators.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import torch

from experiments.synthetic_tau_z.design import (
    BATCH_WINDOWS, BOOTSTRAP_SEED, CONTEXT, DATA_SEED, EVAL_DAYS,
    EVENTS_PER_SESSION, FIT_PROTOCOL, K, N_BOOTSTRAP, ROLLOUT_SEED, SEEDS,
    TRAIN_DAYS, TRUTH_K, TRUTH_SEED, W2, summarize_days,
)
from experiments.synthetic_tau_z.discovery_metrics import (
    MeasurementPanel, evaluate_discovery_model, reference_calibration,
)
from stochopia.common import atomic_write_bytes, atomic_write_json
from stochopia.neural.lob_interface import LOBWorldModel, verify_world_model_contract
from stochopia.neural.lob_training import encode_day, fit_lob_world_model
from stochopia.sim.queue_reactive_data import (
    counterfactual_truth, generate_dataset, generate_evaluation_dataset,
)
from stochopia.sim.queue_reactive_world import standard_worlds

Json = dict[str, Any]
VARIANTS = ("A00", "A10", "A01", "A11")
B0_SHA256 = "548c520efb63cccff9567a22efdd4224ff3a5d126e6aba86d2eb0abd4dd47d35"
WINNER_SHA256 = "563f34670ced48a5762f0e467e93745ce32cb896efce63498458ac745e779660"
METRICS = (
    "val_nll_total", "dt_nll", "event_nll", "level_nll", "size_nll",
    "counterfactual_impact_mae", "post_stop_relaxation_delta",
    "sign_acf_abs_error", "memory_gap_1", "memory_gap_4", "memory_gap_16",
    "memory_gap_64", "exhausted_replicate_rate",
)
CAVEAT = (
    "The prescribed A00 is a 60-dimensional Flat-GRU, while A10/A01/A11 "
    "split state into 52+8. A01 also removes the input-conditioned clock. "
    "The interaction includes the topology and clock changes and does not "
    "identify only two isolated operators. The M3 holdout is disjoint from "
    "training but has already been opened in M3; this is a retrospective "
    "mechanism ablation, not a fresh confirmatory holdout."
)
VARIANT_DETAILS = {
    "A00": {"c03": False, "c08": False, "description": "Unmodified frozen Flat-GRU B0."},
    "A10": {"c03": True, "c08": False, "description": (
        "Unmodified multiscale input-conditioned clock; replace innovation "
        "with measure(x).tanh(). Keep the now-unused predict module registered "
        "to preserve parameter allocation and initialization; 480 disconnected parameters."
    )},
    "A01": {"c03": False, "c08": True, "description": (
        "Keep prediction-subtracted innovation and its -2 gain bias; remove "
        "clock module, use pace=1 and eight identical rates of 0.0707."
    )},
    "A11": {"c03": True, "c08": True, "description": "Unmodified frozen c06 winner."},
}


# --- Frozen source and data receipts ---

def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _read_json(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_exact(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(f"expected exactly one registered source fragment: {before!r}")
    return source.replace(before, after, 1)


def build_variants(baseline: bytes, winner: bytes) -> dict[str, bytes]:
    """Verify both original files, then change only the prescribed source fragments."""
    if digest(baseline) != B0_SHA256 or digest(winner) != WINNER_SHA256:
        raise ValueError("frozen B0/c06 source fingerprint does not match M3")
    source = winner.decode("utf-8")
    a10 = _replace_exact(
        source,
        "        innovation = self.measure(x).tanh() - self.predict(prior).tanh()\n",
        "        innovation = self.measure(x).tanh()\n",
    )
    a01 = _replace_exact(source, "        self.clock = nn.Linear(32, 1)\n", "")
    a01 = _replace_exact(a01, "        nn.init.zeros_(self.clock.weight)\n", "")
    a01 = _replace_exact(a01, "        nn.init.zeros_(self.clock.bias)\n", "")
    a01 = _replace_exact(a01, 'torch.logspace(-2, -0.3, 8)', 'torch.full((8,), 0.0707)')
    a01 = _replace_exact(a01, "        pace = 0.5 + self.clock(x).sigmoid()\n", "        pace = 1.0\n")
    return {"A00": baseline, "A10": a10.encode(), "A01": a01.encode(), "A11": winner}


def load_frozen_sources(artifact: Path) -> dict[str, bytes]:
    folder = artifact / "trajectories" / "env_beta" / "true_z_d" / "candidates"
    final = _read_json(artifact / "freeze.json")["trajectories"]["env_beta/true_z_d"]
    if final["final_id"] != "c06" or final["final_sha256"] != WINNER_SHA256:
        raise ValueError("M3 frozen beta True-Z winner is not the registered c06")
    return build_variants((folder / "B0" / "candidate.py").read_bytes(),
                          (folder / "c06" / "candidate.py").read_bytes())


def load_factory(source: bytes) -> Callable[[], LOBWorldModel]:
    """Execute only the verified frozen source and the two exact static edits."""
    namespace: dict[str, Any] = {"__name__": "candidate"}
    exec(compile(source, "candidate.py", "exec"), namespace)
    factory = namespace["CandidateLOBWorldModel"]
    if not isinstance(factory, type) or not issubclass(factory, LOBWorldModel):
        raise ValueError("wrong frozen model factory")
    return factory


def data_protocol() -> Json:
    return {
        "data_seed": DATA_SEED, "truth_seed": TRUTH_SEED, "train_days": TRAIN_DAYS,
        "dev_days": EVAL_DAYS, "holdout_days": EVAL_DAYS,
        "holdout_first_day": TRAIN_DAYS + EVAL_DAYS,
        "events_per_session": EVENTS_PER_SESSION, "context": CONTEXT, "sessions_per_day": 2,
        "training_seeds": list(SEEDS), "fit": asdict(FIT_PROTOCOL), "model_k": K,
        "truth_k": TRUTH_K, "rollout_seed": ROLLOUT_SEED, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": N_BOOTSTRAP, "batch_windows": BATCH_WINDOWS,
    }


def registered_panel(artifact: Path) -> tuple[Sequence, MeasurementPanel, Json]:
    """Regenerate train and holdout separately, and compare exact M3 pickle hashes."""
    protocol = data_protocol()
    frozen = _read_json(artifact / "registration.json")
    development = _read_json(artifact / "baseline.json")["env_beta"]["development_receipt"]
    heldout = _read_json(artifact / "env_beta-holdout-receipt.json")
    if frozen["data_protocol"] != protocol or development["protocol"] != protocol:
        raise ValueError("runtime protocol differs from the frozen M3 protocol")
    if heldout["freeze_sha256"] != digest((artifact / "freeze.json").read_bytes()):
        raise ValueError("holdout receipt no longer matches the M3 freeze")
    spec = standard_worlds()[W2]
    train = generate_dataset(
        spec, seed=DATA_SEED, train_days=TRAIN_DAYS, eval_days=0,
        events_per_session=EVENTS_PER_SESSION, context_len=CONTEXT,
    ).train
    test = generate_evaluation_dataset(
        spec, seed=DATA_SEED, first_day=TRAIN_DAYS + EVAL_DAYS, eval_days=EVAL_DAYS,
        events_per_session=EVENTS_PER_SESSION, context_len=CONTEXT,
    )
    truth = counterfactual_truth(test, num_replicates=TRUTH_K, seed=TRUTH_SEED)
    scale, participation = reference_calibration(train)
    panel = MeasurementPanel(test.evaluation, test.windows, truth, scale, participation)
    train_hash = digest(pickle.dumps(train, protocol=pickle.HIGHEST_PROTOCOL))
    test_hash = digest(pickle.dumps(panel, protocol=pickle.HIGHEST_PROTOCOL))
    train_dates = [int(day.trading_date) for day in train]
    holdout_dates = [int(day.trading_date) for day in panel.evaluation]
    if train_hash != development["file_hashes"]["train.pkl"] or test_hash != heldout["data_sha256"]:
        raise ValueError("regenerated train/holdout data differs bytewise from M3")
    if train_dates != development["train_dates"] or holdout_dates != heldout["dates"]:
        raise ValueError("regenerated calendars differ from M3")
    if set(train_dates).intersection(holdout_dates):
        raise ValueError("training and holdout dates overlap")
    return train, panel, {
        "world": "env_beta",
        "data_protocol": protocol,
        "train_dates": train_dates,
        "holdout_dates": holdout_dates,
        "train_sha256": train_hash,
        "holdout_sha256": test_hash,
        "source_freeze_sha256": heldout["freeze_sha256"],
        "reference": {"scale_ticks": scale, "participation": participation},
    }


def _number(value: object) -> float | None:
    return float(value) if type(value) in (int, float) and isfinite(value) else None


# --- Equal-seed pooling and physical-day contrasts ---

def pool_training_seeds(seed_daily: Mapping[str, Sequence[Json]]) -> list[Json]:
    """Keep a metric on a day only when all three registered seed fits have it."""
    if set(seed_daily) != set(map(str, SEEDS)):
        raise ValueError("all three registered training seeds are required")
    panels = [seed_daily[str(seed)] for seed in SEEDS]
    daily = []
    for rows in zip(*panels, strict=True):
        date = rows[0]["trading_date"]
        if any(row["trading_date"] != date for row in rows):
            raise ValueError("seed calendars differ")
        item: Json = {"trading_date": date}
        for metric in METRICS:
            values = [_number(row.get(metric)) for row in rows]
            item[metric] = None if None in values else float(np.mean(values))
        daily.append(item)
    return daily


def paired_contrast(panels: Mapping[str, Sequence[Json]], coefficients: Mapping[str, int]) -> dict[str, Json]:
    """One physical day is one bootstrap unit; all nonzero terms must be available.

    The contrast is computed inside every paired day, before sampling those
    day indices with the shared M3 bootstrap seed. In a four-arm interaction,
    missingness in any arm removes that day from the interaction (not from
    all other contrasts); no day, seed or rollout lane is duplicated into an
    independent observation. Under the registered support rule (<6 days or
    <1800 finite draws), the point is descriptive only and CI is unavailable.
    """
    keys = tuple(key for key, value in coefficients.items() if value)
    if not keys:
        raise ValueError("contrast requires a nonzero coefficient")
    dates = [row["trading_date"] for row in panels[keys[0]]]
    if not dates or len(dates) != len(set(dates)):
        raise ValueError("contrast requires a unique physical-day calendar")
    for key in keys:
        if [row["trading_date"] for row in panels[key]] != dates:
            raise ValueError("paired contrast requires matching day calendars")
    result = {}
    for metric in METRICS:
        daily = []
        for day in range(len(dates)):
            terms = [_number(panels[key][day].get(metric)) for key in keys]
            value = None
            if None not in terms:
                value = float(sum(coefficients[key] * term for key, term in zip(keys, terms, strict=True)))
            daily.append(value)
        result[metric] = {**summarize_days(daily), "daily": daily}
    return result


# --- Fixed fits and retrospective holdout measurement ---

def _fit_and_evaluate(
    variant: str, factory: Callable[[], LOBWorldModel], encoded: Sequence,
    panel: MeasurementPanel, output: Path, temporary: Path,
) -> Json:
    fits = []
    capacities = set()
    variant_work = temporary / variant
    variant_work.mkdir()
    for seed in SEEDS:
        print(f"FIT {variant}/{seed}", flush=True)
        fitted = fit_lob_world_model(factory, encoded, seed, protocol=FIT_PROTOCOL)
        capacity = verify_world_model_contract(fitted.model, max_params=28_000, max_state_dim=60)
        capacities.add(capacity)
        weights = variant_work / f"{seed}.pt"
        torch.save(fitted.model.state_dict(), weights)
        receipt = {
            "seed": seed, "fit_seconds": fitted.seconds,
            "last_nll": fitted.last_nll, "checkpoint_sha256": digest(weights.read_bytes()),
        }
        atomic_write_json(output / "fits" / f"{variant}-{seed}.json", receipt)
        fits.append(receipt)
    if len(capacities) != 1:
        raise ValueError("trained model capacity differs across seeds")
    params, state = capacities.pop()
    if state != 60:
        raise ValueError("all four variants must use exactly 60 state scalars")

    seed_daily = {}
    for seed in SEEDS:
        print(f"HOLDOUT {variant}/{seed}", flush=True)
        torch.manual_seed(seed)
        model = factory().eval()
        model.load_state_dict(torch.load(variant_work / f"{seed}.pt", map_location="cpu", weights_only=True))
        if verify_world_model_contract(model, max_params=28_000, max_state_dim=60) != (params, state):
            raise ValueError("checkpoint reload changed registered capacity")
        seed_daily[str(seed)] = evaluate_discovery_model(model, panel)
    daily = pool_training_seeds(seed_daily)
    support = {metric: summarize_days([row[metric] for row in daily]) for metric in METRICS}
    return {
        "params": params, "state": state, "fits": fits, "seed_daily": seed_daily,
        "used_params": params - 480 if variant == "A10" else params,
        "daily": daily, "aggregate": {metric: item["estimate"] for metric, item in support.items()},
        "support": support,
    }


def reference_reproduction(artifact: Path, models: Mapping[str, Json]) -> Json:
    """Compare independently refitted original configurations against frozen M3."""
    previous = _read_json(artifact / "holdout.json")["env_beta"]
    result = {}
    for variant in ("A00", "A11"):
        model = models[variant]
        original = previous[model["source_sha256"]]["evaluation"]
        hashes = original["training_checkpoint_hashes"]
        metrics = {}
        for metric in METRICS:
            prior = _number(original["aggregate"].get(metric))
            current = _number(model["aggregate"].get(metric))
            metrics[metric] = {"m3": prior, "ablated_rerun": current,
                               "difference": None if prior is None or current is None else current - prior}
        result[variant] = {
            "checkpoint_hash_match": {
                str(fit["seed"]): fit["checkpoint_sha256"] == hashes[str(fit["seed"])]
                for fit in model["fits"]
            },
            "aggregate_comparison": metrics,
        }
    return result


def run(artifact: Path, output: Path) -> Json:
    artifact, output = artifact.resolve(), output.resolve()
    if output == artifact or artifact in output.parents or output in artifact.parents:
        raise ValueError("ablation output must be outside the frozen M3 evidence tree")
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a fresh ablation output directory; existing receipts are not overwritten")
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    sources = load_frozen_sources(artifact)
    hashes = {variant: digest(source) for variant, source in sources.items()}
    # All variants are fixed and recorded before fitting or reading held-out outcomes.
    for variant, source in sources.items():
        atomic_write_bytes(output / "sources" / variant / "candidate.py", source)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, panel, registration = registered_panel(artifact)
    registration.update(source_sha256=hashes, variants=VARIANT_DETAILS, caveat=CAVEAT,
                        fit_roster=[{"variant": variant, "seed": seed} for variant in VARIANTS for seed in SEEDS],
                        metrics=list(METRICS), api_calls=0)
    atomic_write_json(output / "registration.json", registration)
    encoded = [encode_day(day) for day in train]
    models = {}
    with TemporaryDirectory(prefix="m31-ablation-") as directory:
        for variant in VARIANTS:
            result = _fit_and_evaluate(variant, load_factory(sources[variant]), encoded, panel,
                                       output, Path(directory))
            result["source_sha256"] = hashes[variant]
            models[variant] = result
            atomic_write_json(output / "models" / f"{variant}.json", result)
    formal_fits = sum(len(model["fits"]) for model in models.values())
    if formal_fits != 12:
        raise ValueError("wrong ablation fit roster")
    panels = {variant: item["daily"] for variant, item in models.items()}
    paired = {variant: paired_contrast(panels, {variant: 1, "A00": -1})
              for variant in VARIANTS if variant != "A00"}
    edges = (("A11", "A10"), ("A01", "A00"), ("A11", "A01"), ("A10", "A00"))
    contrasts = {f"{a}-{b}": paired_contrast(panels, {a: 1, b: -1}) for a, b in edges}
    synergy = paired_contrast(panels, {"A11": 1, "A10": -1, "A01": -1, "A00": 1})
    result = {
        "schema": "m31-c06-ablation-v1", "registration": registration, "formal_fits": formal_fits,
        "models": models, "paired_vs_A00": paired, "contrasts": contrasts, "synergy": synergy,
        "m3_reference_reproduction": reference_reproduction(artifact, models),
        "wall_seconds": perf_counter() - started, "api_calls": 0,
    }
    atomic_write_json(output / "ablation.json", result)
    print(f"Completed {formal_fits} fits; evidence: {output / 'ablation.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.m3_artifact, args.output_dir)


if __name__ == "__main__":
    main()
