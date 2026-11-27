"""Run the public, local-only 78-fit P2-M2 synthetic experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import subprocess
from time import perf_counter
from typing import Any

import numpy as np

from experiments.p2_m2_design import (
    CONTEXT, DATA_SEED, EVAL_DAYS, EVENTS_PER_SESSION, TRAIN_DAYS,
    TRUTH_K, TRUTH_SEED, W1, W2, W3, W4, W6, Job,
    all_effects, freeze_laws, predeclared_protocol, roster,
)
from stochopia.common import atomic_write_json
from stochopia.measure.data_properties import compute_property_drift, measure_data_properties
from stochopia.sim.queue_reactive_world import (
    CHILD_SHARES, CHILD_TIMES_MS, GRID_MS, counterfactual_truth, generate_dataset,
    standard_worlds, tier0_manipulation_check,
)

Json = dict[str, Any]
_WORLD_CACHE: dict[str, tuple[Any, np.ndarray]] = {}
ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _freeze(path: Path, value: Json) -> None:
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"frozen registration differs: {path.name}")
    else:
        atomic_write_json(path, value)


def register(artifact: Path, work: Path, jobs: int) -> Json:
    """Register the exact 78 fits and generator configuration before model training."""
    worlds = standard_worlds()
    if set(worlds) != {W1, W2, W3, W4, W6}:
        raise ValueError("expected five independent synthetic world recipes")
    from stochopia.neural.p2_probes import make_probe

    registration = predeclared_protocol()
    parameter_counts = {}
    for model in sorted({job.model for job in roster()}):
        module = make_probe(model)
        parameter_counts[model] = sum(p.numel() for p in module.parameters() if p.requires_grad)
    registration.update(
        code_commit=_git(ROOT, "rev-parse", "HEAD"), worlds={name: asdict(spec) for name, spec in worlds.items()},
        world_aliases={"W5_thin_poisson": W3}, grid_ms=GRID_MS.tolist(),
        child_times_ms=list(CHILD_TIMES_MS), child_shares=CHILD_SHARES,
        workers=jobs, training_device="cpu", parameter_counts=parameter_counts, state_scalars=60,
        generation_calibration={
            "initial_generation_only": "Depth 2000 and 8 x 200-share children produced degenerate thick truth and low W6 support.",
            "final_train_anchor_calibration": "Depth 300/100 and 4 x 100-share children chosen before any model fit using training anchors.",
            "formal_model_outcomes_seen": False,
            "truth_condition": "Standard ten-level aggregated snapshot with true latent sign/clock state; common random marks.",
        },
        manipulation_rules={
            "thin": "both thick worlds median shift < .05; both thin worlds > .20",
            "memory": "long memory ACF sum exceeds short by >2 and VR is higher, on both thicknesses",
            "burst": "same thin/short world with alpha=.7 increases median zero waits and interarrival CV²",
            "valid_book": "no empty/crossed pre-event touch in the training panel",
        },
    )
    _freeze(work / "registration.json", registration)
    _freeze(artifact / "registration.json", registration)
    return registration


def _median(properties: Json, dimension: str, name: str) -> float:
    return properties[dimension][name]["median"]


def manipulation_checks(properties: dict[str, Json], invalid: dict[str, float]) -> Json:
    """Predeclared threshold checks, computed only from historical training sessions."""
    def value(world: str, dimension: str, name: str) -> float:
        return _median(properties[world], dimension, name)

    thin = all(value(world, "z_thin", "midprice_shift_rate") < 0.05 for world in (W1, W2))
    thin = thin and all(value(world, "z_thin", "midprice_shift_rate") > 0.20 for world in (W3, W4))
    memory = True
    for short, long in ((W1, W2), (W3, W4)):
        acf_difference = value(long, "z_mem", "sign_acf_sum_1_32") - value(short, "z_mem", "sign_acf_sum_1_32")
        vr_higher = value(long, "z_mem", "flow_vr_32_4") > value(short, "z_mem", "flow_vr_32_4")
        memory = memory and acf_difference > 2.0 and vr_higher
    burst = all(value(W6, "z_burst", name) > value(W3, "z_burst", name)
                for name in ("zero_wait_prob", "interarrival_cv2"))
    return {"thin": thin, "memory": memory, "burst": burst, "valid_book": all(x == 0 for x in invalid.values())}


def _invalid_touch(days: Any) -> float:
    touch = np.concatenate([day.pre_book_ticks[:, :, 0] for day in days], axis=0)
    return float(((touch[:, 0] <= 0) | (touch[:, 1] <= touch[:, 0])).mean())


def _prepare_worlds(artifact: Path, work: Path, registration: Json) -> Json:
    """Generate five datasets once and cache exact-generator truth, before any fit."""
    base = work / "worlds"
    base.mkdir(exist_ok=True)
    properties = {}
    invalid = {}
    receipts = {}
    for name, spec in standard_worlds().items():
        folder = base / name
        folder.mkdir(exist_ok=True)
        dataset_path = folder / "dataset.pkl"
        truth_path = folder / "truth.npy"
        support_path = folder / "truth-support.npy"
        complete = folder / "complete.json"
        if complete.exists():
            receipt = _read(complete)
            if (receipt["dataset_sha256"] != _digest(dataset_path) or receipt["truth_sha256"] != _digest(truth_path)
                    or receipt["truth_support_sha256"] != _digest(support_path)):
                raise ValueError("cached synthetic dataset or truth differs")
            with dataset_path.open("rb") as handle:
                dataset = pickle.load(handle)
            truth = np.load(truth_path, allow_pickle=False)
            support = np.load(support_path, allow_pickle=False)
        else:
            started = perf_counter()
            print(f"GENERATE {name}", flush=True)
            dataset = generate_dataset(spec, seed=DATA_SEED, train_days=TRAIN_DAYS, eval_days=EVAL_DAYS,
                                       events_per_session=EVENTS_PER_SESSION, context_len=CONTEXT)
            if len(dataset.train) != TRAIN_DAYS or len(dataset.evaluation) != EVAL_DAYS:
                raise ValueError("wrong synthetic calendar")
            with dataset_path.open("wb") as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            truth, support = counterfactual_truth(dataset, grid_ms=GRID_MS, child_times_ms=CHILD_TIMES_MS,
                                                  child_shares=CHILD_SHARES, num_replicates=TRUTH_K,
                                                  seed=TRUTH_SEED, return_support=True)
            truth = np.asarray(truth, dtype=float)
            np.save(truth_path, truth, allow_pickle=False)
            np.save(support_path, support, allow_pickle=False)
            receipt = {"dataset_sha256": _digest(dataset_path), "truth_sha256": _digest(truth_path),
                       "truth_support_sha256": _digest(support_path),
                       "generation_truth_seconds": perf_counter() - started}
            atomic_write_json(complete, receipt)
        if truth.shape != (2 * EVAL_DAYS, len(GRID_MS)) or not np.isfinite(truth).all():
            raise ValueError(f"{name} truth must provide all planned window/time marks")
        if support.shape != truth.shape or (support < 1).any() or (support > TRUTH_K).any():
            raise ValueError("known-generator truth support does not align")
        peak = float(np.abs(truth.mean(axis=0)).max())
        if peak <= 0.0:
            raise ValueError(f"{name} has a degenerate zero mean intervention response")
        properties[name] = measure_data_properties(dataset.train)
        invalid[name] = _invalid_touch(dataset.train)
        receipts[name] = {
            **receipt, "train_days": [int(day.trading_date) for day in dataset.train],
            "eval_days": [int(day.trading_date) for day in dataset.evaluation],
            "true_curve_mean_ticks": truth.mean(axis=0).tolist(),
            "true_curve_by_day_ticks": truth.reshape(EVAL_DAYS, 2, len(GRID_MS)).mean(axis=1).tolist(),
            "truth_peak_abs_mean_ticks": peak,
            "truth_k": TRUTH_K, "truth_support_min": int(support.min()),
            "truth_support_mean_fraction": float(support.mean() / TRUTH_K),
        }
        print(f"WORLD {name} READY", flush=True)
    checks = manipulation_checks(properties, invalid)
    drift = {}
    for low, high in ((W1, W2), (W3, W4), (W1, W3), (W2, W4), (W3, W6)):
        drift[f"{low}->{high}"] = compute_property_drift(properties[low], properties[high])
    result = {"properties": properties, "training_invalid_spread_rate": invalid, "checks": checks,
              "property_drift": drift, "tier0": tier0_manipulation_check(),
              "world_receipts": receipts, "protocol": registration["manipulation_rules"]}
    atomic_write_json(artifact / "manipulation_check.json", result)
    if not all(checks.values()):
        raise RuntimeError(f"synthetic manipulation not established before formal fits: {checks}")
    return result


def _initialize_worker() -> None:
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _worker(job: Job, work: Path) -> Json:
    from experiments.p2_m2_evaluate import fit_and_measure

    if job.world not in _WORLD_CACHE:
        base = work / "worlds" / job.world
        with (base / "dataset.pkl").open("rb") as handle:
            dataset = pickle.load(handle)
        truth = np.load(base / "truth.npy", allow_pickle=False)
        _WORLD_CACHE[job.world] = dataset, truth
    dataset, truth = _WORLD_CACHE[job.world]
    return fit_and_measure(job, dataset, truth, work / "jobs" / job.key)


def _run_jobs(work: Path, jobs: int) -> dict[str, Json]:
    records = {}
    pending = []
    for job in roster():
        path = work / "jobs" / job.key / "result.json"
        if path.is_file():
            result = _read(path)
            if result["job_id"] != job.key:
                raise ValueError("saved job differs from the roster")
            records[job.key] = result
        else:
            pending.append(job)
    print(f"FIT/MEASURE {len(records)}/78 complete; {len(pending)} remaining", flush=True)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn"),
                             initializer=_initialize_worker) as pool:
        tasks = {pool.submit(_worker, job, work): job for job in pending}
        for future in as_completed(tasks):
            job = tasks[future]
            result = future.result()
            if result["job_id"] != job.key:
                raise ValueError("worker returned another registered tuple")
            records[job.key] = result
            print(f"DONE {len(records)}/78 {job.key} fit={result['fit']['fit_seconds']:.2f}s "
                  f"eval={result['evaluation_seconds']:.2f}s", flush=True)
    if len(records) != 78 or len(list((work / "jobs").glob("*/fit.json"))) != 78:
        raise ValueError("formal fit roster must contain exactly 78 completed training slots")
    return records


def _publish(artifact: Path, records: dict[str, Json], manipulation: Json, started: float) -> None:
    """Publish complete evaluations, day traces of metrics, and honest frozen laws."""
    effects = all_effects(records)
    laws = freeze_laws(effects, manipulation["checks"])
    out = []
    daily = {}
    for job in roster():
        result = records[job.key]
        out.append({
            "job_id": job.key, "experiment": job.experiment, "world": job.world,
            "model": job.model, "train_seed": job.seed, "parameters": result["parameters"],
            "state_scalars": result["state_scalars"], "model_k": result["model_k"],
            "fit_seconds": result["fit"]["fit_seconds"], "evaluation_seconds": result["evaluation_seconds"],
            "last_train_nll": result["fit"]["last_train_nll"], **result["aggregate"],
        })
        daily[job.key] = result["daily"]
    with (artifact / "experiment_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    atomic_write_json(artifact / "daily_results.json", daily)
    atomic_write_json(artifact / "paired_effects.json", effects)
    atomic_write_json(artifact / "frozen_synthetic_laws.json", laws)
    atomic_write_json(artifact / "execution_summary.json", {
        "completed": len(records), "counts": dict(Counter(row["experiment"] for row in out)),
        "total_fit_seconds": sum(row["fit_seconds"] for row in out),
        "total_eval_seconds": sum(row["evaluation_seconds"] for row in out),
        "wall_seconds": perf_counter() - started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "api_calls": 0,
    })
    print("P2-M2 complete: 78 fits, 78 paired day panels; all hypotheses frozen", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true", help="Generate/check worlds without training")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    artifact = args.artifact_root.resolve()
    work = args.work_root.resolve()
    if any(part.startswith("p1-") or part == "p2-m1-validation" for part in artifact.parts + work.parts):
        raise ValueError("M2 outputs must not be under historical experiment directories")
    if work.is_relative_to(artifact):
        raise ValueError("private synthetic cache must be separate from public evidence")
    artifact.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    registration = register(artifact, work, args.jobs)
    _initialize_worker()
    manipulation = _prepare_worlds(artifact, work, registration)
    if args.prepare_only:
        print("Preparation accepted; no model fitting performed.", flush=True)
        return
    records = _run_jobs(work, args.jobs)
    _publish(artifact, records, manipulation, started)


if __name__ == "__main__":
    main()
