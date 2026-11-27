"""Fixed-budget fits and K=4 synthetic day-level measurements."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from experiments.synthetic_tau_z.design import (
    BATCH_WINDOWS, CONTEXT, FIT_PROTOCOL, K, ROLLOUT_SEED, Job,
)
from stochopia.common import atomic_write_json
from stochopia.harness.lob_rollout import RolloutArm, RolloutTrace, RolloutWindow, rollout_paired_windows
from stochopia.measure.continuous_impact import counterfactual_impact_mae
from stochopia.neural.lob_interface import LOBBatch, LOBWorldModel, make_batch, verify_world_model_contract
from stochopia.neural.lob_training import (
    EncodedDay, compute_memory_horizon_profile, encode_day, evaluate_world_model_nll, fit_lob_world_model,
)
from stochopia.neural.probes import make_probe

Json = dict[str, Any]
LAGS = 32


def _visible_response(replica: dict[RolloutArm, RolloutTrace]) -> np.ndarray:
    """[grid] signed buy/sell displacement, both measured against the free lane."""
    free = replica[RolloutArm.NO_INJECTION].mid_ticks
    buy = replica[RolloutArm.BUY_VISIBLE].mid_ticks - free
    sell = replica[RolloutArm.SELL_VISIBLE].mid_ticks - free
    return 0.5 * (buy - sell)


def _mean_finite(values: Sequence[float | None]) -> float | None:
    finite = [float(x) for x in values if x is not None and isfinite(x)]
    return float(np.mean(finite)) if finite else None


def _mean_marks(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    return np.divide(np.where(finite, values, 0.0).sum(axis=0), counts,
                     out=np.full(values.shape[1:], np.nan), where=counts > 0)


def _acf(actions: np.ndarray) -> np.ndarray:
    """Lag 1..32 correlation of aggressive trade signs; unavailable lag = NaN."""
    sides = actions[(actions == 2) | (actions == 3)]
    signs = 1 - 2 * (sides % 2)
    result = np.full(LAGS, np.nan)
    if len(signs) < 2:
        return result
    centered = signs.astype(float) - signs.mean()
    denom = np.dot(centered, centered)
    # The variance of a one-sided trace is zero, so no normalized ACF.
    if denom <= 0:
        return result
    for lag in range(1, min(LAGS + 1, len(centered))):
        result[lag - 1] = np.dot(centered[:-lag], centered[lag:]) / denom
    return result


def _clock(times: np.ndarray) -> tuple[float | None, float | None]:
    """Consecutive timestamp gaps, omitting arbitrary first-event anchoring."""
    gaps = np.diff(times)
    if not len(gaps):
        return None, None
    mean = float(gaps.mean())
    return float((gaps == 0).mean()), float(gaps.var() / (mean + 1e-9) ** 2)


def _historical(window: RolloutWindow, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    day = window.day
    start = window.start_index
    session_end = next(end for lo, end in day.session_bounds if lo <= start < end)
    times = day.time_ms[start:session_end] - day.time_ms[start]
    end = int(np.searchsorted(times, horizon, side="right"))
    return times[:end], day.action_type[start:start + end]


def _window_metrics(
    window: RolloutWindow, replicas: Sequence[dict[RolloutArm, RolloutTrace]],
    truth: np.ndarray, grid: np.ndarray,
) -> Json:
    """Average K inside the original physical window; record quote support."""
    if len(replicas) != K:
        raise ValueError("model rollout must have exactly the registered K replicas")
    paths = np.stack([_visible_response(replica) for replica in replicas])
    curve = _mean_marks(paths)
    error = None
    if np.isfinite(curve).all() and np.isfinite(truth).all():
        error = counterfactual_impact_mae(curve, truth)
    historical_times, historical_actions = _historical(window, int(grid[-1]))
    historical_acf = _acf(historical_actions)
    historical_clock = _clock(historical_times)
    generated_acf = []
    generated_clock = []
    for replica in replicas:
        events = replica[RolloutArm.NO_INJECTION].events
        generated_acf.append(_acf(events[:, 1]))
        generated_clock.append(_clock(events[:, 0]))
    profile = _mean_marks(np.stack(generated_acf))
    common = np.isfinite(profile) & np.isfinite(historical_acf)
    acf_error = float(np.abs(profile[common] - historical_acf[common]).mean()) if common.any() else None
    zero = _mean_finite([row[0] for row in generated_clock])
    cv2 = _mean_finite([row[1] for row in generated_clock])
    from stochopia.market.rollout_eval import exhausted_replicate_rate

    return {
        "counterfactual_impact_mae": error,
        "direction_acf_mae": acf_error,
        "acf_support_lags": int(common.sum()),
        "zero_wait_rate_error": None if zero is None or historical_clock[0] is None else abs(zero - historical_clock[0]),
        "interarrival_cv2_error": None if cv2 is None or historical_clock[1] is None else abs(cv2 - historical_clock[1]),
        "zero_wait_rate_generated": zero, "zero_wait_rate_historical": historical_clock[0],
        "interarrival_cv2_generated": cv2, "interarrival_cv2_historical": historical_clock[1],
        "exhausted_replicate_rate": exhausted_replicate_rate(replicas),
        "available_quote_fraction": float(np.isfinite(paths).mean()),
    }


def _day_batch(day: EncodedDay) -> list[LOBBatch]:
    batches = []
    for lo, hi in day.day.session_bounds:
        targets = day.tokens[None, lo:hi]
        batch = make_batch(day.ticks[None, lo:hi], day.shares[None, lo:hi], targets)
        batches.append(batch)
    return batches


def _day_likelihood(model: LOBWorldModel, day: EncodedDay) -> Json:
    """Fixed supervised positions across the full sequence and all reset lengths."""
    batches = _day_batch(day)
    supervised = slice(FIT_PROTOCOL.burn_in, None)
    nll = evaluate_world_model_nll(model, batches, supervised=supervised)
    memory = compute_memory_horizon_profile(model, batches, intervals=(1, 4, 16, 64),
                                            supervised=supervised)
    keys = {"val_nll_total": nll["total"], "dt_nll": nll["dt"], "action_nll": nll["action"],
            "rel_ticks_nll": nll["rel_ticks"], "volume_nll": nll["volume"]}
    keys.update({f"memory_gap_{length}": float(value) for length, value in memory.items()})
    return keys


def evaluate_model(model: LOBWorldModel, dataset: Any, truth: np.ndarray) -> list[Json]:
    """Return one record per evaluation day, never per seed or Monte Carlo lane."""
    from stochopia.sim.queue_reactive_world import CHILD_SHARES, CHILD_TIMES_MS, GRID_MS

    daily = [_day_likelihood(model, encode_day(day)) for day in dataset.evaluation]
    windows = dataset.windows
    if truth.shape != (len(windows), len(GRID_MS)) or len(windows) != 2 * len(daily):
        raise ValueError("truth and 2-session evaluation calendar do not align")
    by_day: list[list[Json]] = [[] for _ in daily]
    for first in range(0, len(windows), BATCH_WINDOWS):
        chunk = windows[first:first + BATCH_WINDOWS]
        result = rollout_paired_windows(
            model, chunk, child_times_ms=CHILD_TIMES_MS, child_shares=CHILD_SHARES,
            grid_ms=GRID_MS, seed=ROLLOUT_SEED, context_len=CONTEXT, num_replicates=K,
        )
        for index, (window, replicas) in enumerate(zip(chunk, result, strict=True), start=first):
            if window.day.trading_date != dataset.evaluation[index // 2].trading_date:
                raise ValueError("evaluation windows must retain day/session order")
            by_day[index // 2].append(_window_metrics(window, replicas, truth[index], GRID_MS))
    for index, (row, windows_rows) in enumerate(zip(daily, by_day, strict=True)):
        if len(windows_rows) != 2:
            raise ValueError("expected two physical windows per independent day")
        row["trading_date"] = int(dataset.evaluation[index].trading_date)
        for key in windows_rows[0]:
            values = [item[key] for item in windows_rows]
            # A primary impact day requires both planned windows. No quotation
            # is fabricated if either model or truth is unavailable.
            row[key] = None if key == "counterfactual_impact_mae" and None in values else _mean_finite(values)
    return daily


def fit_and_measure(job: Job, dataset: Any, truth: np.ndarray, work: Path) -> Json:
    """Fit once per registered tuple, cache weights before any evaluation retry."""
    work.mkdir(parents=True, exist_ok=True)
    import json

    fit_path = work / "fit.json"
    weights = work / "weights.pt"
    factory = lambda: make_probe(job.model)
    if fit_path.is_file():
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        model = factory().eval()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    else:
        fitted = fit_lob_world_model(factory, [encode_day(day) for day in dataset.train], job.seed,
                                     protocol=FIT_PROTOCOL)
        model = fitted.model
        torch.save(model.state_dict(), weights)
        fit = {"fit_seconds": fitted.seconds, "last_train_nll": fitted.last_nll, "seed": job.seed}
        atomic_write_json(fit_path, fit)
    parameters, state = verify_world_model_contract(model, max_state_dim=60)
    if state != 60:
        raise ValueError(f"{job.model} did not use the registered 60-scalar state budget")
    started = perf_counter()
    daily = evaluate_model(model, dataset, truth)
    aggregate = {key: _mean_finite([row[key] for row in daily]) for key in daily[0] if key != "trading_date"}
    result = {
        "job_id": job.key, "experiment": job.experiment, "world": job.world,
        "model": job.model, "train_seed": job.seed, "parameters": parameters, "state_scalars": state,
        "model_k": K, "fit": fit, "evaluation_seconds": perf_counter() - started,
        "aggregate": aggregate, "daily": daily,
    }
    atomic_write_json(work / "result.json", result)
    return result
