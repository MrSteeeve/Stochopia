"""Walk-forward protocol for a joint P/Q market world model.

A *world model* is any forecaster that, at an origin session ``t``, sees only
fixed-length trailing windows of the public panel ending at ``t`` (504
sessions of returns and rates, 126 of Q quotes) and returns ``n_paths`` joint
samples of

* the next ``HORIZON`` daily log returns of CSI500 and CSI1000 (P side), and
* the Q-side quotes observed on session ``t + HORIZON``: CSI1000 30d ATM IV
  and the CSI500 / CSI1000 futures-implied carry.

The realised market is the external environment.  Every sample is scored with
the sample CRPS (binary probes reduce to the Brier score) against what the
market actually did, relative to a fixed EWMA-GBM / random-walk reference.
Path-dependent targets are payoffs of StructurerDSL programs, so the score
measures how well the model predicts quantities that drive derivative design.

Splits: ``dev`` origins end on ``DEV_END`` and are the only ones a search loop
may observe; ``test`` origins come strictly after it and are scored once at
the end.  Inputs carry no calendar dates or index levels, and their fixed
lengths hide the origin's position in the panel.  This protocol is a
research track and never enters Benchmark v3 aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..common import sha256_text
from .market_panel import UNDERLYING_ORDER, MarketPanel

PROTOCOL = "stochopia.market-world-model-forecast.v1"
HORIZON = 21
N_PATHS = 256
RETURN_WINDOW = 504  # sessions of returns / rates given to the forecaster
Q_WINDOW = 126  # sessions of Q quotes given to the forecaster
MIN_HISTORY = RETURN_WINDOW  # every origin sees full-length windows
ORIGIN_STEP = 21
DEV_END = "2023-06-30"
Q_NAMES = ("iv_1m_CSI1000", "carry_CSI500", "carry_CSI1000")
MAX_ABS_DAILY_RETURN = 0.5
MAX_IV = 3.0
MAX_ABS_CARRY = 2.0
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_BLOCK = 3

Forecaster = Callable[[Mapping[str, np.ndarray], Mapping[str, Any]], tuple[np.ndarray, np.ndarray]]


class ForecastError(ValueError):
    """A forecaster output violated the protocol at one origin."""


def origin_indices(panel: MarketPanel, split: str) -> list[int]:
    if split not in {"dev", "test", "all"}:
        raise ValueError("split must be dev, test or all")
    total = len(panel.dates)
    origins = list(range(MIN_HISTORY, total - HORIZON, ORIGIN_STEP))
    if split == "dev":
        return [t for t in origins if panel.dates[t] <= DEV_END]
    if split == "test":
        return [t for t in origins if panel.dates[t] > DEV_END]
    return origins


def public_history(panel: MarketPanel, t: int) -> dict[str, np.ndarray]:
    """Fixed-length trailing windows ending at session ``t``; nothing after ``t``.

    Fixed lengths keep the origin's position in the panel (and therefore its
    calendar date) out of the input: a full prefix would reveal it through its
    length, and the NaN-to-finite switch at a listing date would anchor it.
    """

    if not MIN_HISTORY <= t < len(panel.dates):
        raise ValueError("origin outside panel or without full history windows")
    end = t + 1
    return {
        "returns": panel.log_returns[end - RETURN_WINDOW : end].copy(),
        "rate_3m": panel.rate_3m[end - RETURN_WINDOW : end].copy(),
        "carry": panel.carry[end - Q_WINDOW : end].copy(),
        "iv_1m": panel.iv_1m[end - Q_WINDOW : end].copy(),
    }


def origin_seed(split: str, t: int, seed: int = 0) -> int:
    return int(sha256_text(f"{PROTOCOL}|{split}|{t}|{seed}")[:15], 16)


def query_for(split: str, t: int, seed: int = 0) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "horizon": HORIZON,
        "n_paths": N_PATHS,
        "seed": origin_seed(split, t, seed),
        "underlyings": list(UNDERLYING_ORDER),
        "q_names": list(Q_NAMES),
    }


def realized(panel: MarketPanel, t: int) -> tuple[np.ndarray, np.ndarray]:
    returns = panel.log_returns[t + 1 : t + 1 + HORIZON]
    if returns.shape != (HORIZON, 2) or not np.all(np.isfinite(returns)):
        raise ValueError("origin has no complete realised horizon")
    target = t + HORIZON
    q = np.array([panel.iv_1m[target], panel.carry[target, 0], panel.carry[target, 1]])
    return returns, q


def validate_forecast(returns: Any, q: Any) -> tuple[np.ndarray, np.ndarray]:
    returns = np.asarray(returns, dtype=float)
    q = np.asarray(q, dtype=float)
    if returns.shape != (N_PATHS, HORIZON, 2):
        raise ForecastError(f"returns must have shape {(N_PATHS, HORIZON, 2)}, got {returns.shape}")
    if q.shape != (N_PATHS, len(Q_NAMES)):
        raise ForecastError(f"q must have shape {(N_PATHS, len(Q_NAMES))}, got {q.shape}")
    if not np.all(np.isfinite(returns)) or np.any(np.abs(returns) > MAX_ABS_DAILY_RETURN):
        raise ForecastError("returns must be finite daily log returns with |r| <= 0.5")
    if not np.all(np.isfinite(q)):
        raise ForecastError("q samples must be finite")
    if np.any(q[:, 0] <= 0) or np.any(q[:, 0] > MAX_IV):
        raise ForecastError(f"implied volatility samples must lie in (0, {MAX_IV}]")
    if np.any(np.abs(q[:, 1:]) >= MAX_ABS_CARRY):
        raise ForecastError(f"carry samples must satisfy |carry| < {MAX_ABS_CARRY}")
    return returns, q


# --------------------------------------------------------------------------
# Reference forecaster (also the seed program's semantics)
# --------------------------------------------------------------------------

Q_DEFAULTS = (0.20, 0.10, 0.10)


def _last_finite(values: np.ndarray, default: float) -> float:
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float(default)


def _change_scale(values: np.ndarray, lag: int, window: int, default: float) -> float:
    recent = values[-window:]
    changes = recent[lag:] - recent[:-lag]
    changes = changes[np.isfinite(changes)]
    return float(np.std(changes)) if changes.size >= 20 else float(default)


def reference_forecast(history: Mapping[str, np.ndarray], query: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """EWMA(0.94) Gaussian GBM with trailing correlation; Q random walk."""

    rng = np.random.default_rng(int(query["seed"]))
    n, horizon = int(query["n_paths"]), int(query["horizon"])
    returns = np.asarray(history["returns"])
    returns = returns[np.all(np.isfinite(returns), axis=1)]
    variance = np.var(returns[:20], axis=0)
    for row in returns[20:]:
        variance = 0.94 * variance + 0.06 * row**2
    corr = np.corrcoef(returns[-252:].T)[0, 1]
    chol = np.linalg.cholesky(np.array([[1.0, corr], [corr, 1.0]]))
    shocks = rng.standard_normal((n, horizon, 2)) @ chol.T
    paths = shocks * np.sqrt(variance)
    series = (np.asarray(history["iv_1m"]), np.asarray(history["carry"])[:, 0], np.asarray(history["carry"])[:, 1])
    q = np.empty((n, len(Q_NAMES)))
    for j, values in enumerate(series):
        level = _last_finite(values, Q_DEFAULTS[j])
        scale = _change_scale(values, horizon, Q_WINDOW, 0.02)
        q[:, j] = level + scale * rng.standard_normal(n)
    q[:, 0] = np.maximum(q[:, 0], 0.01)
    return paths, q


# --------------------------------------------------------------------------
# Targets: plain statistics and StructurerDSL payoff probes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    name: str
    family: str
    underlying: int | None  # index into UNDERLYING_ORDER; None for Q targets
    kind: str  # "continuous" or "binary"


def _probe_programs():
    from structurerdsl import Program, catalog

    daily = tuple(range(1, HORIZON + 1))

    def touch(level: float, direction: str):
        return Program.from_sexpr(
            f"(contract (meta (underlyings u) (maturity {HORIZON}) (schedules)) "
            f"(premium (pay (touched (spot u) {level} {direction} daily) maturity)))"
        )

    return {
        "digital_up": (catalog.digital_call(HORIZON, 1.0), "binary"),
        "ki_touch_95": (touch(0.95, "down"), "binary"),
        "ko_touch_103": (touch(1.03, "up"), "binary"),
        "asian_call": (catalog.arithmetic_asian_call(HORIZON, 1.0, daily), "continuous"),
        "range_accrual": (catalog.range_accrual(HORIZON, 0.97, 1.03, 1.0 / HORIZON, daily), "continuous"),
        "mini_snowball": (catalog.snowball(HORIZON, ko=1.02, ki=0.93, coupon=0.12, obs=(5, 10, 15)), "continuous"),
    }


_PROBES: dict[str, tuple[Any, str]] | None = None


def probes() -> dict[str, tuple[Any, str]]:
    global _PROBES
    if _PROBES is None:
        _PROBES = _probe_programs()
    return _PROBES


def targets() -> tuple[Target, ...]:
    items: list[Target] = []
    for j, name in enumerate(UNDERLYING_ORDER):
        items.append(Target(f"logret_{name}", "return", j, "continuous"))
        items.append(Target(f"realized_vol_{name}", "volatility", j, "continuous"))
        for probe, (_, kind) in probes().items():
            items.append(Target(f"probe_{probe}_{name}", "payoff_probe", j, kind))
    for q_name in Q_NAMES:
        items.append(Target(f"q_{q_name}", "q_quote", None, "continuous"))
    return tuple(items)


def _probe_payoffs(log_returns: np.ndarray) -> dict[str, np.ndarray]:
    """``log_returns``: (M, HORIZON) for one underlying -> payoff per probe."""

    import torch
    from structurerdsl import Paths, run

    spot = np.ones((log_returns.shape[0], HORIZON + 1))
    spot[:, 1:] = np.exp(np.cumsum(log_returns, axis=1))
    paths = Paths(
        spot={"u": torch.as_tensor(spot, dtype=torch.float64)},
        t_years=torch.arange(HORIZON + 1, dtype=torch.float64) / 252.0,
        rate=0.0,
    )
    result = {}
    for name, (program, _) in probes().items():
        cashflows = run(program, paths).cashflows
        result[name] = cashflows[:, 1:].sum(dim=1).cpu().numpy()
    return result


def target_values(returns: np.ndarray, q: np.ndarray) -> dict[str, np.ndarray]:
    """``returns``: (M, HORIZON, 2); ``q``: (M, 3) -> each target as (M,)."""

    values: dict[str, np.ndarray] = {}
    for j, name in enumerate(UNDERLYING_ORDER):
        series = returns[:, :, j]
        values[f"logret_{name}"] = series.sum(axis=1)
        values[f"realized_vol_{name}"] = np.sqrt(252.0 * np.mean(series**2, axis=1))
        for probe, payoff in _probe_payoffs(series).items():
            values[f"probe_{probe}_{name}"] = payoff
    for k, q_name in enumerate(Q_NAMES):
        values[f"q_{q_name}"] = q[:, k]
    return values


def crps(samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Sample CRPS per row: ``samples`` (O, P), ``observed`` (O,)."""

    ordered = np.sort(samples, axis=1)
    count = ordered.shape[1]
    weights = 2.0 * np.arange(1, count + 1) - count - 1.0
    spread = ordered @ weights / count**2
    return np.mean(np.abs(ordered - observed[:, None]), axis=1) - spread


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastSet:
    """Target samples for one forecaster over the origins of one split."""

    split: str
    origins: tuple[int, ...]
    samples: Mapping[str, np.ndarray]  # target -> (O, P)


def collect_forecasts(
    forecaster: Forecaster,
    panel: MarketPanel,
    split: str,
    *,
    seed: int = 0,
) -> ForecastSet:
    origins = origin_indices(panel, split)
    if not origins:
        raise ValueError(f"panel has no {split} origins")
    returns_all = np.empty((len(origins), N_PATHS, HORIZON, 2))
    q_all = np.empty((len(origins), N_PATHS, len(Q_NAMES)))
    for k, t in enumerate(origins):
        try:
            returns, q = forecaster(public_history(panel, t), query_for(split, t, seed))
            returns_all[k], q_all[k] = validate_forecast(returns, q)
        except ForecastError as exc:
            raise ForecastError(f"origin {k} of {len(origins)}: {exc}") from exc
    flat = target_values(
        returns_all.reshape(-1, HORIZON, 2), q_all.reshape(-1, len(Q_NAMES))
    )
    samples = {name: value.reshape(len(origins), N_PATHS) for name, value in flat.items()}
    return ForecastSet(split, tuple(origins), samples)


_REFERENCE_CACHE: dict[tuple[int, str, int], tuple[MarketPanel, ForecastSet]] = {}


def reference_forecasts(panel: MarketPanel, split: str, *, seed: int = 0) -> ForecastSet:
    key = (id(panel), split, seed)
    cached = _REFERENCE_CACHE.get(key)
    if cached is None or cached[0] is not panel:  # the stored panel pins its id
        cached = (panel, collect_forecasts(reference_forecast, panel, split, seed=seed))
        _REFERENCE_CACHE[key] = cached
    return cached[1]


def _observed(panel: MarketPanel, origins: Sequence[int]) -> dict[str, np.ndarray]:
    returns = np.stack([realized(panel, t)[0] for t in origins])
    q = np.stack([realized(panel, t)[1] for t in origins])
    finite_q = np.where(np.isfinite(q), q, 0.0)
    values = target_values(returns, finite_q)
    for k, q_name in enumerate(Q_NAMES):
        values[f"q_{q_name}"] = q[:, k]  # keep NaN: target unavailable
    return values


MIN_TARGET_OBSERVATIONS = 3


def _aggregate(crps_candidate: Mapping[str, np.ndarray], crps_reference: Mapping[str, np.ndarray],
               rows: np.ndarray, eligible: Sequence[str], *, require_all: bool
               ) -> tuple[float | None, dict[str, float], dict[str, float]]:
    """Skill over a fixed target set; rows may repeat (bootstrap)."""

    target_skill: dict[str, float] = {}
    for name in eligible:
        r = crps_reference[name][rows]
        observed = np.isfinite(r)
        denominator = float(r[observed].sum())
        if not observed.any() or denominator <= 0:
            if require_all:
                return None, {}, {}
            continue
        target_skill[name] = 1.0 - float(crps_candidate[name][rows][observed].sum()) / denominator
    family_skill: dict[str, float] = {}
    for family in sorted({item.family for item in targets()}):
        members = [target_skill[item.name] for item in targets()
                   if item.family == family and item.name in target_skill]
        if members:
            family_skill[family] = float(np.mean(members))
    score = float(np.mean(list(family_skill.values()))) if family_skill else None
    return score, family_skill, target_skill


def _calibration(samples: np.ndarray, observed: np.ndarray, kind: str) -> dict[str, float] | None:
    mask = np.isfinite(observed)
    if mask.sum() < 3:
        return None
    s, y = samples[mask], observed[mask]
    if kind == "binary":
        return {"mean_forecast_probability": float(s.mean()), "realized_frequency": float(y.mean())}
    lower, upper = np.quantile(s, 0.1, axis=1), np.quantile(s, 0.9, axis=1)
    pit = (s <= y[:, None]).mean(axis=1)
    return {
        "central_80_coverage": float(np.mean((y >= lower) & (y <= upper))),
        "pit_mean": float(pit.mean()),
        "pit_below_10pct": float(np.mean(pit < 0.1)),
        "pit_above_90pct": float(np.mean(pit > 0.9)),
    }


def evaluate_forecasts(
    candidate: ForecastSet,
    panel: MarketPanel,
    *,
    reference: ForecastSet | None = None,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Score ``candidate`` against realised outcomes and the reference model."""

    reference = reference or reference_forecasts(panel, candidate.split)
    if reference.origins != candidate.origins:
        raise ValueError("candidate and reference must share origins")
    origins = np.asarray(candidate.origins)
    observed = _observed(panel, candidate.origins)
    crps_c = {item.name: crps(candidate.samples[item.name], np.nan_to_num(observed[item.name])) for item in targets()}
    crps_r = {item.name: crps(reference.samples[item.name], np.nan_to_num(observed[item.name])) for item in targets()}
    for item in targets():
        missing = ~np.isfinite(observed[item.name])
        # Availability depends only on the market; a non-finite candidate
        # score on an observed target is a protocol violation, never a skip.
        if not np.all(np.isfinite(crps_c[item.name][~missing])):
            raise ForecastError(f"non-finite CRPS for {item.name}")
        crps_c[item.name][missing] = np.nan
        crps_r[item.name][missing] = np.nan
    # The scored target set is fixed per split before any resampling.
    eligible = [item.name for item in targets()
                if np.isfinite(observed[item.name]).sum() >= MIN_TARGET_OBSERVATIONS]

    all_rows = np.arange(len(origins))
    score, family_skill, target_skill = _aggregate(crps_c, crps_r, all_rows, eligible, require_all=True)

    # Circular moving-block bootstrap; draws missing an eligible target are
    # redrawn so every draw scores the same composition.
    rng = np.random.default_rng(bootstrap_seed)
    count = len(origins)
    blocks = max(1, int(np.ceil(count / BOOTSTRAP_BLOCK)))
    draws, attempts = [], 0
    while len(draws) < BOOTSTRAP_RESAMPLES and attempts < 5 * BOOTSTRAP_RESAMPLES:
        attempts += 1
        starts = rng.integers(0, count, size=blocks)
        rows = ((starts[:, None] + np.arange(BOOTSTRAP_BLOCK)[None, :]) % count).ravel()[:count]
        value, _, _ = _aggregate(crps_c, crps_r, rows, eligible, require_all=True)
        if value is not None:
            draws.append(value)
    lcb = float(np.quantile(draws, 0.05)) if draws else None

    # Volatility-regime slices use only information available at each origin.
    trailing = np.array([
        np.sqrt(252.0 * np.mean(panel.log_returns[t - 20 : t + 1, 0] ** 2)) for t in origins
    ])
    edges = np.quantile(trailing, [1 / 3, 2 / 3])
    regime = np.digitize(trailing, edges)
    slices = {}
    for label, code in (("low_trailing_vol", 0), ("mid_trailing_vol", 1), ("high_trailing_vol", 2)):
        rows = np.flatnonzero(regime == code)
        value, families, scored = _aggregate(crps_c, crps_r, rows, eligible, require_all=False)
        slices[label] = {"origins": int(rows.size), "score": value, "families": families,
                         "targets_without_observations": sorted(set(eligible) - set(scored))}

    calibration = {}
    for item in targets():
        report = _calibration(candidate.samples[item.name], observed[item.name], item.kind)
        if report is not None:
            calibration[item.name] = report

    return {
        "protocol": PROTOCOL,
        "split": candidate.split,
        "origins": int(len(origins)),
        "score": score,
        "score_lcb_95": lcb,
        "bootstrap_draws": len(draws),
        "scored_targets": eligible,
        "score_definition": "mean over families of mean target CRPS skill vs EWMA-GBM/random-walk reference",
        "family_skill": family_skill,
        "target_skill": target_skill,
        "slices": slices,
        "calibration": calibration,
    }


def evaluate_forecaster(forecaster: Forecaster, panel: MarketPanel, split: str, *, seed: int = 0) -> dict[str, Any]:
    candidate = collect_forecasts(forecaster, panel, split, seed=seed)
    return evaluate_forecasts(candidate, panel, reference=reference_forecasts(panel, split, seed=seed))


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from .market_panel import DEFAULT_OUTPUT_DIR, load_panel

    parser = argparse.ArgumentParser(description="Score the reference world model (sanity check)")
    parser.add_argument("--panel-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args(argv)
    panel = load_panel(args.panel_dir)
    report = evaluate_forecaster(reference_forecast, panel, args.split)
    print(json.dumps({k: report[k] for k in ("split", "origins", "score", "score_lcb_95", "family_skill")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
