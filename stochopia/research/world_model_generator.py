"""P-side scenario generator backed by a registered world model.

Given an as-of session of the real panel, the generator asks a frozen world
model (``stochopia.research.world_models``) for joint daily paths of CSI500 /
CSI1000 and the Q-side quotes, chaining 21-session steps for longer horizons.
The model sees exactly the fixed windows of the forecast protocol: no dates,
no levels, nothing after the as-of session.  The output is a P-measure
scenario set for evaluating structures (for example through StructurerDSL);
it is not a pricing measure and never replaces Q-side valuation.

Model calls always use ``fp.N_PATHS`` samples, the sample size the model was
scored at: r02s01 standardises its shocks across the sample axis, so its law
depends on the requested sample count.  The first step subsamples that call;
every continuation step makes one such call per path and keeps one draw.
With ``continuation_samples`` below ``fp.N_PATHS`` (a speed option) the
continuation law is slightly thinner-tailed than the scored law (measured at
32: 21-day sd about 2% lower, kurtosis 3.8 vs 4.1).

Chaining: after each step every path rolls its own windows forward with its
simulated returns; ``rate_3m`` is held flat.  A Q window is extended only if
it already holds at least ``MIN_QUOTES`` real or previously extended quotes,
by noiseless linear interpolation (log space for IV) to the step's sampled
quote; otherwise it stays NaN.  The noiseless segments shrink the dispersion
r02s01 estimates for later-step Q quotes (AR(1) residual sd about 25% lower
by step 3); P paths are unaffected because r02s01's returns never read the Q
windows.  ``manifest["q_supported"]`` flags Q series whose as-of window had
too few quotes: their ``q`` samples are model defaults, not data-driven.
Paths are exchangeable draws, not independent ones (each model call pins the
cross-sample mean of its shocks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..common import sha256_text
from . import forecast_protocol as fp
from .market_panel import UNDERLYING_ORDER, MarketPanel
from .world_models import get_world_model

GENERATOR_SCHEMA = "stochopia.world-model-scenario.v1"
STEP = fp.HORIZON
MIN_QUOTES = 15  # r02s01 falls back to default quotes below this many


def _seed(*parts: Any) -> int:
    return int(sha256_text("|".join(str(part) for part in (GENERATOR_SCHEMA, *parts)))[:15], 16)


def _check(returns: np.ndarray, q: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    returns, q = np.asarray(returns, dtype=float), np.asarray(q, dtype=float)
    if returns.shape != (n, STEP, 2) or q.shape != (n, len(fp.Q_NAMES)):
        raise fp.ForecastError(f"world model returned shapes {returns.shape}, {q.shape}")
    if not (np.all(np.isfinite(returns)) and np.all(np.isfinite(q))):
        raise fp.ForecastError("world model returned non-finite samples")
    if (np.any(np.abs(returns) > fp.MAX_ABS_DAILY_RETURN) or np.any(q[:, 0] <= 0)
            or np.any(q[:, 0] > fp.MAX_IV) or np.any(np.abs(q[:, 1:]) >= fp.MAX_ABS_CARRY)):
        raise fp.ForecastError("world model returned out-of-range samples")
    return returns, q


def _extend_quote(window: np.ndarray, target: float, log_space: bool) -> np.ndarray:
    """Roll a Q window forward one step, interpolating to ``target``.

    Windows with fewer than ``MIN_QUOTES`` finite values are extended with NaN
    so that a model's default quote is never written into the history.
    """

    finite = window[np.isfinite(window)]
    if finite.size < MIN_QUOTES:
        added = np.full(STEP, np.nan)
    else:
        start = finite[-1]
        weights = np.arange(1, STEP + 1) / STEP
        if log_space:
            added = np.exp(np.log(start) + weights * (np.log(target) - np.log(start)))
        else:
            added = start + weights * (target - start)
    return np.concatenate([window[STEP:], added])


@dataclass(frozen=True)
class WorldModelScenario:
    model: str
    as_of: str
    returns: np.ndarray  # (n, horizon_days, 2) daily log returns
    q: np.ndarray  # (n, steps, 3) quotes at the end of each 21-session step
    rate_3m: float
    manifest: Mapping[str, Any]

    @property
    def relative_spot(self) -> np.ndarray:
        """``(n, horizon_days + 1, 2)``, 1.0 at the as-of session."""

        spot = np.ones((self.returns.shape[0], self.returns.shape[1] + 1, 2))
        spot[:, 1:] = np.exp(np.cumsum(self.returns, axis=1))
        return spot

    def dsl_paths(self, underlying: str, *, name: str = "u", rate: float = 0.0):
        """StructurerDSL ``Paths`` for one index (levels relative to the as-of close).

        ``rate`` defaults to 0 so payoffs are undiscounted.  Passing a rate
        makes ``structurerdsl.price`` return a discounted *P* expectation,
        which is not a price: pricing must use Q-side inputs.
        """

        import torch
        from structurerdsl import Paths

        column = UNDERLYING_ORDER.index(underlying)
        steps = self.returns.shape[1]
        return Paths(
            spot={name: torch.as_tensor(self.relative_spot[:, :, column], dtype=torch.float64)},
            t_years=torch.arange(steps + 1, dtype=torch.float64) / 252.0,
            rate=float(rate),
        )


def generate_scenario(
    panel: MarketPanel,
    as_of: int | str,
    *,
    model: str = "r02s01",
    n_paths: int = fp.N_PATHS,
    horizon_days: int = 63,
    seed: int = 0,
    continuation_samples: int = fp.N_PATHS,
) -> WorldModelScenario:
    if horizon_days <= 0 or horizon_days % STEP:
        raise ValueError(f"horizon_days must be a positive multiple of {STEP}")
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    if continuation_samples < 2:
        raise ValueError("continuation_samples must be at least 2")
    t = panel.dates.index(as_of) if isinstance(as_of, str) else int(as_of)
    entry = get_world_model(model)
    forecast = entry.forecast()
    history = fp.public_history(panel, t)  # raises without full windows
    day = panel.dates[t]  # seeds follow the date, not the panel index
    calls = max(1, -(-n_paths // fp.N_PATHS))
    blocks = []
    for call in range(calls):
        query = {**fp.query_for("scenario", t), "n_paths": fp.N_PATHS, "seed": _seed(model, day, seed, 0, call)}
        blocks.append(_check(*forecast(history, query), fp.N_PATHS))
    first_returns = np.concatenate([block[0] for block in blocks])[:n_paths]
    first_q = np.concatenate([block[1] for block in blocks])[:n_paths]

    steps = horizon_days // STEP
    returns = np.empty((n_paths, horizon_days, 2))
    quotes = np.empty((n_paths, steps, len(fp.Q_NAMES)))
    returns[:, :STEP], quotes[:, 0] = first_returns, first_q
    if steps > 1:
        windows = {
            "returns": np.repeat(history["returns"][None], n_paths, axis=0),
            "rate_3m": np.repeat(history["rate_3m"][None], n_paths, axis=0),
            "carry": np.repeat(history["carry"][None], n_paths, axis=0),
            "iv_1m": np.repeat(history["iv_1m"][None], n_paths, axis=0),
        }
        for step in range(1, steps):
            previous = slice((step - 1) * STEP, step * STEP)
            for i in range(n_paths):
                windows["returns"][i] = np.concatenate([windows["returns"][i][STEP:], returns[i, previous]])
                windows["rate_3m"][i] = np.concatenate(
                    [windows["rate_3m"][i][STEP:], np.full(STEP, windows["rate_3m"][i][-1])]
                )
                windows["iv_1m"][i] = _extend_quote(windows["iv_1m"][i], quotes[i, step - 1, 0], True)
                for j in range(2):
                    windows["carry"][i][:, j] = _extend_quote(
                        windows["carry"][i][:, j], quotes[i, step - 1, 1 + j], False
                    )
                local = {key: value[i] for key, value in windows.items()}
                local_query = {**fp.query_for("scenario", t), "n_paths": continuation_samples,
                               "seed": _seed(model, day, seed, step, i)}
                sample_returns, sample_q = _check(*forecast(local, local_query), continuation_samples)
                # Samples are exchangeable, so keeping the first is an unbiased draw.
                returns[i, step * STEP : (step + 1) * STEP] = sample_returns[0]
                quotes[i, step] = sample_q[0]

    q_windows = (history["iv_1m"], history["carry"][:, 0], history["carry"][:, 1])
    manifest = {
        "schema": GENERATOR_SCHEMA,
        "model": model,
        "model_provenance": dict(entry.provenance),
        "model_scores": dict(entry.scores),
        "generator_gates": dict(entry.generator_gates),
        "development_only": entry.development_only,
        "as_of": day,
        "horizon_days": horizon_days,
        "n_paths": n_paths,
        "seed": seed,
        "seed_basis": "as_of date string",
        "measure": "P (physical scenario); not a pricing measure",
        "chaining": (f"{STEP}-session steps; model called with {fp.N_PATHS} samples for the first step and "
                     f"{continuation_samples} per path per continuation, one draw kept"),
        "q_supported": {name: bool(np.isfinite(values).sum() >= MIN_QUOTES)
                        for name, values in zip(fp.Q_NAMES, q_windows)},
        "panel_schema": panel.manifest.get("schema"),
    }
    return WorldModelScenario(model, day, returns, quotes, float(panel.rate_3m[t]), manifest)


# --------------------------------------------------------------------------
# Demonstration: a 3-month snowball under a world model, and walk-forward
# calibration of its knock-in / knock-out probabilities against reality.
# --------------------------------------------------------------------------

def _snowball_programs(horizon_days: int):
    from structurerdsl import Program, catalog

    observations = tuple(range(STEP, horizon_days, STEP))
    note = catalog.snowball(horizon_days, ko=1.03, ki=0.85, coupon=0.15, obs=observations)
    knock_in = Program.from_sexpr(
        f"(contract (meta (underlyings u) (maturity {horizon_days}) (schedules)) "
        "(premium (pay (touched (spot u) 0.85 down daily) maturity)))"
    )
    return note, knock_in


def snowball_events(spot: np.ndarray, horizon_days: int) -> dict[str, np.ndarray]:
    """``spot``: (M, horizon_days + 1) relative levels -> per-path KO / KI / payoff."""

    import torch
    from structurerdsl import Paths, run

    note, knock_in = _snowball_programs(horizon_days)
    paths = Paths(spot={"u": torch.as_tensor(spot, dtype=torch.float64)},
                  t_years=torch.arange(horizon_days + 1, dtype=torch.float64) / 252.0, rate=0.0)
    result = run(note, paths)
    return {
        "knock_out": result.terminated_early.cpu().numpy().astype(float),
        "knock_in": run(knock_in, paths).cashflows[:, 1:].sum(dim=1).cpu().numpy(),
        "payoff": result.cashflows[:, 1:].sum(dim=1).cpu().numpy(),
    }


def snowball_calibration(panel: MarketPanel, split: str, *, models=("ewma_gbm_reference", "r02s01"),
                         underlying: str = "CSI1000", horizon_days: int = 63, n_paths: int = 256,
                         continuation_samples: int = 32, bootstrap_draws: int = 2000) -> dict[str, Any]:
    """Brier score of forecast KO / KI probabilities vs realised events at each origin.

    Origins are 21 sessions apart while outcomes span ``horizon_days``, so
    neighbouring windows overlap; the paired difference of the last two
    models gets a circular block-bootstrap 90% interval (block 3).
    ``knock_in`` means "touched 0.85 at any time", also after a knock-out.
    """

    column = UNDERLYING_ORDER.index(underlying)
    origins = [t for t in fp.origin_indices(panel, split) if t + horizon_days < len(panel.dates)]
    realised = np.stack([panel.close[t : t + horizon_days + 1, column] / panel.close[t, column] for t in origins])
    actual = snowball_events(realised, horizon_days)
    report: dict[str, Any] = {"split": split, "underlying": underlying, "origins": len(origins),
                              "horizon_days": horizon_days,
                              "realised_frequency": {k: float(actual[k].mean()) for k in ("knock_out", "knock_in")},
                              "models": {}}
    for model in models:
        probabilities = {"knock_out": [], "knock_in": []}
        for t in origins:
            scenario = generate_scenario(panel, t, model=model, n_paths=n_paths, horizon_days=horizon_days,
                                         continuation_samples=continuation_samples)
            events = snowball_events(scenario.relative_spot[:, :, column], horizon_days)
            for key in probabilities:
                probabilities[key].append(float(events[key].mean()))
        report["models"][model] = {
            key: {"mean_forecast": float(np.mean(values)),
                  "brier": float(np.mean((np.asarray(values) - actual[key]) ** 2)),
                  "_per_origin": ((np.asarray(values) - actual[key]) ** 2).tolist()}
            for key, values in probabilities.items()
        }
    if len(models) >= 2:
        base, other = models[-2], models[-1]
        rng = np.random.default_rng(0)
        count, block = len(origins), 3
        report["paired_difference"] = {"definition": f"brier({other}) - brier({base})"}
        for key in ("knock_out", "knock_in"):
            diff = (np.asarray(report["models"][other][key]["_per_origin"])
                    - np.asarray(report["models"][base][key]["_per_origin"]))
            draws = []
            for _ in range(bootstrap_draws):
                starts = rng.integers(0, count, size=-(-count // block))
                rows = ((starts[:, None] + np.arange(block)[None, :]) % count).ravel()[:count]
                draws.append(diff[rows].mean())
            report["paired_difference"][key] = {
                "mean": float(diff.mean()),
                "ci90": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
            }
    for model in models:
        for key in ("knock_out", "knock_in"):
            report["models"][model][key].pop("_per_origin")
    report["continuation_samples"] = continuation_samples
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from .market_panel import DEFAULT_OUTPUT_DIR, load_panel

    parser = argparse.ArgumentParser(description="World-model scenario demo: 3-month snowball")
    parser.add_argument("--panel-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="r02s01")
    args = parser.parse_args(argv)
    panel = load_panel(args.panel_dir)
    as_of = panel.dates[-1]
    summary = {}
    for model in ("ewma_gbm_reference", args.model):
        runs = []
        for seed in range(8):  # 256 paths per seed; small probabilities need several seeds
            scenario = generate_scenario(panel, as_of, model=model, horizon_days=63, seed=seed)
            events = snowball_events(scenario.relative_spot[:, :, 1], 63)
            runs.append({key: float(value.mean()) for key, value in events.items()})
        summary[model] = {key: {"mean": round(float(np.mean([run[key] for run in runs])), 4),
                                "sd_across_seeds": round(float(np.std([run[key] for run in runs])), 4)}
                          for key in runs[0]}
    calibration = {split: snowball_calibration(panel, split, models=("ewma_gbm_reference", args.model))
                   for split in ("dev", "test")}
    print(json.dumps({"as_of": as_of, "snowball_csi1000_3m": summary, "calibration": calibration},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
