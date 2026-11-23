"""Tier-0 feasibility: a small anonymous sequence model against simulator truth.

Trains a model on observed (sign, return) events from a world whose impact
mechanism is known, injects metaorders into the model's rollouts, and compares
the response with the exact counterfactual of the simulator.

Three response paths are measured on the same windows:

* truth    exact counterfactual from the simulator;
* free     the model generates background signs and returns;
* clamped  background signs are the simulator's, the model generates returns.

``free - clamped`` is the error from the model's flow reaction to the injected
orders (the true background flow does not react). ``clamped - truth`` is the
error in the model's price reaction.

Training recipes (``--recipe``):

* anonymous        the model sees (sign, return) only;
* self             one metaorder per training window is flagged as the model's
                   own; injected trades carry the same flag at rollout;
* self-exogenous   as ``self``, but only uninformed metaorders are flagged,
                   the analogue of labelled exogenous flow.

With ``--own-start inside`` only metaorders that begin inside the training
window can be flagged, so no own trade is hidden before the window. With the
default ``any`` a long own metaorder is usually already running when the
window opens, and the decay of its unseen earlier trades is in the data.

``--label-miss`` and ``--label-false`` corrupt the training flags the way an
imperfect identity labeller would: a share of the own trades loses its flag
and stays in the background, or trades of other metaorders gain it. Injected
trades at rollout are always flagged correctly.

The flag is derived from metaorder identity. It is the only identity-derived
input; no other participant is ever labelled.

``--cell`` picks the recurrent cell: ``gru`` (the reference), ``ssm`` (a
diagonal linear state space whose hypothesis class closely approximates the
simulator's fixed-kernel propagator, so a failure of the injection response
is unlikely to come from the representation), or ``lru`` (the same linear
state with a nonlinear readout). The decay timescales are initialised on a
geometric grid from 1 to 1000 trades.

    .venv/bin/python experiments/tier0_feasibility.py --world plain

Outputs go to ``outputs/sim/tier0_feasibility/<tag>/`` (git-ignored simulator
artefacts): ``manifest.json`` holds every parameter and seed needed to rerun,
``results.json`` the measurements. ``--save-model`` also writes the trained
weights to ``model.pt`` so the model can be inspected afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stochopia.common import atomic_write_json  # noqa: E402
from stochopia.measure.impact import (  # noqa: E402
    ImpactLaw,
    antisymmetrise,
    child_impact,
    delta_bootstrap_ci,
    identify_law,
    relaxation,
)
from stochopia.neural.seq import (  # noqa: E402
    ModelSpec,
    make_model,
    n_parameters,
    next_event_loss,
    rollout,
    train,
)
from stochopia.sim.flow import FlowSpec  # noqa: E402
from stochopia.sim.world import (  # noqa: E402
    Day,
    WorldSpec,
    injection_schedule,
    simulate_day,
    true_response,
)
from stochopia.stats import derive_seed  # noqa: E402

WORLDS = {
    "plain": WorldSpec(name="tier0-plain"),
    "informed": WorldSpec(
        name="tier0-informed", flow=FlowSpec(informed_frac=0.3), info_drift=0.5
    ),
}


NO_SELF_PROB = 0.25  # share of training windows with no own trades


# --- Observations and training labels ---


def metaorder_starts(day: Day) -> np.ndarray:
    """For each trade, the index of the first trade of its metaorder; -1 when
    the metaorder began before the day (during burn-in)."""
    n = day.meta_id.size
    _, inverse = np.unique(day.meta_id, return_inverse=True)
    first = np.full(inverse.max() + 1, n)
    np.minimum.at(first, inverse, np.arange(n))
    starts = first[inverse]
    starts[np.isin(day.meta_id, day.pre_meta_id)] = -1
    return starts


def make_own_flags(
    days: Sequence[Day],
    exogenous_only: bool,
    start_inside: bool = False,
    miss: float = 0.0,
    false_share: float = 0.0,
) -> Callable[[int, int, int, np.random.Generator], np.ndarray]:
    """Own-trade column for one training window.

    A random trade in the window is drawn and its whole metaorder is flagged,
    so long metaorders are picked in proportion to their length. With
    ``exogenous_only`` the draw is restricted to uninformed metaorders, with
    ``start_inside`` to metaorders whose first trade lies inside the window.

    Label errors: each own trade loses its flag with probability ``miss``;
    trades of other metaorders gain it so that, before misses, a share
    ``false_share`` of the flagged trades is wrong in expectation.
    """
    starts = [metaorder_starts(d) for d in days] if start_inside else None

    def own_flags(day_index: int, start: int, length: int, rng: np.random.Generator) -> np.ndarray:
        flags = np.zeros(length, dtype=np.float32)
        if rng.random() < NO_SELF_PROB:
            return flags
        day = days[day_index]
        ids = day.meta_id[start : start + length]
        ok = np.ones(length, dtype=bool)
        if exogenous_only:
            ok &= ~day.informed[start : start + length]
        if start_inside:
            ok &= starts[day_index][start : start + length] >= start
        eligible = np.flatnonzero(ok)
        if eligible.size == 0:
            return flags
        own = ids == ids[rng.choice(eligible)]
        flags[own] = 1.0
        if miss > 0.0:
            flags[own & (rng.random(length) < miss)] = 0.0
        if false_share > 0.0:
            others = np.flatnonzero(~own)
            wanted = own.sum() * false_share / (1.0 - false_share)
            picked = rng.random(others.size) < min(1.0, wanted / max(others.size, 1))
            flags[others[picked]] = 1.0
        return flags

    return own_flags


def events(day: Day) -> np.ndarray:
    return np.stack([day.signs.astype(np.float32), day.returns.astype(np.float32)], axis=1)


def sign_autocorr(signs: np.ndarray, lags: Sequence[int] = (1, 10, 100)) -> dict[str, float]:
    """Mean signed lag products over a ``[rows, time]`` array.

    A positive lag without overlapping observations is reported as NaN.
    """
    output = {}
    for lag in lags:
        if lag <= 0:
            raise ValueError("autocorrelation lag must be positive")
        output[str(lag)] = float(np.mean(signs[:, :-lag] * signs[:, lag:])) if lag < signs.shape[1] else float("nan")
    return output


# --- Fitting and auxiliary summaries ---


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with None so unidentified laws can be saved."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarise(name: str, paths: np.ndarray, schedule: np.ndarray, seed: int) -> dict[str, Any]:
    """``paths`` is ``[n_days, schedule.size]`` of antisymmetrised response.

    Relaxation is relative to the peak, so it is reported when a law is
    identified and its 20-event tail exists. Short runs still record the peak.
    """
    impact = child_impact(paths, schedule)
    days, children = impact.shape
    if days < 2 or children < 3:
        peak = float(impact[:, -1].mean())
        peak_se = float(impact[:, -1].std(ddof=1) / np.sqrt(days)) if days > 1 else float("nan")
        law = ImpactLaw(False, "needs at least two days and three children", float("nan"), float("nan"), peak, peak_se)
    else:
        law = identify_law(impact)
    out = {"name": name, **asdict(law)}
    if law.identified:
        lo, hi = delta_bootstrap_ci(impact, seed=seed)
        out["delta_ci"] = [lo, hi]
        if schedule.size - int(np.flatnonzero(schedule)[-1]) >= 20:
            out.update(asdict(relaxation(paths.mean(axis=0), schedule)))
    return out


def initialize_model(args: argparse.Namespace, train_days: Sequence[Day], train_seed: int) -> torch.nn.Module:
    return_scale = float(np.concatenate([d.returns for d in train_days]).std())

    flagged = args.recipe != "anonymous"
    torch.manual_seed(train_seed)  # weight initialisation
    model = make_model(ModelSpec(args.hidden, args.layers, return_scale, flagged, args.cell))
    return model


def fit_model(
    model: torch.nn.Module, args: argparse.Namespace, train_days: Sequence[Day], train_seed: int,
) -> list[float]:
    flagged = model.spec.own_flag
    history = train(
        model,
        [events(d) for d in train_days],
        steps=args.steps,
        batch=args.batch,
        window=args.window,
        lr=args.lr,
        seed=train_seed,
        device=args.device,
        own_flags=(
            make_own_flags(
                train_days,
                args.recipe == "self-exogenous",
                args.own_start == "inside",
                args.label_miss,
                args.label_false,
            )
            if flagged
            else None
        ),
    )
    return history


def evaluate_model(
    model: torch.nn.Module, args: argparse.Namespace, world: WorldSpec, test_days: Sequence[Day],
    *, rollout_seed: int, seed_ci: int, background_model: torch.nn.Module | None = None,
    return_observations: bool = False, rollout_unit_seeds: Sequence[int] | None = None,
    include_impact_summary: bool = True,
) -> dict[str, Any]:
    """Score existing weights and optionally return raw paths in input day order.

    Per-unit rollout seeds keep a day's draws stable across registered subsets.
    Omitting them preserves the original shared stream. The harness can skip
    the auxiliary impact-law fit and use its own registered estimators.
    """
    schedule = injection_schedule(args.children, args.spacing, args.relax)
    n_bg = int((~schedule).sum())
    return_scale = model.spec.return_scale
    flagged = model.spec.own_flag
    n_days = len(test_days)

    # held-out next-event likelihood
    with torch.no_grad():
        probe = np.stack([events(d)[: 4 * args.window] for d in test_days])
        probe[..., 1] /= return_scale
        if flagged:  # held-out likelihood with no own trades
            probe = np.concatenate([probe, np.zeros_like(probe[..., :1])], axis=-1)
        probe_tensor = torch.as_tensor(probe, device=args.device)
        sign_logits = None
        if background_model is not None and background_model is not model:
            sign_logits, _, _ = background_model(probe_tensor[:, :-1, :2])
        heldout = next_event_loss(model, probe_tensor, sign_logits=sign_logits).item()
        if return_observations:
            heldout_by_day = [
                next_event_loss(
                    model, probe_tensor[i : i + 1],
                    sign_logits=None if sign_logits is None else sign_logits[i : i + 1],
                ).item()
                for i in range(n_days)
            ]

    # injection windows: evenly spaced starts with room for context and schedule
    first, last = args.context, world.n_trades - n_bg
    if last < first:
        raise SystemExit("context plus schedule do not fit inside one day")
    starts = np.linspace(first, last, args.windows_per_day).astype(int)
    context = np.stack(
        [events(d)[s - args.context : s] for d in test_days for s in starts]
    )
    background = np.stack(
        [d.signs[s : s + n_bg].astype(np.float32) for d in test_days for s in starts]
    )
    if rollout_unit_seeds is None:
        rng = np.random.default_rng(rollout_seed)
        uniforms = rng.random((context.shape[0], schedule.size))
        normals = rng.standard_normal((context.shape[0], schedule.size))
    else:
        if len(rollout_unit_seeds) != n_days:
            raise ValueError("need one rollout seed per day")
        rngs = [np.random.default_rng(seed) for seed in rollout_unit_seeds]
        shape = (args.windows_per_day, schedule.size)
        uniforms = np.concatenate([rng.random(shape) for rng in rngs])
        normals = np.concatenate([rng.standard_normal(shape) for rng in rngs])

    def model_arm(side: int, signs: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        return rollout(
            model,
            context,
            schedule,
            side,
            uniforms=uniforms,
            normals=normals,
            clamp_signs=signs,
            device=args.device,
        )

    def per_day(x: np.ndarray) -> np.ndarray:
        return x.reshape(n_days, args.windows_per_day, -1).mean(axis=1)

    t1 = time.time()
    # Generate an action-free stream. In the Tier 0 no-response hypothesis,
    # its signs supply the background in both injected arms.
    idle_s, idle_r = rollout(
        model if background_model is None else background_model,
        context,
        np.zeros(schedule.size, dtype=bool),
        +1,
        uniforms=uniforms,
        normals=normals,
        device=args.device,
    )
    if background_model is not None and background_model is not model:
        _, idle_r = rollout(
            model, context, np.zeros_like(schedule), +1,
            uniforms=uniforms, normals=normals, clamp_signs=idle_s, device=args.device,
        )
    proposed_background = None if background_model is None else idle_s[:, :n_bg]
    free_buy_s, free_buy_r = model_arm(+1, proposed_background)
    free_sell_s, free_sell_r = model_arm(-1, proposed_background)
    _, clamp_buy_r = model_arm(+1, background)
    _, clamp_sell_r = model_arm(-1, background)
    rollout_time = time.time() - t1

    free = per_day(antisymmetrise(free_buy_r.cumsum(axis=1), free_sell_r.cumsum(axis=1)))
    clamped = per_day(antisymmetrise(clamp_buy_r.cumsum(axis=1), clamp_sell_r.cumsum(axis=1)))
    truth = np.stack(
        [
            np.mean(
                [
                    antisymmetrise(
                        true_response(world, d, int(s), schedule, +1),
                        true_response(world, d, int(s), schedule, -1),
                    )
                    for s in starts
                ],
                axis=0,
            )
            for d in test_days
        ]
    )

    # Flow reaction of the model: background sign shift caused by the
    # injection. "During" covers background trades before the last child, the
    # same boundary the relaxation summary uses for the peak.
    induced = per_day(antisymmetrise(free_buy_s, free_sell_s)[:, ~schedule])
    exec_bg = (args.children - 1) * args.spacing
    induced_mean = induced.mean(axis=0)

    last_child = int(np.flatnonzero(schedule)[-1])
    mean = {"truth": truth.mean(0), "free": free.mean(0), "clamped": clamped.mean(0)}

    def decomposition(t: int) -> dict[str, float]:
        return {
            "truth": float(mean["truth"][t]),
            "free": float(mean["free"][t]),
            "clamped": float(mean["clamped"][t]),
            "flow_reaction_error": float(mean["free"][t] - mean["clamped"][t]),
            "price_reaction_error": float(mean["clamped"][t] - mean["truth"][t]),
        }

    def flow_mean(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else float("nan")

    def standard_error(values: np.ndarray) -> float:
        return float(values.std(ddof=1) / np.sqrt(n_days)) if n_days > 1 else float("nan")

    idle_drift = per_day(idle_r.cumsum(axis=1))[:, -1]
    laws = []
    if include_impact_summary:
        for arm, paths in (("truth", truth), ("free", free), ("clamped", clamped)):
            laws.append(summarise(arm, paths, schedule, seed_ci))
    result: dict[str, Any] = {
        "n_parameters": n_parameters(model),
        "heldout_loss": heldout,
        "rollout_seconds": rollout_time,
        "sign_autocorr": {
            "truth": sign_autocorr(background),
            "model_no_injection": sign_autocorr(idle_s),
        },
        "model_free_drift": {
            "mean": float(idle_drift.mean()),
            "se": standard_error(idle_drift),
        },
        "laws": laws,
        "at_peak": decomposition(last_child),
        "at_end": decomposition(schedule.size - 1),
        "induced_background_sign": {
            "during_execution": flow_mean(induced_mean[:exec_bg]),
            "after_execution": flow_mean(induced_mean[exec_bg:]),
            "se_during": standard_error(induced[:, :exec_bg].mean(axis=1)) if exec_bg else float("nan"),
            "se_after": standard_error(induced[:, exec_bg:].mean(axis=1)) if exec_bg < n_bg else float("nan"),
        },
        "mean_paths": {k: [float(v) for v in path] for k, path in mean.items()},
    }
    if return_observations:
        result["schedule"] = schedule.tolist()
        result["day_observations"] = [
            {
                "paths": {"free": free[i].tolist(), "clamped": clamped[i].tolist(), "truth": truth[i].tolist()},
                "heldout_loss": heldout_by_day[i],
                "no_injection_drift": float(idle_drift[i]),
                "induced_background_sign": {
                    "during_execution": flow_mean(induced[i, :exec_bg]),
                    "after_execution": flow_mean(induced[i, exec_bg:]),
                },
            }
            for i in range(n_days)
        ]
    return result


# --- Standalone feasibility CLI ---


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", choices=sorted(WORLDS), default="plain")
    ap.add_argument(
        "--recipe", choices=["anonymous", "self", "self-exogenous"], default="anonymous"
    )
    ap.add_argument("--own-start", choices=["any", "inside"], default="any")
    ap.add_argument("--label-miss", type=float, default=0.0)
    ap.add_argument("--label-false", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument(
        "--cell",
        choices=["gru", "ssm", "lru"],
        default="gru",
        help="recurrent cell: gru, or a diagonal linear state space with a "
        "linear (ssm) or GELU (lru) readout; ssm closely approximates the "
        "simulator's propagator",
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--train-days", type=int, default=150)
    ap.add_argument("--test-days", type=int, default=20)
    ap.add_argument("--windows-per-day", type=int, default=64)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--children", type=int, default=50)
    ap.add_argument("--spacing", type=int, default=9)
    ap.add_argument("--relax", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save-model", action="store_true")
    args = ap.parse_args()

    world = WORLDS[args.world]
    schedule = injection_schedule(args.children, args.spacing, args.relax)
    if args.context > world.n_trades - int((~schedule).sum()):
        raise SystemExit("context plus schedule do not fit inside one day")
    flagged = args.recipe != "anonymous"
    if args.own_start == "inside" and not flagged:
        raise SystemExit("--own-start inside needs a recipe with an own flag")
    if not (0.0 <= args.label_miss < 1.0 and 0.0 <= args.label_false < 1.0):
        raise SystemExit("--label-miss and --label-false must lie in [0, 1)")
    if (args.label_miss or args.label_false) and not flagged:
        raise SystemExit("label errors need a recipe with an own flag")
    recipe = f"-{args.recipe}" if flagged else ""
    recipe += "-fresh" if args.own_start == "inside" else ""
    recipe += f"-miss{100 * args.label_miss:g}" if args.label_miss else ""
    recipe += f"-false{100 * args.label_false:g}" if args.label_false else ""
    if args.cell == "gru":
        shape = f"h{args.hidden}x{args.layers}"
    else:  # the linear state space has no layers
        shape = f"h{args.hidden}-{args.cell}"
    tag = args.tag or f"{args.world}{recipe}-{shape}-s{args.seed}"
    out_dir = ROOT / "outputs" / "sim" / "tier0_feasibility" / tag
    if (out_dir / "manifest.json").exists():
        # the default tag does not encode every setting
        before = json.loads((out_dir / "manifest.json").read_text())["args"]
        changed = sorted(
            k for k, v in vars(args).items()
            if k not in ("tag", "device") and k in before and before[k] != v
        )
        if changed:
            raise SystemExit(f"{out_dir} holds a run with different {changed}; pass --tag")

    t0 = time.time()
    train_days = [simulate_day(world, f"train-{args.seed}", i) for i in range(args.train_days)]
    test_days = [simulate_day(world, f"test-{args.seed}", i) for i in range(args.test_days)]
    train_seed = derive_seed("tier0_feasibility", "train", tag)
    model = initialize_model(args, train_days, train_seed)
    seed_ci = derive_seed("tier0_feasibility", "ci", tag)
    manifest = {
        "experiment": "tier0_feasibility",
        "tag": tag,
        "recipe": args.recipe,
        "no_self_prob": NO_SELF_PROB if flagged else None,
        "world": world.manifest(),
        "model": asdict(model.spec),
        "args": vars(args),
        "seeds": {
            "train_days_namespace": f"train-{args.seed}",
            "test_days_namespace": f"test-{args.seed}",
            "train": train_seed,
            "rollout": derive_seed("tier0_feasibility", "rollout", tag),
            "delta_ci": seed_ci,
        },
        "schedule": {
            "children": args.children,
            "spacing": args.spacing,
            "relax": args.relax,
            "participation": 1.0 / (1 + args.spacing),
        },
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    # written before training so that a second job with the same tag sees it
    atomic_write_json(out_dir / "manifest.json", manifest)
    history = fit_model(model, args, train_days, train_seed)
    train_time = time.time() - t0
    if args.save_model:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out_dir / "model.pt")

    results = evaluate_model(
        model, args, world, test_days,
        rollout_seed=manifest["seeds"]["rollout"], seed_ci=seed_ci,
    )
    results.update(
        train_loss_first_last=[history[0], history[-1]], train_seconds=train_time
    )
    atomic_write_json(out_dir / "results.json", json_safe(results))

    print(f"[{tag}] params={results['n_parameters']:,} device={args.device} "
          f"train={train_time:.0f}s rollout={results['rollout_seconds']:.0f}s heldout_nll={results['heldout_loss']:.4f}")
    print("sign autocorr  truth:", results["sign_autocorr"]["truth"],
          " model, no injection:", results["sign_autocorr"]["model_no_injection"])
    drift = results["model_free_drift"]
    print(f"  model free drift over the horizon: {drift['mean']:+.2f}±{drift['se']:.2f}")
    for law in results["laws"]:
        line = (f"  {law['name']:8s} identified={law['identified']!s:5s} "
                f"peak={law['peak']:.2f}±{law['peak_se']:.2f}")
        if law["identified"]:
            ci = law["delta_ci"]
            line += f" delta={law['delta']:.2f} CI[{ci[0]:.2f},{ci[1]:.2f}]"
            if "end_over_peak" in law:
                line += f" end/peak={law['end_over_peak']:.2f} slope={law['slope']:+.3f}"
        else:
            line += f" ({law['reason']})"
        print(line)
    for key in ("at_peak", "at_end"):
        d = results[key]
        print(f"  {key}: truth={d['truth']:.2f} free={d['free']:.2f} clamped={d['clamped']:.2f} "
              f"flow_err={d['flow_reaction_error']:+.2f} price_err={d['price_reaction_error']:+.2f}")
    ind = results["induced_background_sign"]
    print(f"  induced background sign: during={ind['during_execution']:+.4f}±{ind['se_during']:.4f} "
          f"after={ind['after_execution']:+.4f}±{ind['se_after']:.4f}")
    print(f"  wrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
