"""Trade-level world with a known impact mechanism and exact counterfactuals.

Price increments are the sum of three parts:

* impact: the linear propagator of :mod:`stochopia.sim.impact` applied to signs;
* information: every trade of an *informed* metaorder coincides with a
  fundamental move of ``info_drift`` in its direction. The move is not caused
  by the trade, so it is absent from the response to an exogenous order;
* noise: i.i.d. Gaussian.

Background flow is exogenous: it does not react to prices or to injected
orders. The true response to an injected metaorder is therefore the paired
difference between two runs that share flow, information and noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..stats import derive_seed
from .flow import FlowSpec, simulate_flow
from .impact import KernelSpec, propagate


@dataclass(frozen=True)
class WorldSpec:
    name: str = "tier0-propagator"
    flow: FlowSpec = field(default_factory=FlowSpec)
    kernel: KernelSpec = field(default_factory=KernelSpec)
    noise_std: float = 2.0
    info_drift: float = 0.0
    n_trades: int = 20000
    burn_in: int = 2000

    def manifest(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Day:
    """One independent episode. Only ``signs`` and ``returns`` are observable."""

    seed: int
    signs: np.ndarray  # int8 [n_trades]
    returns: np.ndarray  # float64 [n_trades], price change at each trade
    meta_id: np.ndarray  # verification only
    informed: np.ndarray  # verification only
    pre_signs: np.ndarray  # burn-in signs preceding ``signs`` (sets kernel state)
    pre_meta_id: np.ndarray  # verification only; metaorder labels of the burn-in


def simulate_day(spec: WorldSpec, namespace: str, day_index: int) -> Day:
    seed = derive_seed("stochopia.sim.world", spec.name, namespace, day_index)
    rng = np.random.default_rng(seed)
    total = spec.burn_in + spec.n_trades
    flow = simulate_flow(spec.flow, total, rng)
    impact = propagate(spec.kernel, flow.signs)
    info = spec.info_drift * flow.signs * flow.informed
    noise = rng.normal(0.0, spec.noise_std, size=total)
    returns = np.diff(impact, prepend=0.0) + info + noise
    b = spec.burn_in
    return Day(
        seed=seed,
        signs=flow.signs[b:],
        returns=returns[b:],
        meta_id=flow.meta_id[b:],
        informed=flow.informed[b:],
        pre_signs=flow.signs[:b],
        pre_meta_id=flow.meta_id[:b],
    )


def injection_schedule(n_children: int, spacing: int, n_relax: int) -> np.ndarray:
    """Boolean mask over the merged event clock; True marks an injected child.

    Each child is followed by ``spacing`` background trades, so participation
    during execution is ``1 / (1 + spacing)``. ``n_relax`` background trades
    follow the last execution block.
    """
    if n_children < 1 or spacing < 0 or n_relax < 0:
        raise ValueError("need n_children >= 1, spacing >= 0, n_relax >= 0")
    mask = np.zeros(n_children * (spacing + 1) + n_relax, dtype=bool)
    mask[np.arange(n_children) * (spacing + 1)] = True
    return mask


def true_response(
    spec: WorldSpec, day: Day, start: int, schedule: np.ndarray, side: int
) -> np.ndarray:
    """Price displacement caused by an uninformed injected metaorder.

    ``start`` indexes ``day.signs``; background trades from ``start`` onward
    fill the non-injected positions of ``schedule``. Entry ``t`` is the price
    with the injection minus the price without it, right after merged event
    ``t``. Information and noise are common to both runs and cancel.

    The kernel decays in trade time, so each injected child also ages the
    impact of earlier trades by one step. That term depends on the day but
    not on ``side``: it cancels in the antisymmetrised response
    ``0.5 * (buy - sell)``, which equals the sum of kernel values over the
    injected children.
    """
    if side not in (-1, 1):
        raise ValueError("side must be +1 or -1")
    schedule = np.asarray(schedule, dtype=bool)
    n_bg = int((~schedule).sum())
    if start < 0 or start + n_bg > day.signs.size:
        raise ValueError("injection window does not fit inside the day")
    history = np.concatenate([day.pre_signs, day.signs[:start]]).astype(float)
    background = day.signs[start : start + n_bg].astype(float)
    merged = np.empty(schedule.size)
    merged[schedule] = side
    merged[~schedule] = background
    offset = history.size
    with_injection = propagate(spec.kernel, np.concatenate([history, merged]))[offset:]
    baseline = propagate(spec.kernel, np.concatenate([history, background]))
    # Baseline price after merged event t = price after the background trades
    # seen so far; before any of them it is the price at the end of history.
    seen = np.cumsum(~schedule)
    base_before = baseline[offset - 1] if offset > 0 else 0.0
    padded = np.concatenate([[base_before], baseline[offset:]])
    return with_injection - padded[seen]
