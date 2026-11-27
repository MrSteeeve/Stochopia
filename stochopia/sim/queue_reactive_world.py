"""Small queue-reactive point process with matched physical intervention tapes.

The clock is exponential-kernel Hawkes (integrated-hazard inversion), and the
mark probabilities react to the current JAX-LOB queue. Own injections change the
book, not the exogenous clock or latent sign process directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stochopia.harness.lob_rollout import _BookTape, _Order

GRID_MS = np.arange(0, 1601, 50, dtype=np.int64)
CHILD_TIMES_MS = (100, 300, 500, 700)
CHILD_SHARES = 100
CONTEXT_LEN = 128
EVENTS_PER_SESSION = 256
MAX_EVENTS = 512
INITIAL_TICK = 10_000


@dataclass(frozen=True)
class QueueWorldSpec:
    name: str
    q_scale: int
    cancel_probability: float
    slow_persistence: float
    fast_persistence: float
    alpha_hawk: float = 0.0
    mean_dt_ms: float = 40.0
    beta_hawk: float = 0.25


def standard_worlds() -> dict[str, QueueWorldSpec]:
    """Five independent recipes; W5_thin_poisson aliases W3_thin_short."""
    result = {}
    for number, thickness, memory in (
        (1, "thick", "short"), (2, "thick", "long"),
        (3, "thin", "short"), (4, "thin", "long"),
    ):
        name = f"W{number}_{thickness}_{memory}"
        q, cancel = (300, 0.12) if thickness == "thick" else (100, 0.18)
        slow, fast = (0.97, 0.80) if memory == "long" else (0.5, 0.5)
        result[name] = QueueWorldSpec(name, q, cancel, slow, fast)
    result["W6_thin_hawkes"] = QueueWorldSpec("W6_thin_hawkes", 100, 0.18, 0.5, 0.5, 0.70)
    return result


def initial_book(spec: QueueWorldSpec) -> tuple[np.ndarray, np.ndarray]:
    """Ten levels each side, relative tick spacing exactly one."""
    levels = np.arange(10, dtype=np.int64)
    ticks = np.stack((INITIAL_TICK - levels, INITIAL_TICK + 1 + levels))
    return ticks, np.full_like(ticks, spec.q_scale)


def clock_tape(
    spec: QueueWorldSpec, uniforms: np.ndarray, excitation: np.ndarray, phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact continuous Hawkes clock up to numerical bisection tolerance.

    One unit-exponential integrated-hazard threshold per event; baseline
    intensity (1-alpha)/mean_dt keeps stationary mean intensity fixed. Round
    cumulative arrivals down to millisecond bins, retaining simultaneous
    events. Return integer times and excitation/phase *before* each event
    waiting interval for conditioning the known-generator counterfactual.
    """
    count, steps = uniforms.shape
    elapsed = phase.copy()
    intensity = excitation.copy()
    times = np.empty((count, steps), dtype=np.int64)
    pre_excitation = np.empty((count, steps))
    pre_phase = np.empty((count, steps))
    mu = (1.0 - spec.alpha_hawk) / spec.mean_dt_ms
    beta = spec.beta_hawk
    if not 0 <= spec.alpha_hawk < 1 or mu <= 0 or beta <= 0:
        raise ValueError("Hawkes clock requires positive baseline, decay and subcritical excitation")
    for event in range(steps):
        pre_excitation[:, event] = intensity
        pre_phase[:, event] = elapsed % 1.0
        threshold = -np.log1p(-uniforms[:, event])
        if spec.alpha_hawk == 0:
            wait = threshold / mu
        else:
            lower = np.zeros(count)
            upper = threshold / mu
            # Analytic integrated hazard: mu*t + z/beta * (1-exp(-beta*t)).
            for _ in range(40):
                mid = (lower + upper) / 2
                hazard = mu * mid - intensity / beta * np.expm1(-beta * mid)
                below = hazard < threshold
                lower = np.where(below, mid, lower)
                upper = np.where(below, upper, mid)
            wait = (lower + upper) / 2
        elapsed += wait
        intensity = intensity * np.exp(-beta * wait) + spec.alpha_hawk * beta
        times[:, event] = np.floor(elapsed).astype(np.int64)
    return times, pre_excitation, pre_phase


def _place(
    path: _BookTape, spec: QueueWorldSpec, side: int, levels: int, now: int, mark: float,
) -> _Order:
    direction = 1 - 2 * side
    touch = int(path.prices[side, 0])
    opposite = int(path.prices[1 - side, 0])
    current = int(path.shares[side, 0])
    if levels == 0:
        return _Order(now, side, 0, spec.q_scale, max(1, opposite - direction))
    if levels < 5:
        price = int(path.prices[side, levels - 1]) - direction
    elif opposite and (opposite - touch) * direction > 1:
        price = touch + direction
    elif current < spec.q_scale:
        return _Order(now, side, 0, spec.q_scale - current, touch)
    else:
        price = touch - direction * (1 + int(9 * mark))
    return _Order(now, side, (touch - price) * direction, spec.q_scale, max(1, price))


def reactive_order(
    path: _BookTape, spec: QueueWorldSpec, signs: np.ndarray, now: int, marks: np.ndarray,
) -> _Order:
    """One state-dependent mark, with previsible quote and trade-clock sign state.

    marks are independently uniform coordinates for event kind, side, slow
    and fast Markov flips, mixture choice and passive/cancel level. Both latent
    sign components update only on aggressive events; the slow component's
    weight is 3/4, giving two distinct persistence scales in long-memory worlds.
    """
    levels = (path.shares[:, :10] > 0).sum(axis=1)
    depth = path.shares[:, 0]
    urgent = levels < 5
    if spec.q_scale > 100:
        urgent = urgent | (depth < spec.q_scale / 2)
    side = int(marks[1] >= 0.5)
    if urgent.any():
        side = int(np.argmin(np.where(urgent, levels * spec.q_scale + depth, np.inf)))
        return _place(path, spec, side, int(levels[side]), now, marks[5])
    if marks[0] < 0.52:
        if marks[2] > spec.slow_persistence:
            signs[0] = -signs[0]
        if marks[3] > spec.fast_persistence:
            signs[1] = -signs[1]
        side = int(signs[0 if marks[4] < 0.75 else 1] < 0)
        # Real fully executed marketable limit order; never creates residue.
        target = 1 - side
        return _Order(now, 2 + side, 0, min(100, int(depth[target])), int(path.prices[target, 0]))
    if marks[0] < 0.52 + spec.cancel_probability:
        rank = int(3 * marks[5])
        return _Order(now, 4 + side, rank, min(100, int(path.shares[side, rank])),
                      int(path.prices[side, rank]))
    return _place(path, spec, side, int(levels[side]), now, marks[5])


def generate_dataset(spec: QueueWorldSpec, **kwargs):
    """Lazy import keeps the queue model independent of training data adapters."""
    from .queue_reactive_data import generate_dataset as generate
    return generate(spec, **kwargs)


def counterfactual_truth(dataset, **kwargs):
    """Known-generator matched intervention, not a learned-model forecast."""
    from .queue_reactive_data import counterfactual_truth as truth
    return truth(dataset, **kwargs)


def tier0_manipulation_check(**kwargs):
    from .queue_reactive_data import tier0_manipulation_check as check
    return check(**kwargs)
