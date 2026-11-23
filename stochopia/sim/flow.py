"""Order flow from parallel metaorders (Lillo-Mike-Farmer construction).

``n_slots`` traders are active at once. Every trade is issued by one slot
picked uniformly at random and carries the sign of that slot's current
metaorder. Metaorder lengths follow a truncated power law with tail exponent
``alpha``, which gives a trade-sign autocorrelation decaying roughly as
``lag ** -(alpha - 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FlowSpec:
    n_slots: int = 10
    alpha: float = 1.5
    max_len: int = 5000
    informed_frac: float = 0.0

    def __post_init__(self) -> None:
        if self.n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        if self.alpha <= 1.0:
            raise ValueError("alpha must be > 1")
        if self.max_len < 1:
            raise ValueError("max_len must be >= 1")
        if not 0.0 <= self.informed_frac <= 1.0:
            raise ValueError("informed_frac must lie in [0, 1]")


@dataclass(frozen=True)
class Flow:
    """One stream of trades.

    ``meta_id`` and ``informed`` are identity-level ground truth. They exist
    for verification only and must never be fed to a world model.
    """

    signs: np.ndarray  # int8, +1 buy / -1 sell
    meta_id: np.ndarray  # int64, which metaorder issued the trade
    informed: np.ndarray  # bool, whether that metaorder is informed


def _metaorder_lengths(spec: FlowSpec, count: int, rng: np.random.Generator) -> np.ndarray:
    # P(L >= k) = k ** -alpha for k = 1, 2, ...; 1 - U lies in (0, 1]
    lengths = np.floor((1.0 - rng.random(count)) ** (-1.0 / spec.alpha))
    return np.minimum(lengths, spec.max_len).astype(np.int64)


def simulate_flow(spec: FlowSpec, n_trades: int, rng: np.random.Generator) -> Flow:
    if n_trades < 1:
        raise ValueError("n_trades must be >= 1")
    slot = rng.integers(0, spec.n_slots, size=n_trades)
    signs = np.empty(n_trades, dtype=np.int8)
    meta_id = np.empty(n_trades, dtype=np.int64)
    informed = np.empty(n_trades, dtype=bool)
    next_id = 0
    for s in range(spec.n_slots):
        idx = np.flatnonzero(slot == s)
        need = idx.size
        if need == 0:
            continue
        lengths = _metaorder_lengths(spec, need // 2 + 16, rng)
        while lengths.sum() < need:
            lengths = np.concatenate([lengths, _metaorder_lengths(spec, need // 2 + 16, rng)])
        n_meta = int(np.searchsorted(np.cumsum(lengths), need)) + 1
        lengths = lengths[:n_meta]
        meta_sign = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_meta)
        meta_informed = rng.random(n_meta) < spec.informed_frac
        signs[idx] = np.repeat(meta_sign, lengths)[:need]
        informed[idx] = np.repeat(meta_informed, lengths)[:need]
        meta_id[idx] = np.repeat(np.arange(next_id, next_id + n_meta), lengths)[:need]
        next_id += n_meta
    return Flow(signs=signs, meta_id=meta_id, informed=informed)
