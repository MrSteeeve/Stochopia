"""Fixed-kernel linear propagator with a multi-timescale kernel.

The kernel is a sum of exponentials on log-spaced timescales,

    G(lag) = sum_k w_k * rho_k ** lag,    w_k proportional to tau_k ** -beta,

which tends to ``lag ** -beta`` as ``tau_max`` grows and decays to zero
beyond ``tau_max`` (no permanent impact). With a finite ``tau_max`` the kernel
falls off faster than ``lag ** -beta`` well before ``tau_max``: for the
defaults the local log-slope is about 0.36 at lag 10 and 0.52 at lag 100.
The sum-of-exponentials form makes the impact state Markov in ``n_scales``
variables and avoids the truncation artefact of a finite power-law kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class KernelSpec:
    beta: float = 0.25
    tau_min: float = 1.0
    tau_max: float = 1000.0
    n_scales: int = 7
    g0: float = 1.0  # G(0): immediate price move of one unit trade

    def __post_init__(self) -> None:
        if self.n_scales < 1:
            raise ValueError("n_scales must be >= 1")
        if not 0.0 < self.tau_min <= self.tau_max:
            raise ValueError("need 0 < tau_min <= tau_max")


def kernel_components(spec: KernelSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(weights, rhos)`` with ``weights.sum() == g0``."""
    taus = np.geomspace(spec.tau_min, spec.tau_max, spec.n_scales)
    weights = taus ** (-spec.beta)
    weights = weights * (spec.g0 / weights.sum())
    return weights, np.exp(-1.0 / taus)


def kernel_values(spec: KernelSpec, lags: np.ndarray) -> np.ndarray:
    weights, rhos = kernel_components(spec)
    lags = np.asarray(lags, dtype=float)
    return (weights * rhos ** lags[..., None]).sum(axis=-1)


def propagate(spec: KernelSpec, signed_volume: np.ndarray) -> np.ndarray:
    """Impact path ``sum_{s<=t} G(t-s) * signed_volume[s]`` for every t."""
    x = np.asarray(signed_volume, dtype=float)
    weights, rhos = kernel_components(spec)
    out = np.zeros_like(x)
    for w, rho in zip(weights, rhos):
        out += w * lfilter([1.0], [1.0, -rho], x)
    return out
