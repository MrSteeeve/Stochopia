"""Capacity-matched, causal synthetic-study probes; no model discovery or fitting here."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import log

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .lob_interface import LOBWorldModel, StepObservation, State, decode_events
from .probe_features import (
    STATE_DIM, ProbeBase, advect_level_state, input_projection, level_inputs, previous_flow, volume_clock,
)
from .primitives import parallel_affine_scan


class _SurpriseGate(nn.Module):
    """Estimate current prior-event flow from pre-update state, then gate slow news."""

    def __init__(self) -> None:
        super().__init__()
        self.predict = nn.Linear(STATE_DIM, 1)
        self.gain = nn.Linear(1, 1)
        nn.init.ones_(self.gain.weight)
        nn.init.constant_(self.gain.bias, -1.0)

    def forward(self, proposal: Tensor, prior: Tensor, flow: Tensor) -> Tensor:
        # Genuinely lagged prediction: neither proposal nor current target e[t] is observed.
        residual = flow - self.predict(prior).tanh()
        update = self.gain(residual.abs()).sigmoid()
        fresh, slow = proposal.split(30, dim=-1)
        old = prior[..., 30:]
        return torch.cat((fresh, update * slow + (1 - update) * old), dim=-1)


class GRUProbe(ProbeBase):
    """One 60-dimensional GRU, optionally changing retention or only its readout."""

    def __init__(self, mode: str = "base") -> None:
        super().__init__()
        self.mode = mode
        self.encoder = input_projection(32)
        self.gru = nn.GRU(32, STATE_DIM, batch_first=True)
        if mode == "multiscale":
            self.register_buffer("scales", torch.logspace(0, log(64, 10), STATE_DIM))
        elif mode == "surprise":
            self.surprise = _SurpriseGate()
        elif mode == "readoutgate":
            self.bypass = nn.Linear(32, STATE_DIM)
            self.readout_gate = nn.Linear(2, 1)

    def _update(self, x: Tensor, prior: Tensor, clock: Tensor, flow: Tensor) -> Tensor:
        _, proposal = self.gru(x.unsqueeze(1), prior.unsqueeze(0))
        proposal = proposal[0]
        if self.mode == "multiscale":
            retention = torch.exp(-clock / self.scales)
            return retention * prior + (1 - retention) * proposal
        if self.mode == "surprise":
            return self.surprise(proposal, prior, flow)
        return proposal

    def _readout(self, raw: Tensor, x: Tensor, state: Tensor) -> Tensor:
        if self.mode != "readoutgate":
            return state
        gate = self.readout_gate(raw[..., 40:42]).sigmoid()
        return gate * state + (1 - gate) * self.bypass(x).tanh()

    def sequence_context(self, obs: StepObservation) -> Tensor:
        raw = self.raw_features(obs)
        x = self.encoder(raw).tanh()
        if self.mode in ("base", "readoutgate"):
            states, _ = self.gru(x)
            return self._readout(raw, x, states)
        prior = x.new_zeros((x.size(0), STATE_DIM))
        clock = volume_clock(obs)
        flow = previous_flow(obs)
        states = []
        for t in range(x.size(1)):
            prior = self._update(x[:, t], prior, clock[:, t], flow[:, t])
            states.append(prior)
        return torch.stack(states, dim=1)

    def step_context(self, obs: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        raw = self.raw_features(obs)
        x = self.encoder(raw).tanh()
        prior = x.new_zeros((x.size(0), STATE_DIM)) if state is None else state
        updated = self._update(x, prior, volume_clock(obs), previous_flow(obs))
        return self._readout(raw, x, updated), updated


class DiagonalProbe(ProbeBase):
    """Real stable diagonal LRU: base common time scale, or controlled variants."""

    def __init__(self, mode: str = "base") -> None:
        super().__init__()
        self.mode = mode
        # A larger, genuinely used nonlinear encoder matches the GRU parameter order
        # without storing additional recurrent state or adding dummy parameters.
        width = 100 if mode == "selective" else 128
        self.encoder = nn.Sequential(input_projection(width), nn.Tanh(),
                                     nn.Linear(width, STATE_DIM), nn.Tanh())
        if mode == "selective":
            self.select = nn.Linear(STATE_DIM, STATE_DIM)
        else:
            self.log_tau = nn.Parameter(torch.tensor(log(8.0)))
        if mode == "multiscale":
            self.register_buffer("scales", torch.logspace(-log(8, 10), log(8, 10), STATE_DIM))
        if mode == "surprise":
            self.surprise = _SurpriseGate()

    def _retention(self, x: Tensor, obs: StepObservation) -> Tensor:
        if self.mode == "selective":
            return self.select(x).sigmoid()
        tau = self.log_tau.exp().clamp(0.5, 64)
        if self.mode == "multiscale":
            return torch.exp(-volume_clock(obs) / (tau * self.scales))
        return torch.exp(-1 / tau).expand_as(x)

    def sequence_context(self, obs: StepObservation) -> Tensor:
        x = self.encoder(self.raw_features(obs))
        decay = self._retention(x, obs)
        drive = (1 - decay) * x
        if self.mode != "surprise":
            return parallel_affine_scan(decay, drive)
        flow = previous_flow(obs)
        prior = x.new_zeros((x.size(0), STATE_DIM))
        states = []
        for t in range(x.size(1)):
            proposal = decay[:, t] * prior + drive[:, t]
            prior = self.surprise(proposal, prior, flow[:, t])
            states.append(prior)
        return torch.stack(states, dim=1)

    def step_context(self, obs: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        x = self.encoder(self.raw_features(obs))
        decay = self._retention(x, obs)
        prior = torch.zeros_like(x) if state is None else state
        updated = decay * prior + (1 - decay) * x
        if self.mode == "surprise":
            updated = self.surprise(updated, prior, previous_flow(obs))
        return updated, updated


class LevelMatrixProbe(ProbeBase):
    """[B,10,6] state: three channels per side, with two explicit touch metadata slots.

    The third bid/ask channel at row zero stores prior touch coordinates, leaving
    58 learned scalars. Both controls reserve exactly these slots; only Advect
    transports the hidden coordinates before applying the identical local GRU.
    """

    def __init__(self, advect: bool) -> None:
        super().__init__()
        self.advect = advect
        self.global_encoder = input_projection(24)
        self.local_encoder = nn.Sequential(nn.Linear(29, 96), nn.Tanh(), nn.Linear(96, 6), nn.Tanh())
        self.local_gru = nn.GRU(6, 6, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(58, 128), nn.Tanh(), nn.Linear(128, STATE_DIM), nn.Tanh())
        mask = torch.ones(10, 6)
        mask[0, [2, 5]] = 0
        self.register_buffer("latent_mask", mask, persistent=False)
        self.register_buffer("latent_indices", mask.flatten().nonzero().flatten(), persistent=False)

    def _encode(self, obs: StepObservation) -> Tensor:
        global_features = self.global_encoder(self.raw_features(obs)).tanh()
        wide = global_features.unsqueeze(-2).expand(*global_features.shape[:-1], 10, 24)
        local = level_inputs(obs, global_features.dtype)
        return self.local_encoder(torch.cat((local, wide), dim=-1))

    def _update(self, x: Tensor, touch: Tensor, state: State) -> Tensor:
        prior = torch.zeros_like(x) if state is None else state
        previous = prior[:, 0, [2, 5]]
        current = torch.where(touch > 0, touch.to(x.dtype), previous)
        latent = prior * self.latent_mask
        if self.advect:
            latent = advect_level_state(latent, previous, current)
        _, updated = self.local_gru(x.reshape(-1, 1, 6), latent.reshape(1, -1, 6))
        updated = updated.reshape(-1, 10, 6) * self.latent_mask
        refs = F.pad(current.unsqueeze(-1), (2, 0)).reshape(-1, 1, 6)
        return updated + F.pad(refs, (0, 0, 0, 9))

    def _readout(self, state: Tensor) -> Tensor:
        return self.readout(state.flatten(-2).index_select(-1, self.latent_indices))

    def sequence_context(self, obs: StepObservation) -> Tensor:
        x = self._encode(obs)
        touch = obs.book_ticks[..., 0]
        state = None
        states = []
        for t in range(x.size(1)):
            state = self._update(x[:, t], touch[:, t], state)
            states.append(state)
        return self._readout(torch.stack(states, dim=1))

    def step_context(self, obs: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        updated = self._update(self._encode(obs), obs.book_ticks[..., 0], state)
        return self._readout(updated), updated


class HawkesProbe(ProbeBase):
    """GRU-48 plus six event-kind × two time-scale positive excitation states.

    Each impulse has integrated branching mass alpha < 0.95, split across
    two kernels. The latent intensities are measured per 100 ms (not target
    realized intensity); previous waiting times age the state before the new impulse.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = input_projection(48)
        self.gru = nn.GRU(60, 48, batch_first=True)
        self.alpha_logit = nn.Parameter(torch.tensor(log(0.70 / (0.95 - 0.70))))
        self.mix = nn.Parameter(torch.zeros(2))
        self.register_buffer("tau_ms", torch.tensor((20.0, 200.0)))

    def _terms(self, obs: StepObservation) -> tuple[Tensor, Tensor]:
        dt = decode_events(obs.previous_event)[..., 0].float()
        base = torch.exp(-dt[..., None] / self.tau_ms)
        action = F.one_hot(obs.previous_event[..., 1], 6).to(base.dtype)
        alpha = 0.95 * self.alpha_logit.sigmoid()
        kernel = alpha * self.mix.softmax(dim=-1) * 100 / self.tau_ms
        drive = (action.unsqueeze(-1) * kernel).flatten(-2)
        decay = base.unsqueeze(-2).expand(*base.shape[:-1], 6, 2).flatten(-2)
        return decay, drive

    def sequence_context(self, obs: StepObservation) -> Tensor:
        decay, drive = self._terms(obs)
        excitation = parallel_affine_scan(decay, drive)
        x = self.encoder(self.raw_features(obs)).tanh()
        signal = excitation.log1p()
        hidden, _ = self.gru(torch.cat((x, signal), dim=-1))
        return torch.cat((hidden, signal), dim=-1)

    def step_context(self, obs: StepObservation, state: State) -> tuple[Tensor, State]:
        decay, drive = self._terms(obs)
        x = self.encoder(self.raw_features(obs)).tanh()
        prior = None if state is None else state[0].unsqueeze(0)
        excitation = drive if state is None else decay * state[1] + drive
        signal = excitation.log1p()
        _, hidden = self.gru(torch.cat((x, signal), dim=-1).unsqueeze(1), prior)
        return torch.cat((hidden[0], signal), dim=-1), (hidden[0], excitation)


PROBE_NAMES = (
    "gru-base", "gru-multiscale", "gru-surprise", "lru-base", "lru-multiscale", "lru-surprise",
    "flat-gru", "matrix-noshift", "matrix-advect", "flat-readoutgate", "selective-ssm", "gru-hawkes",
)


def make_probe(name: str) -> LOBWorldModel:
    """Resolve a preregistered architecture; no hidden fit or random-state mutation."""
    if name not in PROBE_NAMES:
        raise ValueError(f"unknown P2 probe {name!r}")
    if name.startswith("lru-"):
        return DiagonalProbe(name.removeprefix("lru-"))
    if name in ("flat-gru", "flat-readoutgate"):
        return GRUProbe("base" if name == "flat-gru" else "readoutgate")
    if name.startswith("matrix-"):
        return LevelMatrixProbe(advect=name == "matrix-advect")
    if name == "selective-ssm":
        return DiagonalProbe("selective")
    if name == "gru-hawkes":
        return HawkesProbe()
    return GRUProbe(name.removeprefix("gru-"))


def model_factory(name: str) -> Callable[[], LOBWorldModel]:
    return partial(make_probe, name)
