"""Differentiable affine scan and the single-layer GRU order-book baseline."""

from __future__ import annotations

from inspect import getsource

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .lob_interface import (
    ASK_DEPTH_FEATURE_SLICE, ASK_PRICE_FEATURE_SLICE, BID_DEPTH_FEATURE_SLICE, BID_PRICE_FEATURE_SLICE,
    BOOK_FEATURE_WIDTH, BUCKET_COUNTS, IMBALANCE_FEATURE_INDEX, IS_OWN_FEATURE_INDEX, LOBWorldModel,
    PACKED_OBSERVATION_WIDTH, PRIOR_ACTION_FEATURE_INDEX, State, StepObservation,
)


def affine_scan_step(gate_t: Tensor, input_t: Tensor, h_prev: Tensor) -> Tensor:
    """Return h[t] = gate[t] * h[t-1] + input[t], with normal tensor broadcasting."""
    if gate_t.shape == input_t.shape[-1:]:
        gate_t = gate_t.expand_as(input_t)
    return gate_t * h_prev + input_t


def parallel_affine_scan(
    gates: Tensor, inputs: Tensor, init_state: Tensor | None = None,
    chunk_size: int = 32, *, h0: Tensor | None = None,
) -> Tensor:
    """Scan [B, T, *state_shape] real or complex diagonal affine transitions.

    Gates must broadcast to the entire input tensor; e.g. static gates with
    shape [*state_shape] or batch-specific gates [B, 1, *state_shape].
    [B, *state_shape] may omit time when B differs from T; otherwise
    use the explicit singleton time axis to distinguish it from [T, *state_shape].
    init_state (historical alias h0) has shape [B, *state_shape]. Parallel
    prefixes within a chunk compose (g, x) without division by decay products.
    Only the final state is carried between chunks. Empty time returns an
    empty result, absent init_state starts from zero.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if inputs.ndim < 3:
        raise ValueError("inputs must have shape [batch, time, *state_shape]")
    if init_state is not None and h0 is not None:
        raise ValueError("specify only init_state or its historical alias h0")
    init_state = h0 if init_state is None else init_state
    state_shape = (inputs.size(0), *inputs.shape[2:])
    if init_state is not None and init_state.shape != state_shape:
        raise ValueError("initial state must have shape [batch, *state_shape]")
    if (
        gates.ndim == inputs.ndim - 1
        and gates.size(0) == inputs.size(0)
        and gates.size(0) != inputs.size(1)
    ):
        gates = gates.unsqueeze(1)
    if gates.shape != inputs.shape:
        gates = gates.expand_as(inputs)
    if inputs.size(1) == 0:
        return inputs

    state = inputs.new_zeros(state_shape) if init_state is None else init_state
    states: list[Tensor] = []
    for gate, value in zip(gates.split(chunk_size, dim=1), inputs.split(chunk_size, dim=1), strict=True):
        stride = 1
        # Compose a range of 2 * stride transitions at each position in parallel.
        while stride < gate.size(1):
            merged_value = gate[:, stride:] * value[:, :-stride] + value[:, stride:]
            merged_gate = gate[:, stride:] * gate[:, :-stride]
            gate = torch.cat((gate[:, :stride], merged_gate), dim=1)
            value = torch.cat((value[:, :stride], merged_value), dim=1)
            stride *= 2
        chunk_states = affine_scan_step(gate, value, state[:, None])
        states.append(chunk_states)
        state = chunk_states[:, -1]
    return torch.cat(states, dim=1)


def real_diagonal(decay_logit: Tensor) -> Tensor:
    """Stable real eigenvalues parameterized by decay logits."""
    return decay_logit.sigmoid()


def complex_diagonal(decay_logit: Tensor, phase: Tensor) -> Tensor:
    """Stable complex eigenvalues with decay magnitude and phase in radians."""
    return torch.polar(real_diagonal(decay_logit), phase)


def causal_depthwise_conv1d(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Depthwise time convolution on [B,T,C] or [B,T,*level_shape,C].

    weight is [C,K] (or torch's [C,1,K]), bias is [C]. Level coordinates
    are independent batch lanes. Pad K-1 samples on the *left* only; the
    current output sees the current and past positions, and preserves shape.
    """
    if x.ndim < 3:
        raise ValueError("x must have shape [batch,time,...,channels]")
    if weight.ndim == 2:
        weight = weight.unsqueeze(1)
    channels = x.size(-1)
    if weight.ndim != 3 or weight.shape[:2] != (channels, 1) or weight.size(-1) < 1:
        raise ValueError("weight must have shape [channels, kernel] or [channels, 1, kernel]")
    steps = x.size(1)
    if steps == 0:
        return x
    transposed = x.movedim(1, -1)
    lanes = transposed.reshape(-1, channels, steps)
    padded = F.pad(lanes, (weight.size(-1) - 1, 0))
    filtered = F.conv1d(padded, weight, bias, groups=channels)
    return filtered.reshape(transposed.shape).movedim(-1, 1)


def apply_rotary_2d(x: Tensor, angles: Tensor) -> Tensor:
    """Rotate each adjacent real channel pair by broadcastable angles (radians)."""
    if x.size(-1) % 2:
        raise ValueError("rotation requires an even number of channels")
    even, odd = x[..., 0::2], x[..., 1::2]
    sine = angles.sin()
    cosine = angles.cos()
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def flip_lob_observation(obs: Tensor) -> Tensor:
    """Reflect a numeric book prefix (43 channels), optionally with four raw prior tokens.

    Signed midpoint-relative prices swap bid/ask *and* negate; depth levels
    swap sides, spread and own flag stay fixed, and imbalance changes sign.
    With the 47-channel packed observation, previous action parity swaps
    within each event-kind pair (0<->1,2<->3,4<->5); relative price/rank is
    already side-relative, so its token remains fixed, as do time and size.
    Learned token embeddings are not a canonical observation and cannot be
    mirrored without their candidate's embedding table.
    """
    if obs.size(-1) not in (BOOK_FEATURE_WIDTH, PACKED_OBSERVATION_WIDTH):
        raise ValueError("expected the 43-channel book prefix or 47-channel packed observation")
    result = obs.clone()
    result[..., BID_PRICE_FEATURE_SLICE] = -obs[..., ASK_PRICE_FEATURE_SLICE]
    result[..., ASK_PRICE_FEATURE_SLICE] = -obs[..., BID_PRICE_FEATURE_SLICE]
    result[..., BID_DEPTH_FEATURE_SLICE] = obs[..., ASK_DEPTH_FEATURE_SLICE]
    result[..., ASK_DEPTH_FEATURE_SLICE] = obs[..., BID_DEPTH_FEATURE_SLICE]
    result[..., IMBALANCE_FEATURE_INDEX] = -obs[..., IMBALANCE_FEATURE_INDEX]
    if obs.size(-1) == PACKED_OBSERVATION_WIDTH:
        action = obs[..., PRIOR_ACTION_FEATURE_INDEX]
        result[..., PRIOR_ACTION_FEATURE_INDEX] = action + 1 - 2 * action.remainder(2)
    return result


# --- Seed model ---

class SeedGRULOBModel(LOBWorldModel):
    """A ten-level book encoder and GRU with discrete conditional order heads."""

    def __init__(self, state_dim: int = 48, embedding_dim: int = 8) -> None:
        super().__init__()
        self.state_dim = state_dim
        token_offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            token_offsets.append(token_offsets[-1] + count)
        self.register_buffer("token_offsets", torch.tensor(token_offsets), persistent=False)
        self.token_embed = nn.Embedding(sum(BUCKET_COUNTS), embedding_dim)
        self.input_projection = nn.Linear(43 + 4 * embedding_dim, state_dim)
        # Historical orders have no own labels. Start with no untrained flag response.
        nn.init.zeros_(self.input_projection.weight[:, IS_OWN_FEATURE_INDEX])
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.heads = nn.ModuleList(
            nn.Linear(state_dim + stage * embedding_dim, count)
            for stage, count in enumerate(BUCKET_COUNTS)
        )

    def _features(self, observation: StepObservation) -> Tensor:
        ticks = observation.book_ticks
        book_shares = observation.book_shares.to(self.input_projection.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        relative = (ticks - mid[..., None, None]) / 20
        price_feature = relative.clamp(-10, 10).masked_fill(ticks <= 0, 0)
        depth_feature = (book_shares / 100).log1p()

        bid, ask = touch.unbind(dim=-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(dim=-1), 0).log1p()
        bid_depth, ask_depth = book_shares.sum(dim=-1).unbind(dim=-1)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth).clamp_min(1)
        previous = self.token_embed(observation.previous_event + self.token_offsets)
        features = torch.cat((price_feature.flatten(-2), depth_feature.flatten(-2),
                              torch.stack((spread, imbalance, observation.is_own), dim=-1),
                              previous.flatten(-2)), dim=-1)
        return self.input_projection(features.to(self.input_projection.weight.dtype)).tanh()

    def sequence_context(self, observation: StepObservation) -> Tensor:
        out, _ = self.gru(self._features(observation))
        return out

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, Tensor]:
        prior = None if state is None else state.unsqueeze(0)
        out, hidden = self.gru(self._features(observation).unsqueeze(1), prior)
        return out[:, 0], hidden[0]

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            tokens = torch.stack(prior, dim=-1) + self.token_offsets[:stage]
            prefix = self.token_embed(tokens).flatten(-2)
            context = torch.cat((context, prefix), dim=-1)
        return self.heads[stage](context)


def seed_lob_source() -> str:
    """Export the exact trusted seed implementation using the candidate entry point."""
    imports = (
        "from __future__ import annotations\n"
        "import torch\n"
        "from torch import Tensor, nn\n"
        "from stochopia.neural.lob_interface import (\n"
        "    BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation,\n"
        ")\n\n\n"
    )
    source = getsource(SeedGRULOBModel)
    return imports + source.replace("class SeedGRULOBModel(", "class CandidateLOBWorldModel(", 1)
