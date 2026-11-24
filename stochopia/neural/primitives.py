"""Differentiable affine scan and the single-layer GRU order-book baseline."""

from __future__ import annotations

from inspect import getsource

import torch
from torch import Tensor, nn

from .lob_interface import BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, State, StepObservation


def affine_scan_step(gate_t: Tensor, input_t: Tensor, h_prev: Tensor) -> Tensor:
    """One diagonal affine transition h[t] = gate[t] * h[t-1] + input[t]."""
    return gate_t * h_prev + input_t


def parallel_affine_scan(gates: Tensor, inputs: Tensor, h0: Tensor | None = None,
                         chunk_size: int = 32) -> Tensor:
    """Scan [batch, time, channels] real or complex diagonal affine transitions.

    Parallel prefixes within a chunk compose (g, x) without division by decay
    products. Only the final state is carried between chunks. With time zero,
    the output is empty. Without h0, the initial state is zero for every batch.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if gates.shape != inputs.shape or inputs.ndim != 3:
        raise ValueError("gates and inputs must have the same [batch, time, channels] shape")
    if inputs.size(1) == 0:
        return inputs

    state = inputs.new_zeros(inputs.size(0), inputs.size(2)) if h0 is None else h0
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
