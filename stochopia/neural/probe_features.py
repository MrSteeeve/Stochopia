"""Shared causal encoder and tick-coordinate operations for P2 probes."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .lob_interface import BUCKET_COUNTS, IS_OWN_FEATURE_INDEX, LOBWorldModel, StepObservation, decode_events

STATE_DIM = 60
FEATURE_DIM = 75
EMBEDDING_DIM = 8


class ProbeBase(LOBWorldModel):
    """All probes share the same 75 features and four conditional likelihood heads."""

    def __init__(self) -> None:
        super().__init__()
        offsets = [0]
        for count in BUCKET_COUNTS[:-1]:
            offsets.append(offsets[-1] + count)
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self.embedding = nn.Embedding(sum(BUCKET_COUNTS), EMBEDDING_DIM)
        self.heads = nn.ModuleList(nn.Linear(STATE_DIM + j * EMBEDDING_DIM, n)
                                   for j, n in enumerate(BUCKET_COUNTS))

    def raw_features(self, obs: StepObservation) -> Tensor:
        ticks = obs.book_ticks
        shares = obs.book_shares.to(self.embedding.weight.dtype)
        touch = ticks[..., 0]
        present = touch > 0
        mid = touch.sum(dim=-1) / present.sum(dim=-1).clamp_min(1)
        prices = ((ticks - mid[..., None, None]) / 20).clamp(-10, 10)
        prices = prices.masked_fill(ticks <= 0, 0)
        depth = (shares / 100).log1p()
        bid, ask = touch.unbind(-1)
        spread = (ask - bid).clamp_min(0).masked_fill(~present.all(-1), 0).log1p()
        bvol, avol = shares.sum(-1).unbind(-1)
        imbalance = (bvol - avol) / (bvol + avol).clamp_min(1)
        previous = self.embedding(obs.previous_event + self.offsets)
        scalars = torch.stack((spread, imbalance, obs.is_own), dim=-1)
        return torch.cat((prices.flatten(-2), depth.flatten(-2), scalars,
                          previous.flatten(-2)), dim=-1).to(shares.dtype)

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        if prior:
            embedded = self.embedding(torch.stack(prior, dim=-1) + self.offsets[:stage])
            context = torch.cat((context, embedded.flatten(-2)), dim=-1)
        return self.heads[stage](context)


def input_projection(out_features: int) -> nn.Linear:
    """At initialization, historical absent own labels do not induce a response."""
    layer = nn.Linear(FEATURE_DIM, out_features)
    nn.init.zeros_(layer.weight[:, IS_OWN_FEATURE_INDEX])
    return layer


def volume_clock(obs: StepObservation) -> Tensor:
    """Decode the causal prior event's representative shares on a common 500-share clock."""
    shares = decode_events(obs.previous_event)[..., 3].float()
    return (shares / 500).clamp(0.125, 8.0).unsqueeze(-1)


def previous_flow(obs: StepObservation) -> Tensor:
    """Prior aggressive buy/sell sign; all passive messages carry zero aggressor flow."""
    action = obs.previous_event[..., 1]
    return ((action == 2).float() - (action == 3).float()).unsqueeze(-1)


def level_inputs(obs: StepObservation, dtype: torch.dtype) -> Tensor:
    """Form [..., 10, 5] tick-offset local inputs; preserve gaps as zero occupancy.

    Each row denotes distance 0..9 ticks from its own-side touch. Raw features
    separately retain farther quoted levels; local state never treats sparse
    rank gaps as physical tick distances.
    """
    ticks = obs.book_ticks
    distance = (ticks - ticks[..., :1]).abs().long()
    valid = (ticks > 0) & (ticks[..., :1] > 0) & (distance < 10)
    indices = distance.clamp(0, 9)
    shares = obs.book_shares.to(dtype)
    depth = torch.zeros_like(shares).scatter_add(-1, indices, (shares / 100).log1p() * valid)
    occupancy = torch.zeros_like(shares).scatter_add(-1, indices, valid.to(dtype))
    offsets = torch.arange(10, device=ticks.device, dtype=dtype) / 9
    offsets = offsets.expand_as(depth[..., 0, :]).unsqueeze(-1)
    return torch.cat((depth.transpose(-1, -2), occupancy.transpose(-1, -2), offsets), dim=-1)


def advect_level_state(latent: Tensor, previous_touch: Tensor, current_touch: Tensor) -> Tensor:
    """Translate [B,10,6] hidden slots preserving absolute bid/ask price coordinates.

    Split six channels into three per side. For a bid rise of d ticks, a saved
    price becomes d slots farther from touch; for an ask rise it becomes d
    slots nearer. Fill entering rows with zeros; no wrap-around.
    """
    orientation = previous_touch.new_tensor((1, -1))
    delta = ((current_touch - previous_touch) * orientation).long()
    delta = torch.where(previous_touch > 0, delta, torch.zeros_like(delta))
    source = torch.arange(10, device=latent.device)[None, :, None] - delta[:, None, :]
    valid = (source >= 0) & (source < 10)
    index = source.clamp(0, 9).unsqueeze(-1).expand(-1, -1, -1, 3)
    sides = latent.reshape(-1, 10, 2, 3)
    shifted = sides.gather(1, index) * valid.unsqueeze(-1)
    return shifted.flatten(-2)
