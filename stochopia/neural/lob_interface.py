"""Discrete, autoregressive order-event interface shared by fitting and replay.

Causal observations, before any candidate-specific encoding:

* book_ticks[..., 2, 10]: integer absolute price ticks; bid side first, ask
  side second, levels ordered from touch to depth; zero denotes an empty level.
* book_shares[..., 2, 10]: resting quantity in shares, on the same level grid.
* previous_event[..., 4]: previous event's discrete (waiting-time, action
  including direction, relative price/rank, size) tokens. The fixed historical
  tokenization bins time and size and clips price distances; it is not a
  lossless representation of the continuous raw event fields. The full token
  values, not one scalar summary, reach the candidate.
* is_own[...]: previous event's own-order bit (available in visible replay).
* has_deep[..., 2]: optional exact beyond-level-ten availability.

The two ten-level book arrays are not aggregated or clipped at this boundary.
Candidates can form relative prices from these ticks. The historical seed
uses a 43-channel numeric book prefix (layout below) followed by learned
embeddings of all four prior-event tokens. It alone chooses its price scaling,
clipping, and depth transform. Four categorical vocabularies, matching masks,
and the unweighted sum of their NLLs remain fixed in the shared framework.
"""

from __future__ import annotations

from typing import NamedTuple

from .lob_state import (
    State, concat_states, detach_or_clone_state, index_select_state, state_num_elements_per_sample,
)

import torch
from torch import Tensor, nn
from torch.nn import functional as F

_DT_UPPER = (0, 10, 20, 30, 50, 80, 130, 210, 340, 550, 890, 1440, 2330, 3770, 10000)
_DT_REP = (0, 10, 20, 30, 40, 70, 100, 170, 270, 440, 710, 1150, 1860, 3030, 6890, 15000)
_LOT_UPPER = (99, 100, 200, 300, 400, 500, 800, 1200, 2000, 5000, 10000)
_LOT_SHARES = (50, 100, 200, 300, 400, 500, 700, 1000, 1600, 3500, 7500, 15000)
DT_BINS, ACTION_BINS, REL_TICK_BINS, LOT_BINS = 16, 6, 26, 12
BUCKET_COUNTS = (DT_BINS, ACTION_BINS, REL_TICK_BINS, LOT_BINS)
# Numeric book prefix in side-major, touch-to-depth order. A packed extension
# uses the four raw prior tokens instead of a candidate's learned embeddings.
BID_PRICE_FEATURE_SLICE = slice(0, 10)
ASK_PRICE_FEATURE_SLICE = slice(10, 20)
BID_DEPTH_FEATURE_SLICE = slice(20, 30)
ASK_DEPTH_FEATURE_SLICE = slice(30, 40)
SPREAD_FEATURE_INDEX = 40
IMBALANCE_FEATURE_INDEX = 41
IS_OWN_FEATURE_INDEX = 42
BOOK_FEATURE_WIDTH = 43
PRIOR_DT_FEATURE_INDEX = 43
PRIOR_ACTION_FEATURE_INDEX = 44
PRIOR_REL_FEATURE_INDEX = 45
PRIOR_VOLUME_FEATURE_INDEX = 46
PACKED_OBSERVATION_WIDTH = 47
MAX_WORLD_MODEL_PARAMS = 28_000
MAX_WORLD_MODEL_STATE = 128


class StepObservation(NamedTuple):
    """Pre-event book [..., 2, 10], previous four tokens, and prior-event own flag."""

    book_ticks: Tensor
    book_shares: Tensor
    previous_event: Tensor
    is_own: Tensor
    has_deep: Tensor | None = None  # True for a side with more than ten resting levels.

    def at(self, index: tuple[slice | int, ...]) -> StepObservation:
        return StepObservation(*(x[index] if x is not None else None for x in self))


class LOBBatch(NamedTuple):
    observation: StepObservation
    targets: Tensor  # [batch, time, 4], ordered time, action, relative price, size.


class FiveTupleLogits(NamedTuple):
    dt: Tensor
    action: Tensor
    rel_ticks: Tensor
    volume: Tensor


class FiveTupleLoss(NamedTuple):
    total: Tensor
    dt: Tensor
    action: Tensor
    rel_ticks: Tensor
    volume: Tensor


def _row_by_side(book: Tensor, side: Tensor) -> Tensor:
    index = side[..., None, None].expand(*side.shape, 1, book.size(-1))
    return book.gather(-2, index).squeeze(-2)


def encode_events(dt_ms: Tensor, action_type: Tensor, rel_ticks: Tensor,
                  volume_shares: Tensor, book_ticks: Tensor) -> Tensor:
    """Encode SZSE events; convert cancellation tick distance to pre-book level rank."""
    action = action_type.long()
    book = _row_by_side(book_ticks, action % 2)
    at_price = (book > 0) & ((book - book[..., :1]).abs() == rel_ticks[..., None])
    rank = torch.arange(10, device=book.device)
    cancel = torch.where(at_price, rank, 10).amin(dim=-1)
    rel = torch.where(action < 2, rel_ticks.clamp(-5, 20) + 5, rel_ticks.clamp(0, 10))
    rel = torch.where(action >= 4, cancel, rel)
    dt = torch.bucketize(dt_ms.contiguous(), dt_ms.new_tensor(_DT_UPPER))
    size = torch.bucketize(volume_shares.contiguous(), volume_shares.new_tensor(_LOT_UPPER))
    return torch.stack((dt, action, rel.long(), size), dim=-1)


def decode_events(tokens: Tensor) -> Tensor:
    """Decode [..., 4] to milliseconds, action, relative ticks (cancel rank), shares."""
    dt, action, rel, size = tokens.unbind(dim=-1)
    dt = tokens.new_tensor(_DT_REP)[dt]
    shares = tokens.new_tensor(_LOT_SHARES)[size]
    return torch.stack((dt, action, torch.where(action < 2, rel - 5, rel), shares), dim=-1)


def make_batch(book_ticks: Tensor, book_shares: Tensor, targets: Tensor,
               is_own: Tensor | None = None) -> LOBBatch:
    """Pair each target with its current pre-book and previous event, resetting at BOS."""
    own = torch.zeros_like(targets[..., 0], dtype=torch.bool) if is_own is None else is_own
    previous = torch.cat((torch.zeros_like(targets[:, :1]), targets[:, :-1]), dim=1)
    prior_own = torch.cat((torch.zeros_like(own[:, :1]), own[:, :-1]), dim=1)
    return LOBBatch(StepObservation(book_ticks, book_shares, previous, prior_own), targets)


# --- Matching constraints ---

def passive_anchor(book_ticks: Tensor, side: Tensor) -> Tensor:
    """Use the same-side touch, or one tick inside the opposite touch for an empty side."""
    bid, ask = book_ticks[..., 0].unbind(dim=-1)
    buy = torch.where(bid > 0, bid, (ask - 1).clamp_min(1))
    sell = torch.where(ask > 0, ask, (bid + 1).clamp_min(1))
    return torch.where(side == 0, buy, sell)


def valid_action_mask(book_ticks: Tensor, book_shares: Tensor) -> Tensor:
    """Keep passive replenishment; require resting depth for execution and cancellation."""
    bid, ask = ((book_ticks > 0) & (book_shares > 0)).any(dim=-1).unbind(dim=-1)
    buy_passive = book_ticks[..., 1, 0] != 1
    return torch.stack((buy_passive, torch.ones_like(bid), ask, bid, bid, ask), dim=-1)


def valid_rel_tick_mask(observation: StepObservation, action: Tensor) -> Tensor:
    """Mask impossible passive prices and known empty cancellation levels."""
    ticks, shares = observation.book_ticks, observation.book_shares
    active = (ticks > 0) & (shares > 0)
    side = action % 2
    index = torch.arange(REL_TICK_BINS, device=ticks.device)
    offset = index - 5
    bid, ask = ticks[..., 0].unbind(dim=-1)
    spread = ask - bid
    improving = (bid > 0) & (ask > 0)
    improving = improving[..., None] & (spread[..., None] + offset > 0)
    direction = 2 * side - 1
    price = passive_anchor(ticks, side)[..., None] + direction[..., None] * offset
    passive = (price > 0) & ((offset >= 0) | improving)

    resting = _row_by_side(active, side)
    deep = observation.has_deep
    tail = resting.all(dim=-1, keepdim=True) if deep is None else deep.gather(-1, side[..., None])
    cancel = F.pad(torch.cat((resting, tail), dim=-1), (0, REL_TICK_BINS - 11))
    return torch.where((action < 2)[..., None], passive,
                       torch.where((action < 4)[..., None], index < 11, cancel))


# --- Shared teacher-forced and sampled heads ---

class StepSampler(NamedTuple):
    model: LOBWorldModel
    context: Tensor
    observation: StepObservation

    def logits(self, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        """Return conditional categorical logits subject to framework masks.

        Candidates supply only the public `head_logits` contract. The book
        masks belong to the sampler, so training and both rollout engines
        use exactly the same legal event vocabulary.
        """
        logits = self.model.head_logits(self.context, stage, prior)
        obs = self.observation
        if stage == 1:
            return logits.masked_fill(~valid_action_mask(obs.book_ticks, obs.book_shares), -torch.inf)
        if stage == 2:
            return logits.masked_fill(~valid_rel_tick_mask(obs, prior[1]), -torch.inf)
        return logits

    def conditioned(self, targets: Tensor) -> FiveTupleLogits:
        prior = targets.unbind(dim=-1)
        return FiveTupleLogits(*(self.logits(i, prior[:i]) for i in range(4)))

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        tokens: list[Tensor] = []
        for stage in range(4):
            logits = self.logits(stage, tuple(tokens))
            chosen = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)
            tokens.append(chosen.squeeze(-1))
        return torch.stack(tokens, dim=-1)


class LOBWorldModel(nn.Module):
    """Subclass the causal state update and four conditional heads."""

    def sequence_context(self, observation: StepObservation) -> Tensor:
        raise NotImplementedError

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        raise NotImplementedError

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError

    def forward_sequence(self, batch: LOBBatch) -> FiveTupleLogits:
        context = self.sequence_context(batch.observation)
        return StepSampler(self, context, batch.observation).conditioned(batch.targets)

    def forward_step(self, step_obs: StepObservation, hidden_state: State = None) -> tuple[StepSampler, State]:
        context, state = self.step_context(step_obs, hidden_state)
        return StepSampler(self, context, step_obs), state

    def _logits(self, sampler: StepSampler, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        """Compatibility alias for old candidates using the former helper."""
        return sampler.logits(stage, prior)


def world_model_loss(logits: FiveTupleLogits, targets: Tensor, *, reduction: str = "mean") -> FiveTupleLoss:
    """Sum four discrete negative log probabilities in nats per event."""
    terms: list[Tensor] = []
    for head, target in zip(logits, targets.unbind(dim=-1)):
        loss = F.cross_entropy(head.flatten(0, -2), target.reshape(-1), reduction=reduction)
        terms.append(loss.reshape(target.shape) if reduction == "none" else loss)
    return FiveTupleLoss(sum(terms), *terms)


# --- Tensor contract checks ---

def _probe(device: torch.device, seed: int, batch_size: int = 1) -> LOBBatch:
    rng = torch.Generator().manual_seed(seed)
    bid = torch.randint(100, 300, (batch_size, 9, 1), generator=rng)
    spread = torch.randint(1, 10, bid.shape, generator=rng)
    levels = torch.arange(10)
    ticks = torch.stack((bid - levels, bid + spread + levels), dim=-2)
    shares = torch.randint(1, 1500, ticks.shape, generator=rng)
    dt = torch.randint(DT_BINS, (batch_size, 9), generator=rng)
    action = (torch.arange(9).unsqueeze(0) + seed + torch.arange(batch_size)[:, None]) % ACTION_BINS
    size = torch.randint(LOT_BINS, dt.shape, generator=rng)
    own = torch.rand(dt.shape, generator=rng) > 0.5
    mask = valid_rel_tick_mask(StepObservation(ticks, shares, torch.empty(0), own), action)
    rel = torch.rand(mask.shape, generator=rng).masked_fill(~mask, -1).argmax(dim=-1)
    targets = torch.stack((dt, action, rel, size), dim=-1).to(device)
    return make_batch(ticks.to(device), shares.to(device), targets, own.to(device))


def _check_batch_contract(
    model: LOBWorldModel, batch: LOBBatch, other: LOBBatch, max_state_dim: int,
) -> int:
    """Check causal prefix, legal likelihood, and step/sequence equality in one batch."""
    obs, targets = batch
    prefix = targets.size(1) // 2 + 1
    edited = StepObservation(*(torch.cat((a[:, :prefix], b[:, prefix:]), dim=1)
                               for a, b in zip(obs[:4], other.observation[:4], strict=True)))
    changed = torch.cat((targets[:, :prefix], other.targets[:, prefix:]), dim=1)
    expected = model.forward_sequence(batch)
    alternate = model.forward_sequence(LOBBatch(edited, changed))
    context = model.sequence_context(obs)
    for head, perturbed, classes in zip(expected, alternate, BUCKET_COUNTS, strict=True):
        if head.shape != (*targets.shape[:-1], classes):
            raise ValueError("wrong categorical head shape")
        torch.testing.assert_close(head[:, :prefix], perturbed[:, :prefix], rtol=0, atol=1e-5)
    action_mask = valid_action_mask(obs.book_ticks, obs.book_shares)
    price_mask = valid_rel_tick_mask(obs, targets[..., 1])
    for head, legal in zip(expected[1:3], (action_mask, price_mask), strict=True):
        if not torch.isneginf(head[~legal]).all():
            raise ValueError("unmasked illegal action or price bucket")
    if not torch.isfinite(world_model_loss(expected, targets).total):
        raise ValueError("non-finite likelihood for a legal event")

    state = None
    largest_state = 0
    for t in range(targets.size(1)):
        sampler, state = model.forward_step(obs.at((slice(None), t)), state)
        size = state_num_elements_per_sample(state, batch_size=targets.size(0))
        largest_state = max(largest_state, size)
        if size > max_state_dim:
            raise ValueError("model exceeds max_state_dim")
        torch.testing.assert_close(sampler.context, context[:, t], rtol=0, atol=1e-5)
        for step, seq in zip(sampler.conditioned(targets[:, t]), expected, strict=True):
            torch.testing.assert_close(step, seq[:, t], rtol=0, atol=1e-5)
    return largest_state


@torch.no_grad()
def verify_world_model_contract(
    model: LOBWorldModel, *, max_params: int = MAX_WORLD_MODEL_PARAMS, max_state_dim: int = MAX_WORLD_MODEL_STATE,
) -> tuple[int, int]:
    """Check fixed categorical heads, causality and sequence/step equivalence.

    Independent probes at batch sizes one and five exercise dynamic batching;
    each returned state is charged by real scalar elements per sample (not the
    product of batch size and state geometry). The tolerance is 1e-5 absolute
    for prefix and step equivalence. Restore the caller's training mode.
    """
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters > max_params:
        raise ValueError("model exceeds max_params")
    device = next(model.parameters(), torch.empty(0)).device
    training = model.training
    model.eval()
    largest_state = 0
    try:
        for batch_size in (1, 5):
            batch = _probe(device, 291, batch_size)
            other = _probe(device, 106, batch_size)
            size = _check_batch_contract(model, batch, other, max_state_dim)
            largest_state = max(largest_state, size)
    finally:
        model.train(training)
    return parameters, largest_state
