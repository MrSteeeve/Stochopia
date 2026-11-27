"""Shared fixed-budget teacher-forced fitting for P1 order-book models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from stochopia.data import SZSEDaySlice

from .lob_interface import (
    FiveTupleLogits, FiveTupleLoss, LOBBatch, LOBWorldModel, StepObservation,
    encode_events, make_batch, world_model_loss,
)


@dataclass(frozen=True)
class FitProtocol:
    subseq_len: int = 256
    burn_in: int = 64
    batch_size: int = 16
    steps: int = 1_200
    start_lr: float = 0.002
    end_lr: float = 0.0001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    max_train_seconds: float = 180.0


DEFAULT_FIT_PROTOCOL = FitProtocol()
DEFAULT_TRAIN_SEEDS = (103, 137, 211)


class NonfiniteModelLoss(ValueError):
    """The model did not produce finite loss or training gradients."""


@dataclass(frozen=True)
class EncodedDay:
    day: SZSEDaySlice
    ticks: Tensor
    shares: Tensor
    tokens: Tensor


@dataclass(frozen=True)
class LobFit:
    model: LOBWorldModel
    seconds: float
    last_nll: float


def encode_day(day: SZSEDaySlice) -> EncodedDay:
    ticks = torch.tensor(day.pre_book_ticks)
    shares = torch.tensor(day.pre_book_shares)
    tokens = encode_events(
        torch.tensor(day.dt_ms), torch.tensor(day.action_type), torch.tensor(day.rel_ticks),
        torch.tensor(day.volume_shares), ticks,
    )
    return EncodedDay(day, ticks, shares, tokens)


def fit_lob_world_model(
    factory: Callable[[], LOBWorldModel], days: Sequence[EncodedDay], seed: int,
    *, protocol: FitProtocol = DEFAULT_FIT_PROTOCOL, device: str | torch.device = "cpu",
) -> LobFit:
    """Fit the same set of sampled session windows for any factory at a fixed seed.

    The model consumes [batch, 256, ...]. Only targets after the 64-event burn-in
    contribute terms to the four-head objective. History still shapes the latent
    state used by supervised positions. Abort rather than evaluate a partial fit.
    """
    if not 0 <= protocol.burn_in < protocol.subseq_len or min(protocol.batch_size, protocol.steps) < 1:
        raise ValueError("invalid fit window or step budget")
    eligible = [(item, start, end) for item in days for start, end in item.day.session_bounds
                if end - start >= protocol.subseq_len]
    if not eligible:
        raise ValueError("no same-session windows meet the registered training length")

    started = perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    selected = torch.device(device)
    model = factory().to(selected).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.start_lr, weight_decay=protocol.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, protocol.steps - 1), eta_min=protocol.end_lr,
    )
    supervised = slice(protocol.burn_in, None)
    for step in range(protocol.steps):
        if perf_counter() - started >= protocol.max_train_seconds:
            raise TimeoutError(f"seed {seed} exceeded {protocol.max_train_seconds:g}s training budget")
        ticks = []
        shares = []
        tokens = []
        for index in rng.integers(len(eligible), size=protocol.batch_size):
            item, start, end = eligible[index]
            lo = int(rng.integers(start, end - protocol.subseq_len + 1))
            window = slice(lo, lo + protocol.subseq_len)
            ticks.append(item.ticks[window])
            shares.append(item.shares[window])
            tokens.append(item.tokens[window])

        # Keep the full encoded archive on CPU; move only each sampled
        # minibatch to the same device as the candidate.
        targets = torch.stack(tokens).to(selected)
        batch = make_batch(torch.stack(ticks).to(selected), torch.stack(shares).to(selected), targets)
        full = model.forward_sequence(batch)
        logits = FiveTupleLogits(*(head[:, supervised] for head in full))
        loss = world_model_loss(logits, targets[:, supervised]).total
        if not torch.isfinite(loss):
            raise NonfiniteModelLoss(f"seed {seed} produced a nonfinite training objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol.grad_clip)
        if not torch.isfinite(norm):
            raise NonfiniteModelLoss(f"seed {seed} produced a nonfinite gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step + 1 < protocol.steps:
            scheduler.step()

    elapsed = perf_counter() - started
    if elapsed >= protocol.max_train_seconds:
        raise TimeoutError(f"seed {seed} exceeded {protocol.max_train_seconds:g}s training budget")
    return LobFit(model.eval(), elapsed, float(loss.detach()))


# --- Trained-model likelihood and memory ablation ---

@torch.no_grad()
def world_model_nll_events(
    model: LOBWorldModel, batch: LOBBatch, *, stateless: bool = False,
    reset_interval: int | None = None,
) -> FiveTupleLoss:
    """Return fixed four-head losses [B, T] in nats per event.

    A finite reset interval L partitions each sequence at 0,L,2L,... and
    resets only the recurrent state. Complete blocks run in one batched
    forward pass and any short tail in another. Previous tokens, book,
    markers, targets and original event indices are unchanged. Stateless
    means L=1; None uses the full stateful sequence. Supervision masks must
    be applied after restoring [B,T], so denominators are identical.
    """
    if reset_interval is not None and (
        isinstance(reset_interval, bool) or not isinstance(reset_interval, int) or reset_interval < 1
    ):
        raise ValueError("reset_interval must be a positive integer or None")
    if stateless and reset_interval not in (None, 1):
        raise ValueError("stateless=True requires reset_interval=1")
    interval = 1 if stateless else reset_interval
    was_training = model.training
    model.eval()
    try:
        batch_size, time_steps = batch.targets.shape[:2]
        if interval is None or interval >= time_steps:
            return world_model_loss(model.forward_sequence(batch), batch.targets, reduction="none")

        complete = time_steps // interval * interval
        observations: list[Tensor | None] = []
        for field in batch.observation:
            if field is None:
                observations.append(None)
            else:
                observations.append(field[:, :complete].reshape(-1, interval, *field.shape[2:]))
        folded_targets = batch.targets[:, :complete].reshape(-1, interval, 4)
        folded = LOBBatch(StepObservation(*observations), folded_targets)
        loss = world_model_loss(model.forward_sequence(folded), folded_targets, reduction="none")
        restored = [field.reshape(batch_size, complete) for field in loss]
        if complete < time_steps:
            tail_observation = StepObservation(*[
                None if field is None else field[:, complete:] for field in batch.observation
            ])
            tail_targets = batch.targets[:, complete:]
            tail_batch = LOBBatch(tail_observation, tail_targets)
            tail = world_model_loss(model.forward_sequence(tail_batch), tail_targets, reduction="none")
            restored = [torch.cat((prefix, suffix), dim=1) for prefix, suffix in zip(restored, tail, strict=True)]
        return FiveTupleLoss(*restored)
    finally:
        model.train(was_training)


def evaluate_world_model_nll(
    model: LOBWorldModel, batches: Sequence[LOBBatch], *, stateless: bool = False,
    supervised: slice = slice(None), reset_interval: int | None = None,
) -> dict[str, float]:
    """Aggregate fixed head means, weighting by number of supervised events.

    With like-shaped batches, setting supervised=slice(burn_in,None) excludes
    the same burn-in positions from both recurrent and stateless denominators.
    Total is the unweighted sum of four per-event means. Empty supervision and
    non-finite likelihood are invalid, not silently replaced with zero.
    """
    head_sums = [0.0, 0.0, 0.0, 0.0]
    events = 0
    for batch in batches:
        loss = world_model_nll_events(model, batch, stateless=stateless, reset_interval=reset_interval)
        events += loss.total[:, supervised].numel()
        for position, value in enumerate(loss[1:]):
            head_sums[position] += float(value[:, supervised].sum(dtype=torch.float64))
    if events == 0:
        raise ValueError("no supervised events for NLL evaluation")
    mean_heads = np.array(head_sums) / events
    if not np.isfinite(mean_heads).all():
        raise NonfiniteModelLoss("non-finite evaluation likelihood")
    means = {name: float(value) for name, value in zip(FiveTupleLoss._fields[1:], mean_heads, strict=True)}
    means["total"] = float(mean_heads.sum())
    return means


def compute_memory_horizon_profile(
    model: LOBWorldModel, batches: Sequence[LOBBatch], intervals: Sequence[int] = (1, 4, 16, 64),
    *, supervised: slice = slice(None),
) -> dict[int, float]:
    """G_mem(L)=NLL_reset_every_L - NLL_stateful on identical target positions.

    Each estimate uses the same batches, fitted weights and supervised slice.
    Non-finite likelihood is an error; no clipping of signed gaps is applied.
    """
    stateful = evaluate_world_model_nll(model, batches, supervised=supervised)["total"]
    profile: dict[int, float] = {}
    for interval in intervals:
        reset = evaluate_world_model_nll(model, batches, reset_interval=interval, supervised=supervised)["total"]
        profile[interval] = reset - stateful
    return profile
