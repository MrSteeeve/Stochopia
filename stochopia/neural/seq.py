"""Autoregressive model of trade events ``(sign, return)``.

The model sees what is public in an anonymous market: the sign of each trade
and the price change at that trade. At every step it predicts the next sign
and, conditional on that sign, a Gaussian over the next return. It is trained
by teacher forcing on next-event likelihood.

With ``own_flag`` the events carry a third column marking trades issued by
the party designated as "self". Own trades are actions, not predictions: the
sign head is trained on background trades only, and the return head has a
separate output for own trades. No other identity information enters.

Rollouts can inject exogenous trades. In ``free`` mode the model generates
both the background signs and the returns. In ``clamped`` mode the background
signs are supplied and the model generates returns only, which separates
errors in the model's flow reaction from errors in its price reaction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    hidden: int = 256
    layers: int = 2
    return_scale: float = 1.0  # returns are divided by this before entering the model
    own_flag: bool = False
    cell: str = "gru"  # "gru" | "ssm" | "lru"


class FlowReturnModel(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.n_inputs = 3 if spec.own_flag else 2
        self.n_owners = 2 if spec.own_flag else 1
        self.embed = nn.Linear(self.n_inputs, spec.hidden)
        self.gru = nn.GRU(spec.hidden, spec.hidden, spec.layers, batch_first=True)
        self.sign_head = nn.Linear(spec.hidden, 1)
        self.return_head = nn.Linear(spec.hidden, self.n_owners * 4)

    def forward(self, events: torch.Tensor, state: torch.Tensor | None = None):
        """``events`` is ``[batch, time, n_inputs]``: (sign, scaled return[, own]).

        Returns the logit of P(next background sign = +1) ``[batch, time]``,
        the return parameters ``[batch, time, n_owners, 2, 2]`` indexed by
        (next trade is own, next sign is buy, (mean, log_std)), and the state.
        """
        out, state = self.gru(torch.tanh(self.embed(events)), state)
        logits = self.sign_head(out).squeeze(-1)
        params = self.return_head(out).view(*out.shape[:2], self.n_owners, 2, 2)
        return logits, params, state


_CHUNK = 16
_RHO_MIN, _RHO_MAX = 0.005, 1.0 - 1e-6


def _scan(x: torch.Tensor, rho: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
    """``h_t = rho * h_{t-1} + (1 - rho) * x_t`` along dim 1.

    ``x`` is ``[batch, time, channels]``, ``rho`` is ``[channels]``, the
    incoming ``state`` is ``h_{-1}`` (zeros when ``None``). Computed in chunks
    of ``_CHUNK`` steps with the cumulative-sum form
    ``h_t = rho^(t+1) h_{-1} + (1-rho) sum_{k<=t} rho^(t-k) x_k``; the largest
    exponent inside a chunk is ``(_CHUNK - 1) / tau_min`` with
    ``tau_min = -1 / log(_RHO_MIN) ~= 0.189``, i.e. about ``e^79.5 ~ 3e34``.
    With the 16 summed terms the worst intermediate is about ``5e35 * |x|``,
    which stays inside float32 for ``|x|`` below a few hundred.
    """
    batch, time, _ = x.shape
    pad = (-time) % _CHUNK
    if pad:
        x = nn.functional.pad(x, (0, 0, 0, pad))
    s = torch.arange(_CHUNK, device=x.device, dtype=x.dtype)
    log_rho = torch.log(rho)
    rho_s = torch.exp(s[:, None] * log_rho[None, :])  # [chunk, channels]
    inv_rho_s = torch.exp(-s[:, None] * log_rho[None, :])
    carry = x.new_zeros(batch, rho.shape[0]) if state is None else state
    outs = []
    for c in range(x.shape[1] // _CHUNK):
        xc = x[:, c * _CHUNK : (c + 1) * _CHUNK]
        cs = (xc * (1.0 - rho) * inv_rho_s).cumsum(dim=1)
        h = cs * rho_s[None] + (rho_s * rho)[None] * carry[:, None, :]
        outs.append(h)
        carry = h[:, -1]
    return torch.cat(outs, dim=1)[:, :time]


class LinearStateModel(nn.Module):
    """Diagonal linear state space with a linear (``ssm``) or nonlinear (``lru``)
    readout, sharing the ``FlowReturnModel`` interface.

    The state update is ``h_t = rho * h_{t-1} + (1 - rho) * x_t`` with a
    per-channel decay ``rho = sigmoid(decay_logit)`` clamped to
    ``[_RHO_MIN, _RHO_MAX]``. The initial decay logits place the timescales
    ``1 / (1 - rho)`` on a geometric grid from 1 to 1000 trades, the range of
    the simulator's kernel, so with ``cell="ssm"`` the conditional mean of the
    next return is an exact linear function of the state and the model class
    closely approximates the fixed-kernel propagator: the 256-channel decay
    grid is dense around the simulator's seven scales, though only the
    endpoints coincide exactly. With ``cell="lru"`` a GELU readout
    adds expressiveness while the state stays linear.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        if spec.cell not in ("ssm", "lru"):
            raise ValueError(f"unknown cell {spec.cell!r}")
        self.spec = spec
        self.n_inputs = 3 if spec.own_flag else 2
        self.n_owners = 2 if spec.own_flag else 1
        self.embed = nn.Linear(self.n_inputs, spec.hidden)
        taus = np.geomspace(1.0, 1000.0, spec.hidden)
        rho_init = np.exp(-1.0 / taus)
        self.decay_logit = nn.Parameter(
            torch.as_tensor(np.log(rho_init / (1.0 - rho_init)), dtype=torch.float32)
        )
        self.readout = nn.Linear(spec.hidden, spec.hidden)
        self.sign_head = nn.Linear(spec.hidden, 1)
        self.return_head = nn.Linear(spec.hidden, self.n_owners * 4)

    def forward(self, events: torch.Tensor, state: torch.Tensor | None = None):
        x = self.embed(events)
        rho = torch.sigmoid(self.decay_logit).clamp(_RHO_MIN, _RHO_MAX)
        h = _scan(x, rho, state)
        out = h if self.spec.cell == "ssm" else torch.nn.functional.gelu(self.readout(h))
        logits = self.sign_head(out).squeeze(-1)
        params = self.return_head(out).view(*out.shape[:2], self.n_owners, 2, 2)
        return logits, params, h[:, -1]


def make_model(spec: ModelSpec) -> nn.Module:
    if spec.cell == "gru":
        return FlowReturnModel(spec)
    return LinearStateModel(spec)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def next_event_loss(
    model: FlowReturnModel, events: torch.Tensor, *, sign_logits: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean negative log-likelihood of event t+1 given events up to t."""
    logits, params, _ = model(events[:, :-1])
    if sign_logits is not None:
        logits = sign_logits
    next_sign, next_ret = events[:, 1:, 0], events[:, 1:, 1]
    is_buy = (next_sign > 0).long()
    sign_nll = nn.functional.binary_cross_entropy_with_logits(
        logits, is_buy.float(), reduction="none"
    )
    if model.spec.own_flag:
        is_own = (events[:, 1:, 2] > 0).long()
        background = 1.0 - is_own.float()
        sign_nll = (sign_nll * background).sum() / background.sum().clamp(min=1.0)
    else:
        is_own = torch.zeros_like(is_buy)
        sign_nll = sign_nll.mean()
    index = (is_own * 2 + is_buy)[..., None, None].expand(-1, -1, 1, 2)
    chosen = params.flatten(2, 3).gather(2, index).squeeze(2)
    mean, log_std = chosen[..., 0], chosen[..., 1].clamp(-6.0, 4.0)
    ret_nll = (log_std + 0.5 * ((next_ret - mean) / log_std.exp()) ** 2).mean()
    return sign_nll + ret_nll


def train(
    model: FlowReturnModel,
    days: list[np.ndarray],
    *,
    steps: int,
    batch: int,
    window: int,
    lr: float,
    seed: int,
    device: str,
    own_flags: Callable[[int, int, int, np.random.Generator], np.ndarray] | None = None,
) -> list[float]:
    """Fit by teacher forcing on random windows. ``days`` are ``[n, 2]`` arrays
    of (sign, raw return). For a model with ``own_flag``, ``own_flags(day,
    start, length, rng)`` returns the 0/1 own-trade column of one window.
    Returns the loss every 50 steps."""
    if model.spec.own_flag != (own_flags is not None):
        raise ValueError("own_flags must be given exactly when the model has own_flag")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data = [torch.as_tensor(d, dtype=torch.float32) for d in days]
    scale = model.spec.return_scale
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    history: list[float] = []
    for step in range(steps):
        picks = rng.integers(0, len(data), size=batch)
        starts = [int(rng.integers(0, data[i].shape[0] - window - 1)) for i in picks]
        x = torch.stack([data[i][s : s + window + 1] for i, s in zip(picks, starts)])
        if own_flags is not None:
            flags = np.stack(
                [own_flags(int(i), s, window + 1, rng) for i, s in zip(picks, starts)]
            )
            x = torch.cat([x, torch.as_tensor(flags, dtype=torch.float32)[..., None]], dim=-1)
        x = x.to(device)
        x[..., 1] = x[..., 1] / scale
        loss = next_event_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            history.append(float(loss.detach()))
    return history


@torch.no_grad()
def rollout(
    model: FlowReturnModel,
    context: np.ndarray,
    schedule: np.ndarray,
    side: int,
    *,
    uniforms: np.ndarray,
    normals: np.ndarray,
    clamp_signs: np.ndarray | None = None,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``schedule.size`` events after ``context``.

    ``context`` is ``[batch, time, 2]`` of (sign, raw return); with
    ``own_flag`` it is treated as containing no own trades. Positions where
    ``schedule`` is True are injected trades of sign ``side`` (flagged as own
    when the model has ``own_flag``); the model still generates their
    returns. ``uniforms`` and ``normals`` are ``[batch, schedule.size]``
    random draws supplied by the caller so that arms can share them.
    ``clamp_signs`` ``[batch, n_background]`` fixes the background signs
    (clamped mode); otherwise they are sampled from the model (free mode).

    Returns ``(signs, returns)``, both ``[batch, schedule.size]``, returns in
    raw units.
    """
    if side not in (-1, 1):
        raise ValueError("side must be +1 or -1")
    schedule = np.asarray(schedule, dtype=bool)
    scale = model.spec.return_scale
    flagged = model.spec.own_flag
    model.to(device).eval()
    ctx = torch.as_tensor(context, dtype=torch.float32, device=device).clone()
    ctx[..., 1] = ctx[..., 1] / scale
    if flagged:
        ctx = torch.cat([ctx, torch.zeros_like(ctx[..., :1])], dim=-1)
    u = torch.as_tensor(uniforms, dtype=torch.float32, device=device)
    z = torch.as_tensor(normals, dtype=torch.float32, device=device)
    clamp = None
    if clamp_signs is not None:
        clamp = torch.as_tensor(clamp_signs, dtype=torch.float32, device=device)
    logits, params, state = model(ctx)
    logit, param = logits[:, -1], params[:, -1]
    batch = ctx.shape[0]
    rows = torch.arange(batch, device=device)
    signs = torch.empty(batch, schedule.size, device=device)
    rets = torch.empty(batch, schedule.size, device=device)
    bg = 0
    for t in range(schedule.size):
        own = bool(schedule[t])
        if own:
            sign = torch.full((batch,), float(side), device=device)
        elif clamp is not None:
            sign = clamp[:, bg]
            bg += 1
        else:
            sign = torch.where(u[:, t] < torch.sigmoid(logit), 1.0, -1.0)
        chosen = param[rows, int(own and flagged), (sign > 0).long()]
        ret = chosen[:, 0] + chosen[:, 1].clamp(-6.0, 4.0).exp() * z[:, t]
        signs[:, t], rets[:, t] = sign, ret
        columns = [sign, ret]
        if flagged:
            columns.append(torch.full((batch,), float(own), device=device))
        logits, params, state = model(torch.stack(columns, dim=-1)[:, None, :], state)
        logit, param = logits[:, -1], params[:, -1]
    return signs.cpu().numpy(), (rets * scale).cpu().numpy()
