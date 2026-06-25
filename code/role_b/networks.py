"""Neural networks for Role B.

FactoredActorCritic — a permutation-invariant, dimension-robust policy/value net
for the discrete dispatcher. Logits are produced by *shared* per-action heads
(one assign-head applied to every drone-order pair, one charge-head per drone, a
state-conditioned noop bias), so the parameter shapes never depend on
n_drones/k_max/n_actions and a trained model loads under any config. The critic
is a Deep Sets encoder (shared per-entity MLP + masked mean/max pooling).

DDPGActor / DDPGCritic — plain MLPs for the continuous DroneControl-v0 sub-env.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import FD, FO, FP, FG

_NEG = -1.0e8


def mlp(sizes, activation=nn.Tanh, out_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(activation() if i < len(sizes) - 2 else out_activation())
    return nn.Sequential(*layers)


def _masked_pool(enc: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Masked mean+max pool over dim=1. enc:(B,N,h) valid:(B,N) -> (B,2h)."""
    v = valid.unsqueeze(-1)                                  # (B,N,1)
    cnt = v.sum(dim=1).clamp(min=1.0)                        # (B,1)
    mean = (enc * v).sum(dim=1) / cnt                        # (B,h)
    filled = enc.masked_fill(v < 0.5, _NEG)
    mx = filled.max(dim=1).values                           # (B,h)
    has = (valid.sum(dim=1, keepdim=True) > 0).float()
    mx = mx * has                                           # zero rows with no valid entries
    return torch.cat([mean, mx], dim=-1)


class FactoredActorCritic(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.assign_head = mlp([FD + FO + FP, hidden, hidden, 1])
        self.charge_head = mlp([FD, hidden, 1])
        self.noop_head = mlp([FG, hidden, 1])
        self.phi_d = mlp([FD, hidden, hidden])
        self.phi_o = mlp([FO, hidden, hidden])
        self.v_head = mlp([4 * hidden + FG, hidden, 1])

    def forward(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        drone, order, pair = b["drone"], b["order"], b["pair"]
        glob, mask = b["global"], b["mask"]
        B, n_drones, k_max, _ = pair.shape

        drone_exp = drone.unsqueeze(2).expand(B, n_drones, k_max, drone.shape[-1])
        order_exp = order.unsqueeze(1).expand(B, n_drones, k_max, order.shape[-1])
        assign_in = torch.cat([drone_exp, order_exp, pair], dim=-1)
        assign_logits = self.assign_head(assign_in).squeeze(-1)        # (B,n_drones,k_max)
        assign_flat = assign_logits.reshape(B, n_drones * k_max)       # index = d*k_max+s

        charge_logits = self.charge_head(drone).squeeze(-1)            # (B,n_drones)
        noop_logit = self.noop_head(glob)                             # (B,1)

        logits = torch.cat([assign_flat, charge_logits, noop_logit], dim=-1)
        logits = torch.where(mask > 0.5, logits, torch.full_like(logits, _NEG))

        d_enc = self.phi_d(drone)
        o_enc = self.phi_o(order)
        d_pool = _masked_pool(d_enc, b["drone_alive"])
        o_pool = _masked_pool(o_enc, b["order_valid"])
        value = self.v_head(torch.cat([d_pool, o_pool, glob], dim=-1)).squeeze(-1)
        return logits, value

    def distribution(self, b: dict):
        logits, value = self.forward(b)
        return torch.distributions.Categorical(logits=logits), value

    def evaluate_actions(self, b: dict, actions: torch.Tensor):
        dist, value = self.distribution(b)
        return dist.log_prob(actions), dist.entropy(), value


class DDPGActor(nn.Module):
    """Maps obs -> action; speed in [0,1] (sigmoid), heading_delta in [-1,1] (tanh)."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.body = mlp([obs_dim, hidden, hidden, 2], activation=nn.ReLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        # min speed 0.3: a delivery drone must keep moving, which removes the
        # degenerate "hover in place" local optimum and forces it to learn to
        # steer toward the target (0.3 < the 0.7-cell reach radius, so it can
        # still arrive).
        speed = 0.3 + 0.7 * torch.sigmoid(out[..., 0:1])
        heading = torch.tanh(out[..., 1:2])
        return torch.cat([speed, heading], dim=-1)


class DDPGCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.body = mlp([obs_dim + act_dim, hidden, hidden, 1], activation=nn.ReLU)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([x, a], dim=-1)).squeeze(-1)
