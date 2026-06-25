"""Generalized Advantage Estimation (GAE) shared by REINFORCE and A2C.

compute_gae implements the standard lambda-return advantage (Schulman et al.,
2016). It is written in the CleanRL style: a single reverse scan that handles
multiple episodes inside one buffer via the per-step `dones` flags (the
bootstrap term is zeroed at episode boundaries), plus a `last_value` bootstrap
for a rollout that is cut mid-episode (A2C n-step). For REINFORCE we pass
complete episodes, so `last_value=0`.
"""
from __future__ import annotations

import numpy as np


def compute_gae(rewards, values, dones, gamma: float, lam: float,
                last_value: float = 0.0):
    """rewards/values/dones: 1-D arrays length T. `dones[t]` is True if the
    episode ended *after* step t. Returns (advantages, returns), each length T."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)
