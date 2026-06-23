from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from drone_dispatch_env.config import Config


def obs_to_vector(obs: dict, cfg: Config) -> np.ndarray:
    drones = np.asarray(obs["drones"], dtype=np.float32).copy()
    drones[:, 0] /= max(cfg.H - 1, 1)
    drones[:, 1] /= max(cfg.W - 1, 1)

    orders = np.asarray(obs["orders"], dtype=np.float32).copy()
    orders[:, [0, 2]] /= max(cfg.H - 1, 1)
    orders[:, [1, 3]] /= max(cfg.W - 1, 1)
    orders[:, 4] /= max(cfg.sla_steps, 1)

    grid = np.asarray(obs["grid"], dtype=np.float32).reshape(-1) / 3.0
    time = np.asarray(obs["time"], dtype=np.float32).reshape(-1)
    return np.concatenate([drones.reshape(-1), orders.reshape(-1), grid, time]).astype(np.float32)


def obs_dim(cfg: Config) -> int:
    return cfg.n_drones * 10 + cfg.k_max * 5 + cfg.H * cfg.W + 1


class QNetwork(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden_sizes: Iterable[int], dueling: bool = False):
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        layers: list[nn.Module] = []
        last = in_dim
        for size in hidden_sizes:
            layers += [nn.Linear(last, size), nn.ReLU()]
            last = size
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(last, 1)
            self.advantage = nn.Linear(last, n_actions)
        else:
            self.head = nn.Linear(last, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        if not self.dueling:
            return self.head(z)
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DQNPolicy:
    def __init__(self, cfg: Config, q_net: QNetwork, device: str = "cpu"):
        self.cfg = cfg
        self.q_net = q_net.to(device)
        self.device = torch.device(device)
        self.q_net.eval()

    def q_values(self, obs: dict) -> np.ndarray:
        x = torch.as_tensor(obs_to_vector(obs, self.cfg), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(x).squeeze(0).cpu().numpy()

    def act(self, obs: dict) -> int:
        mask = np.asarray(obs["action_mask"], dtype=bool)
        q = np.where(mask, self.q_values(obs), -np.inf)
        return int(np.argmax(q))

    def action_values(self, obs: dict):
        mask = np.asarray(obs["action_mask"], dtype=bool)
        return np.where(mask, self.q_values(obs), np.nan)

    def action_probs(self, obs: dict):
        return None

    def state_values(self, obs: dict):
        return None


def build_q_net(cfg: Config, hidden_sizes: Iterable[int], dueling: bool = False) -> QNetwork:
    return QNetwork(obs_dim(cfg), cfg.n_actions, hidden_sizes, dueling=dueling)


def save_checkpoint(path: str | Path, cfg: Config, q_net: QNetwork, train_config: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "env_config": asdict(cfg),
            "train_config": train_config,
            "model_state": q_net.state_dict(),
        },
        path,
    )


def load_policy(path: str | Path, device: str = "cpu") -> DQNPolicy:
    checkpoint = torch.load(path, map_location=device)
    cfg = Config.from_dict(checkpoint["env_config"])
    train_config = checkpoint["train_config"]
    q_net = build_q_net(
        cfg,
        train_config.get("hidden_sizes", [256, 256]),
        dueling=bool(train_config.get("dueling", False)),
    )
    q_net.load_state_dict(checkpoint["model_state"])
    return DQNPolicy(cfg, q_net, device=device)
