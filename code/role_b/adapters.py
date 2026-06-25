"""Adapters that satisfy the frozen agent_interface.Policy / Introspectable API.

PolicyGradientAgent wraps a trained FactoredActorCritic so REINFORCE/A2C policies
plug straight into evaluate() and the visualizer. GoStraight is the comparison
baseline for the continuous DroneControl-v0 sub-env (DDPG's opponent).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from drone_dispatch_env.config import Config
from .features import RoutedCache, extract_features
from .networks import FactoredActorCritic
from .utils import single_batch


class PolicyGradientAgent:
    """Wrap a FactoredActorCritic as a dispatch Policy (+ action-prob overlay)."""

    def __init__(self, net: FactoredActorCritic, cfg: Config,
                 device: Optional[torch.device] = None, deterministic: bool = True):
        self.net = net
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.deterministic = deterministic
        self.cache = RoutedCache(cfg.neighborhood)
        self.net.eval()

    def _logits(self, obs) -> torch.Tensor:
        feat = extract_features(obs, self.cfg, self.cache)
        b = single_batch(feat, self.device)
        with torch.no_grad():
            logits, _ = self.net(b)
        return logits[0]

    def act(self, obs):
        logits = self._logits(obs)
        if self.deterministic:
            return int(torch.argmax(logits).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def action_probs(self, obs) -> Optional[np.ndarray]:
        logits = self._logits(obs)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def action_values(self, obs):
        return None

    def state_values(self, obs):
        return None


def load_dispatch_agent(weight_path: str, cfg: Config,
                        device: Optional[torch.device] = None,
                        hidden: int = 128) -> PolicyGradientAgent:
    device = device or torch.device("cpu")
    ckpt = torch.load(weight_path, map_location=device)
    hidden = ckpt.get("hidden", hidden) if isinstance(ckpt, dict) and "state_dict" in ckpt else hidden
    net = FactoredActorCritic(hidden=hidden)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(state)
    net.to(device)
    return PolicyGradientAgent(net, cfg, device, deterministic=True)


class GoStraight:
    """Full-speed straight-line controller for DroneControl-v0 (DDPG baseline).

    Heads directly at the target each step; ignores no-fly geometry and energy, so
    it stalls against walls — the gap a learned DDPG controller should close.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        diff_x = obs[0] * self.cfg.H      # (target_x - pos_x)
        diff_y = obs[1] * self.cfg.W      # (target_y - pos_y)
        heading = float(obs[4])
        desired = float(np.arctan2(diff_y, diff_x))
        delta = (desired - heading + np.pi) % (2 * np.pi) - np.pi
        return np.array([1.0, float(np.clip(delta / np.pi, -1.0, 1.0))], dtype=np.float32)
