"""Shared utilities: seeding, config loading, feature batching, CSV logging, eval."""
from __future__ import annotations

import csv
import os
import random
from typing import Optional

import numpy as np
import torch
import yaml

from drone_dispatch_env.config import Config
from drone_dispatch_env.evaluate import evaluate


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_experiment(path: str) -> tuple[Config, dict]:
    """Load an experiment YAML. If it has a top-level `env:` block, that becomes
    the simulator Config and the rest are algorithm hyperparameters; otherwise the
    whole file is treated as an env config (so plain eval configs also load)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if "env" in raw:
        cfg = Config.from_dict(raw["env"])
        params = {k: v for k, v in raw.items() if k != "env"}
    else:
        cfg = Config.from_dict(raw)
        params = {}
    return cfg, params


_FEATURE_KEYS = ("drone", "order", "pair", "global", "drone_alive", "order_valid", "mask")


def batch_features(feat_list: list[dict], device: torch.device) -> dict:
    """Stack a list of per-state feature dicts into batched tensors (B, ...)."""
    out = {}
    for k in _FEATURE_KEYS:
        arr = np.stack([f[k] for f in feat_list], axis=0)
        out[k] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return out


def single_batch(feat: dict, device: torch.device) -> dict:
    return batch_features([feat], device)


class CSVLogger:
    """Append rows to a CSV, writing the header from the first row's keys."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fields: Optional[list[str]] = None
        self._fh = open(path, "w", newline="")
        self._writer: Optional[csv.DictWriter] = None

    def log(self, row: dict) -> None:
        if self._writer is None:
            self._fields = list(row.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def evaluate_dispatch_agent(agent, cfg: Config, seeds) -> dict:
    """Mean metrics dict for a dispatch Policy over `seeds` (uses the frozen harness)."""
    return evaluate(agent, cfg, seeds=list(seeds))["mean"]
