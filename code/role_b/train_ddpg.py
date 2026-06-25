"""CLI: train DDPG on DroneControl-v0.

    python code/role_b/train_ddpg.py --config configs/ddpg.yaml --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # put code/ on the path -> import role_b

from role_b.ddpg import train                   # noqa: E402
from role_b.utils import load_experiment         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default=None)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    cfg, params = load_experiment(args.config)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    log = args.log or f"logs/ddpg_seed{args.seed}.csv"
    weights = args.weights or f"weights/ddpg_seed{args.seed}.pt"
    res = train(cfg, params, args.seed, log, weights)
    print(f"DONE ddpg seed={args.seed} best_return={res['best_return']:.2f} -> {weights}")


if __name__ == "__main__":
    main()
