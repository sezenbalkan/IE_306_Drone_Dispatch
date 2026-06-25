"""CLI: train REINFORCE+GAE on DroneDispatch-v0.

    python code/role_b/train_reinforce.py --config configs/reinforce.yaml --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # put code/ on the path -> import role_b

from role_b.reinforce_gae import train          # noqa: E402
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
    log = args.log or f"logs/reinforce_seed{args.seed}.csv"
    weights = args.weights or f"weights/reinforce_seed{args.seed}.pt"
    res = train(cfg, params, args.seed, log, weights)
    print(f"DONE reinforce seed={args.seed} best_cost={res['best_cost']:.3f} -> {weights}")


if __name__ == "__main__":
    main()
