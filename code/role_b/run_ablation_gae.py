"""GAE-lambda ablation (Role B required ablation).

Sweeps the GAE lambda on **A2C** (the stable method, so the lambda effect is not
swamped by Monte-Carlo seed variance) across a couple of seeds on a reduced step
budget, recording per-run learning curves + final validation cost_per_order. Runs
the sweep points in parallel processes.

    python code/role_b/run_ablation_gae.py --config configs/ablation_gae.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # put code/ on the path -> import role_b

from role_b import a2c                          # noqa: E402
from role_b.utils import load_experiment        # noqa: E402


def _tag(lam: float) -> str:
    return ("lam" + str(lam)).replace(".", "p")


def _worker(task):
    lam, seed, cfg, params, log, weights = task
    import torch
    torch.set_num_threads(1)
    p = dict(params)
    p["gae_lambda"] = lam
    res = a2c.train(cfg, p, seed, log, weights)
    return (lam, seed, res["best_cost"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ablation_gae.yaml")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    cfg, params = load_experiment(args.config)
    lambdas = params.pop("lambdas", [0.0, 0.9, 0.95, 0.99, 1.0])
    seeds = params.pop("seeds", [0, 1])
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    # task = (lambda, seed, cfg, params, log_path, weight_path)
    tasks = [(lam, seed, cfg, params,
              f"logs/ablation_gae_{_tag(lam)}_seed{seed}.csv",
              f"weights/_ablation_{_tag(lam)}_seed{seed}.pt")
             for lam in lambdas for seed in seeds]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_worker, tasks):
            results.append(r)
            print(f"done lambda={r[0]} seed={r[1]} best_cost={r[2]:.3f}", flush=True)

    print("\nABLATION SUMMARY (lambda, seed, best_cost):")
    for r in sorted(results):
        print(f"  lambda={r[0]:<5} seed={r[1]} best_cost={r[2]:.3f}")


if __name__ == "__main__":
    main()
