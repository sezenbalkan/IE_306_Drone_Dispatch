from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from drone_dispatch_env.evaluate import evaluate

sys.path.append(str(Path(__file__).resolve().parent))
from dqn_agent import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/dqn_seed0.pt")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    policy = load_policy(args.weights)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    results = evaluate(policy, policy.cfg, seeds)
    print(json.dumps(results["mean"], indent=2))


if __name__ == "__main__":
    main()
