"""Launch the full 3-seed training for all Role B methods in parallel.

Each (method, seed) runs as its own process (1 torch thread each), so the seeds
train concurrently. Per-run stdout goes to logs/run_<method>_seed<seed>.out.

    python code/role_b/run_training.py --methods reinforce,a2c,ddpg --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

JOBS = {
    "reinforce": ("code/role_b/train_reinforce.py", "configs/reinforce.yaml"),
    "a2c": ("code/role_b/train_a2c.py", "configs/a2c.yaml"),
    "ddpg": ("code/role_b/train_ddpg.py", "configs/ddpg.yaml"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="reinforce,a2c,ddpg")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)
    procs = []
    for m in args.methods.split(","):
        script, cfg = JOBS[m]
        for s in args.seeds.split(","):
            out = open(f"logs/run_{m}_seed{s}.out", "w")
            p = subprocess.Popen([sys.executable, script, "--config", cfg, "--seed", s],
                                 stdout=out, stderr=subprocess.STDOUT)
            procs.append((m, s, p, out))
            print(f"launched {m} seed {s} (pid {p.pid})", flush=True)

    failures = 0
    for m, s, p, out in procs:
        rc = p.wait()
        out.close()
        print(f"finished {m} seed {s} rc={rc}", flush=True)
        failures += int(rc != 0)

    print(f"ALL TRAINING DONE failures={failures}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
