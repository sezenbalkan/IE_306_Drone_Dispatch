"""3-seed aggregation: read each method's per-seed eval CSVs and report
mean +/- std of best / final / post-decay cost_per_order across seeds {0,1,2}.

This backs deliverable #3 (learning curves over >=3 seeds, not a single lucky
run). Writes logs/results_seeds.md.
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

POST_DECAY = 40000  # epsilon hits 0.05 here

# method label -> (seed0_csv, seed1_csv, seed2_csv)
METHODS = {
    "DQN n=3 (600k)": (
        "logs/dqn_nstep_600k_eval.csv",
        "logs/dqn_nstep_600k_seed1_eval.csv",
        "logs/dqn_nstep_600k_seed2_eval.csv"),
    "Dueling DQN n=3 (600k)": (
        "logs/dueling_dqn_nstep_600k_eval.csv",
        "logs/dueling_dqn_nstep_600k_seed1_eval.csv",
        "logs/dueling_dqn_nstep_600k_seed2_eval.csv"),
    "Double DQN n=3 (600k)": (
        "logs/double_dqn_nstep_600k_eval.csv",
        "logs/double_dqn_nstep_600k_seed1_eval.csv",
        "logs/double_dqn_nstep_600k_seed2_eval.csv"),
    "Double DQN n=3 (3M)": (
        "logs/double_dqn_nstep_3m_eval.csv",
        "logs/double_dqn_nstep_3m_seed1_eval.csv",
        "logs/double_dqn_nstep_3m_seed2_eval.csv"),
}


def per_seed(csv_path: str):
    rows = list(csv.DictReader(open(csv_path)))
    cost = [float(r["eval_cost_per_order"]) for r in rows]
    post = [float(r["eval_cost_per_order"]) for r in rows
            if int(r["step"]) >= POST_DECAY] or cost
    return min(cost), cost[-1], sum(post) / len(post)


def ms(xs):
    return f"{st.mean(xs):.2f} ± {st.pstdev(xs):.2f}"


def main():
    lines = [
        "# 3-seed results (seeds 0, 1, 2) — mean ± std",
        "",
        "Baselines: random 18.78, greedy_nearest 4.57. Lower cost is better. "
        "'best' = best eval checkpoint; 'final' = last-step weights; 'post-decay' = "
        "mean over epsilon=0.05 eval points.",
        "",
        "| Method | best cost | final cost | post-decay mean |",
        "|---|---|---|---|",
    ]
    for label, files in METHODS.items():
        files = [f for f in files if Path(f).exists()]
        if len(files) < 3:
            lines.append(f"| {label} | _missing ({len(files)}/3)_ | | |")
            continue
        bests, finals, posts = zip(*[per_seed(f) for f in files])
        lines.append(f"| {label} | {ms(bests)} | {ms(finals)} | {ms(posts)} |")
    out = "\n".join(lines) + "\n"
    Path("logs/results_seeds.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
