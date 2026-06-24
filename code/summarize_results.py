"""Build logs/results_table.md from the eval CSVs of the 600k/3M runs.

Reusable: pass run specs as (label, eval_csv) and it computes best/final/post-decay
cost and return, the best-checkpoint action mix, and whether it beats random.
Baselines (random=18.78, greedy_nearest=4.57) are fixed reference points.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

RANDOM = 18.78
GREEDY = 4.57
POST_DECAY_STEP = 40000  # epsilon hits 0.05 at epsilon_decay_steps

# (label, eval_csv). Edit/extend as runs complete.
RUNS = [
    ("DQN  n=1 (control)", "logs/dqn_n1_600k_eval.csv"),
    ("DQN  n=3",           "logs/dqn_nstep_600k_eval.csv"),
    ("Double DQN  n=3",    "logs/double_dqn_nstep_600k_eval.csv"),
    ("Dueling DQN n=3",    "logs/dueling_dqn_nstep_600k_eval.csv"),
    ("Double DQN n=3 (3M)", "logs/double_dqn_nstep_3m_eval.csv"),
    ("Double DQN n=3 (6M)", "logs/double_dqn_nstep_6m_eval.csv"),
]


def summarize(eval_csv: str):
    rows = list(csv.DictReader(open(eval_csv)))
    if not rows:
        return None
    cost = [float(r["eval_cost_per_order"]) for r in rows]
    post = [r for r in rows if int(r["step"]) >= POST_DECAY_STEP] or rows
    pc = [float(r["eval_cost_per_order"]) for r in post]
    pr = [float(r["eval_episode_return"]) for r in post]
    bi = cost.index(min(cost))
    best = rows[bi]
    final = rows[-1]
    return {
        "steps": int(final["step"]),
        "best_cost": min(cost),
        "best_step": int(best["step"]),
        "best_mix": f'{best["eval_assign"]}/{best["eval_charge"]}/{best["eval_noop"]}',
        "final_cost": float(final["eval_cost_per_order"]),
        "pd_mean_cost": sum(pc) / len(pc),
        "pd_mean_return": sum(pr) / len(pr),
        "beats_random_pts": f"{sum(c < RANDOM for c in pc)}/{len(pc)}",
    }


def main():
    lines = [
        "# Results table (value-based DQN family)",
        "",
        f"Baselines: random = {RANDOM}, greedy_nearest = {GREEDY} (target to beat).",
        "Post-decay = eval points at epsilon=0.05 (step >= 40k). Mix = assign/charge/noop "
        "at the best checkpoint. Lower cost is better.",
        "",
        "| Method | steps | best cost (step) | final cost | post-decay mean cost | "
        "post-decay mean return | post-decay pts < random | best-ckpt mix a/c/n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, csv_path in RUNS:
        if not Path(csv_path).exists():
            lines.append(f"| {label} | _pending_ | | | | | | |")
            continue
        s = summarize(csv_path)
        if s is None:
            lines.append(f"| {label} | _empty_ | | | | | | |")
            continue
        lines.append(
            f"| {label} | {s['steps']//1000}k | {s['best_cost']:.2f} ({s['best_step']//1000}k) "
            f"| {s['final_cost']:.2f} | {s['pd_mean_cost']:.2f} | {s['pd_mean_return']:.0f} "
            f"| {s['beats_random_pts']} | {s['best_mix']} |"
        )
    out = "\n".join(lines) + "\n"
    Path("logs/results_table.md").write_text(out, encoding="utf-8")
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
