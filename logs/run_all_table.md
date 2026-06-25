# run_all results table

Eval config = configs/eval_standard.yaml | seeds = [0, 1, 2] | primary metric = cost_per_order (lower better)

| policy | cost/order (mean±std) | success rate |
|---|---|---|
| random | 18.78 ± 1.27 | 0.653 |
| greedy_nearest | 4.57 ± 0.85 | 0.855 |
| milp_rolling | 4.72 ± 1.38 | 0.836 |
| DQN n=3 | 20.67 ± 6.57 | 0.495 |
| Double DQN n=3 | 6.76 ± 1.80 | 0.749 |
| Dueling DQN n=3 | 26.07 ± 3.49 | 0.428 |
| Offline CQL (joint) | 8.42 ± 1.91 | 0.680 |

**Joint multi-agent** (DroneDispatchMA-v0, separate env): cost_per_order = 6.49, delivered/ep = 100.7, return = 793.8.
