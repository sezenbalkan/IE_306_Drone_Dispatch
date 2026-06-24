# 3-seed results (seeds 0, 1, 2) — mean ± std

Baselines: random 18.78, greedy_nearest 4.57. Lower cost is better. 'best' = best eval checkpoint; 'final' = last-step weights; 'post-decay' = mean over epsilon=0.05 eval points.

| Method | best cost | final cost | post-decay mean |
|---|---|---|---|
| DQN n=3 (600k) | 13.87 ± 0.71 | 19.37 ± 1.19 | 22.65 ± 0.79 |
| Dueling DQN n=3 (600k) | 13.17 ± 6.43 | 21.80 ± 3.02 | 28.87 ± 6.95 |
| Double DQN n=3 (600k) | 9.97 ± 0.18 | 15.81 ± 1.53 | 19.51 ± 0.72 |
| Double DQN n=3 (3M) | 6.39 ± 0.41 | 31.00 ± 13.38 | 16.24 ± 1.32 |
