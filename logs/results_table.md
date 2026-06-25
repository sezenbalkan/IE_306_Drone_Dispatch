# Results table (value-based DQN family)

Baselines: random = 18.78, greedy_nearest = 4.57 (target to beat).
Post-decay = eval points at epsilon=0.05 (step >= 40k). Mix = assign/charge/noop at the best checkpoint. Lower cost is better.

| Method | steps | best cost (step) | final cost | post-decay mean cost | post-decay mean return | post-decay pts < random | best-ckpt mix a/c/n |
|---|---|---|---|---|---|---|---|
| DQN  n=1 (control) | 600k | 13.96 (180k) | 45.63 | 34.72 | -847 | 3/29 | 325/619/365 |
| DQN  n=3 | 600k | 13.33 (500k) | 20.67 | 23.61 | -475 | 5/29 | 297/190/355 |
| Double DQN  n=3 | 600k | 10.19 (60k) | 14.90 | 20.22 | -301 | 12/29 | 327/388/104 |
| Dueling DQN n=3 | 600k | 21.65 (60k) | 26.07 | 38.25 | -936 | 0/29 | 219/947/967 |
| Double DQN n=3 (3M) | 3000k | 6.62 (1650k) | 20.26 | 17.18 | -25 | 44/60 | 362/152/17 |
| Double DQN n=3 (6M) | 6000k | 6.62 (1650k) | 53.06 | 32.65 | -577 | 45/120 | 362/152/17 |
