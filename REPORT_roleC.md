## Role C — Rollout-Style Planning Policy

For Role C, we implemented a rollout-style planning policy for the centralized dispatch environment. The policy does not modify the simulator or the frozen agent interface. It only implements the required `act(obs)` policy interface and selects valid actions using the provided action mask.

The planner evaluates valid assignment and charging actions. For assignment actions, it scores each drone-order pair using routed pickup distance, delivery distance, deadline risk, and battery feasibility. For charging actions, it prioritizes drones with low battery and avoids unnecessary charging for drones with sufficient state of charge.

We performed a rollout-depth ablation with depths 0, 1, and 2. Depth 0 behaves like the greedy baseline and mainly considers pickup distance. Depth 1 adds one-step planning terms such as delivery distance, deadline risk, and battery feasibility. Depth 2 adds a post-delivery charging-distance proxy.

The best result was obtained with depth 1. On seeds 0, 1, and 2 using `configs/eval_standard.yaml`, greedy_nearest achieved a cost_per_order of 4.5700, while our Role C planner with depth 1 achieved 2.9230. This shows that the planning terms improved dispatch decisions by increasing delivered orders, reducing dropped orders, improving on-time delivery rate, and reducing depletion events.

### Role C Results

| Method | cost_per_order | success_rate | ontime_rate | n_delivered | n_dropped | episode_return |
|---|---:|---:|---:|---:|---:|---:|
| random | 18.7804 | 0.6528 | 0.8901 | 39.67 | 21.67 | -168.33 |
| greedy_nearest | 4.5700 | 0.8549 | 0.9028 | 118.33 | 20.00 | 1183.26 |
| milp_rolling | 4.7223 | 0.8364 | 0.9109 | 118.00 | 23.00 | 1173.00 |
| Role C depth=0 | 4.5700 | 0.8549 | 0.9028 | 118.33 | 20.00 | 1183.26 |
| Role C depth=1 | 2.9230 | 0.8814 | 0.9815 | 126.33 | 17.00 | 1515.00 |
| Role C depth=2 | 3.3306 | 0.8691 | 0.9816 | 124.33 | 18.67 | 1443.73 |