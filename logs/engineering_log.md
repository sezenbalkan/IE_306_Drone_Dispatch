# Engineering Log

## Day 2 - charging diagnosis

The first DQN run stayed close to random performance: `cost_per_order` was 17.62 versus 18.78 for random and 4.57 for `greedy_nearest`. The main symptom was `charger_utilization = 0.0` with 8 depletion events.

Checks:
- Action space includes charging actions. With the standard config, assignment actions are 0-159, charge actions are 160-167, and no-op is 168.
- The environment mask opens charge actions for idle drones with `soc < 1.0`.
- In a diagnostic rollout over seeds 0, 1, and 2, random selected 49 charge actions and `greedy_nearest` selected 79, so the environment can charge.
- The trained DQN selected 0 charge actions while charge actions were available on all 99 decision steps.

The structural checks ruled out an action-index or masking bug. A second review of the training log showed the actual root cause: the run ended after only 5,664 environment steps with epsilon still at 0.73. Learning started at step 1,000 and ran every four steps, so the network received only about 1,166 gradient updates and never reached a low-exploration phase. Therefore, zero charge actions were evidence of an under-trained policy, not proof that delayed charging reward was the main cause.

Fix:
- Use a fixed 60,000-step training budget so DQN variants are compared with equal environment interaction.
- Decay epsilon to 0.05 over 40,000 steps, leaving the final 20,000 steps for low-exploration training.
- Keep gamma at 0.99 and the simulator reward unchanged. The existing depletion penalty is already large enough to support charging without reward shaping.
- Log action types, charge availability, and cumulative gradient updates for each episode.

Result after the fix:
- Training completed 60,000 environment steps, epsilon reached 0.05, and the network received 14,751 gradient updates.
- Charging was learned, but the policy overused charge and no-op actions. Evaluation selected 1,039 charge actions and 683 no-op actions over seeds 0, 1, and 2.
- Depletion fell from 8.0 to 3.67, but `cost_per_order` worsened to 29.39 because success rate fell to 0.385 and dropped orders increased to 83.67.
- The final 100 training episodes had a mean return of -693.57, so longer training alone did not produce a useful vanilla DQN policy.

Next test: train Double DQN with the same 60,000-step budget. The fixed budget keeps the comparison fair, and Double DQN directly tests whether max-Q overestimation is contributing to the charge/no-op policy collapse.

## Double DQN check

Double DQN was trained with the same seed, network, exploration schedule, and 60,000-step budget. It did not fix the collapse:
- `cost_per_order` increased to 37.51.
- Success rate remained low at 0.382.
- The policy selected 183 charge actions and 750 no-op actions.
- Depletion increased to 6.67.

Double DQN reduced the extreme charging seen in vanilla DQN but shifted the policy toward no-op, so max-Q overestimation is not the only cause. The next controlled test is the Dueling architecture, which separates state value from action advantages and may help in the 169-action space where many assignment actions are state-dependent.

## Stability check

The three costs were compared directly: random 18.78, DQN 29.39, and Double DQN 37.51. Since both learned policies were worse than random, the next test targeted training stability before adding Dueling.

The visualizer compared DQN and `greedy_nearest` on seed 0. DQN needed 799 recorded decision frames while greedy needed 159. Its reward stream contained -1530 from dropped orders and -150 from depletion. The replay also showed many lost-drone frames and long periods of charging or deferring while orders accumulated.

First change:
- Reduce learning rate from 0.0005 to 0.0001.
- Divide rewards stored in replay by 10 while keeping environment rewards and evaluation metrics unchanged.
- Keep relative reward magnitudes instead of clipping them.

This did not improve the policy. The scaled run produced `cost_per_order = 33.61`, success rate 0.365, 371 charge actions, and 714 no-op actions.

A review of the state vector found that positions, deadlines, and the grid were normalized, but time was still passed to the network as a raw value from 0 to 500. This feature could dominate the other inputs. Time normalization was added as an optional checkpoint setting so old checkpoints keep their original preprocessing.

With `time / T_max`, the same learning rate, reward scale, seed, and 60,000-step budget:
- `cost_per_order` improved to 22.33.
- Success rate improved to 0.494.
- Mean dropped orders fell to 69.33.
- Episode return improved to -364.50.

Time normalization fixed part of the shared problem, but the policy still did not beat random. The last 100 training episodes still averaged -570.61 and contained 12,899 no-op actions. The next test should target temporal credit assignment instead of another value-head architecture.

## Long-run benchmark

Before starting a long run, training now logs `episode_return` and the official `cost_per_order` formula side by side. The 20,000-step benchmark produced a Pearson correlation of -0.961 between them. Higher return therefore strongly corresponds to lower cost in the observed episodes, so the optimization target is not visibly misaligned with the grading metric.

The benchmark took 46.973 seconds, or about 425.8 environment steps per second. Estimated training times on the same machine:
- 600,000 steps: 23.5 minutes
- 3,000,000 steps: 117.4 minutes
- 6,000,000 steps: 234.9 minutes

Checkpoint evaluation over seeds 0, 1, and 2 improved from `cost_per_order = 60.83` at 10,000 steps to 37.03 at 20,000 steps. Both are still immature policies. A 6,000,000-step run was selected with checkpoints every 500,000 steps. The measured runtime estimate is about 3 hours 55 minutes.

The long run was checked at 3,000,000 steps. Cost over the 500,000-step checkpoints was:
- 500K: 77.14
- 1M: 79.73
- 1.5M: 27.89
- 2M: 68.31
- 2.5M: 59.09
- 3M: 80.66

The best point was still worse than random at 18.78, and performance collapsed again after 1.5M. The policy at 3M selected 1,068 no-op actions over three evaluation seeds and delivered only 25 orders on average. Training was stopped at 3M because the issue is instability, not insufficient interaction.

Stopping also exposed a logging problem: the CSV was written only after a normal training exit. Training now writes and flushes each episode row immediately, so future interrupted long runs retain their learning curve.

## n-step returns (n=3) — 60k controlled check

### Step 1: masked_fill leak ruled out
Suspected bug: the bootstrap target `target_net(next_obs).masked_fill(~next_mask, -1e9).max()`
explodes if any non-terminal next state has an all-False action mask
(`reward + gamma*(-1e9)`). Checked `env_dispatch.py:285`: `mask[c.noop_index] = 1`
is set **unconditionally** in every state (`noop_index = n_actions - 1 = 168`).
So every state always has ≥1 valid action and the `-1e9` max can never be
selected. **The leak cannot happen — no fix needed.** Corollary: this rules the
masked_fill leak out as the cause of the earlier 3M-step divergence, so that
divergence is value instability, not a target bug.

### Step 2: what n-step changed
Added forward-view n-step returns (Sutton & Barto Ch. 7) in the replay path. A
per-episode deque accumulates `sum_k gamma^k r_{t+k}`; the transition stored for
state s_t bootstraps from the state `n` steps ahead with a per-sample discount
`gamma^window`. Windows are flushed at every episode boundary (terminal or
T_max timeout) so they never cross episodes — and since the MDP is finite-horizon
(T_max=500), the timeout IS a true terminal, so not bootstrapping past it is
correct, not merely convenient. `n` is a config param (`n_step`, default 1, which
reduces exactly to the previous 1-step DQN). The per-sample discount is applied to
both the vanilla and double-DQN target branches. Config: `configs/dqn_nstep.yaml`
(seed 0, n=3, normalize_time=true, reward_scale=10, lr=1e-4, 60k steps).
Aligned greedy eval (episode_return + cost_per_order + action mix) every 2000
steps on seeds 0,1,2 -> `logs/dqn_nstep_seed0_eval.csv`.

### Step 4: 60k verdict (vs random=18.78, greedy_nearest=4.57; before = n=1 normalized 60k: cost 22.33, return -364.5, mix a/c/n = 260/414/564)

1. **Does episode_return keep improving after eps=0.05? — NO, not steadily.**
   Post-decay (40k–60k) return oscillates hard: mean -135.7, range -861 to +457.
   It is *better* than before (mean -135.7 vs final -364.5) but it is not a clean
   climb — the value instability we saw at 3M is reduced, not eliminated.

2. **Does cost_per_order beat random? — Intermittently, yes.**
   15/30 eval points overall and 7/11 post-decay beat random (18.78). Best
   checkpoint (step 32k) hit cost = 6.77 / return = +677 — the first time this
   DQN family has come near greedy_nearest (4.57), and far past anything the n=1
   normalized 60k or the diverged 3M run ever reached. BUT the post-decay *mean*
   is 19.39 (≈ random) and the final 60k checkpoint is 20.29 (slightly worse than
   random). So it can beat random and frequently does, but not stably; the final
   weights are not trustworthy.

3. **Did the passive collapse reduce? — YES, clearly.**
   At the strong checkpoints the no-op collapse breaks: no-op count crashes from
   564 (before) to 51 (step 32k) and assignment rises 260 -> 345. n-step's main
   measurable win is killing the no-op collapse. Trade-off: at the unstable/bad
   eval points a new *over-charging* spike appears (final mix a/c/n = 263/834/657).

**Aligned metrics co-move (Step 3 confirmed).** episode_return and cost_per_order
track each other tightly across the eval log (step 32k: cost 6.77 / return +677;
step 22k: cost 87.3 / return -1338). The optimization target is not misaligned
with the graded metric; the problem is stability, not the wrong objective.

**Eval-seed note.** Eval seeds {0,1,2} coincide with the training reset seeds for
episodes 0-2 (`seed*100000 + episode`). That is 3 of 415 episodes, all in the
high-epsilon phase, so the leakage is negligible; the seeds were kept to match the
released before-baseline (also 0,1,2) for a fair comparison.

**Recommendation: NO-GO on 600k. Proceed to Double DQN (target stabilization)
instead.** The discriminating question is #1 (does return keep improving after
eps=0.05), and the answer is no: at convergence the policy oscillates with no
trend and *averages random* (post-decay mean cost 19.39; final checkpoint 20.29,
worse than random). The best point (6.77 at 32k) is a single variance spike
(30k=28.4, 32k=6.77, 34k=14.9), not a stable level. The project already ruled out
"more compute" as the fix (the 3M run diverged), so escalating this still-unstable
policy to 600k would repeat that mistake. n-step is nonetheless a real win on
*behavior* - it broke the no-op collapse (564 -> 51) and produced the first
checkpoints to approach greedy_nearest - which shows credit assignment was a
genuine bottleneck. But value stability is unaddressed (and a new over-charging
instability appeared). The correct next step is the already-planned **Double DQN**
for target stabilization; only escalate compute once stability is demonstrated.

## 600k run matrix (value-based DQN family)

At the user's request, ran the Role-A family to 600k on the validated base
(normalize_time, reward_scale=10, lr=1e-4, n_step=3 except the n=1 control),
seed 0, eval every 20k, checkpoint every 100k. Full table: logs/results_table.md.

| Method | best cost (step) | final | post-decay mean cost | mean return | pts<random | best mix a/c/n |
|---|---|---|---|---|---|---|
| DQN n=1 (control) | 13.96 (180k) | 45.63 | 34.72 | -847 | 3/29 | 325/619/365 |
| DQN n=3 | 13.33 (500k) | 20.67 | 23.61 | -475 | 5/29 | 297/190/355 |
| Double DQN n=3 | 10.19 (60k) | 14.90 | 20.22 | -301 | 12/29 | 327/388/104 |
| Dueling DQN n=3 | 21.65 (60k) | 26.07 | 38.25 | -936 | 0/29 | 219/947/967 |

Findings (baselines: random 18.78, greedy_nearest 4.57):
- **n-step helps at scale.** n=3 beats the n=1 control on every aggregate
  (post-decay mean cost 23.61 vs 34.72, return -475 vs -847, final 20.67 vs 45.63).
  The control degrades worst, confirming credit assignment was a real bottleneck.
- **No method beats random on the post-decay *mean*** (best is Double DQN, 20.22).
  Single best checkpoints do dip below random (Double 10.19, both DQNs ~13-14) but
  these are transient spikes, and final >> best everywhere (instability persists,
  same pattern as the 3M DQN divergence).
- **Double DQN n=3 is the clear best and the only improver.** Lowest mean cost,
  best return (-301), most points beating random (12/29), and the no-op collapse
  is most reduced (best-checkpoint mix assign 327 / noop 104). Crucially its eval
  trajectory *damps*: high-amplitude swings early (28->10->27->29) shrink to a
  ~15-21 band after 280k that repeatedly dips below random (300k:13.0, 500k:15.1,
  580k:15.1, 600k:14.9). This is qualitatively different from DQN n=3 (flat
  oscillation, no damping) and Dueling n=3 (diverged: over-charge 947, no-op 967,
  0/29 below random).

**Decision: escalate ONLY Double DQN n=3 to 3M** (per the rule "escalate improvers").
DQN n=3, the n=1 control, and Dueling n=3 show no damping/improvement and are not
escalated. The 3M check tests whether Double DQN's stabilization continues toward
beating random, or whether it diverges like the original 3M plain-DQN run did.

## 3M escalation of Double DQN n=3 (pre-registered)

Important correction to the success bar: the objective is to beat
**greedy_nearest = 4.57**, not random (18.78). Double DQN's 600k "damping band"
sits *at* random (second-half mean ~18), and its best spike (10.2) is still ~2x
greedy. "Below random" is the floor, not success. The 3M run is judged against
4.57, with random as a sanity floor only.

This is a fair test, not a repeat of the prior diverged 3M: that run was plain
DQN n=1; this is Double DQN + n=3, the most stable variant on the 600k evidence.
Config: configs/double_dqn_nstep_3m.yaml (seed 0, eval every 50k, checkpoint every
250k so the ~1.5M divergence-onset zone from the prior run is captured).

Pre-registered interpretation (the objective allows "beat greedy OR honestly
diagnose why it doesn't" -- the 600k data already points to the diagnosis branch):
- Diverges like the prior 3M -> strongest evidence that compute is not the fix;
  completes the diagnosis.
- Stabilizes below random but short of greedy (4.57) -> instability tamed by
  Double+n-step, but capacity/credit assignment still short of the target.
- Approaches 4.57 -> genuine win.

Banked regardless of outcome: n=3 beats the n=1 control on every 600k aggregate,
a clean defensible result that n-step helped. Oral-defense caveat: a checkpoint
may be validation-selected on our seeds, but grading uses ONE fixed policy on
held-out seeds/config, and the observed instability means the selected checkpoint
may not transfer.

### 3M result and verdict (pre-registered outcome: "stabilizes below random, short of greedy" -- confirmed, stronger than expected)

Double DQN n=3 at 3M (eval every 50k, checkpoint every 250k):
- Unstable warm-up to ~250k, then a long, GENUINELY STABLE good zone from ~1M to
  ~2.5M: cost stays 6.6-13 with positive episode returns. Best eval cost 6.62 at
  1.65M; best SAVED checkpoint 1.0M at cost 6.76.
- Post-decay mean cost 17.18 (beats random 18.78 on the MEAN -- a first for this
  family) and mean return -25 (~0, vs -301..-936 for every 600k run).
- Diverges after ~2.5M (final 3M cost 20.26, return -329), so unbounded compute
  still eventually destabilizes -- but it held stable to ~2.5M, far longer than the
  prior plain-DQN 3M run that was already gone by 2M.

Best submittable policy = weights/double_dqn_nstep_3m_step_1000000.pt
(logs/double_dqn_nstep_3m_best1M_eval.json, seeds 0,1,2):
cost_per_order 6.76, success_rate 0.749 (was 0.49), ontime 0.80, return +738,
no-op only 40 (the passive collapse is SOLVED in this policy), depletion 2.33.

**Verdict against the objective (beat greedy_nearest=4.57 OR diagnose why not):**
We did NOT beat greedy. Best achieved cost 6.76 (~1.48x greedy). This lands on
the diagnosis branch, with a clear, defensible story:
- The fixes worked and stack: time-normalization (input scaling) + n-step (credit
  assignment) + Double DQN (max-Q overestimation/target stability) together turned
  the passive-collapse policy (cost ~22-29, success 0.38-0.49, no-op-dominated)
  into a useful one (cost 6.76, success 0.75, assignment-dominated).
- Correction to an earlier claim: "more compute is ruled out" was true only for
  plain DQN n=1 (it diverges). For the stabilized variant, more compute DID help
  substantially up to ~1-2.5M -- pushing to 3M was the right call and surfaced the
  best policy. Compute is not free of limits, though: divergence returns past 2.5M.
- Remaining gap to 4.57 is attributable to model capacity/representation and
  residual late-training instability, NOT exploration or credit assignment (both
  now addressed). Plausible next levers (out of scope here): prioritized replay,
  a larger/again-tuned target-update period, or n>3 -- but the honest reading is
  that this value-based family plateaus around 1.4-1.5x greedy on this task.

Oral-defense caveat (unchanged): the 1M checkpoint is validation-selected on seeds
0,1,2; grading uses one fixed policy on held-out seeds/config, and the observed
late divergence means the selected checkpoint carries transfer risk. We submit the
1M checkpoint as the best stable point, not the final-step weights.
