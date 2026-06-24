from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

sys.path.append(str(Path(__file__).resolve().parent))

from dqn_agent import DQNPolicy, build_q_net, obs_to_vector, save_checkpoint
from drone_dispatch_env.config import Config
from drone_dispatch_env.env_dispatch import DroneDispatchEnv
from drone_dispatch_env.evaluate import evaluate


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def add(self, obs, action, nstep_return, next_obs, next_mask, done, discount):
        # `nstep_return` is the accumulated discounted reward over the n-step
        # window; `discount` is gamma**window_len, the factor that multiplies the
        # bootstrapped value of `next_obs` (the state at the end of the window).
        self.data.append((obs, action, nstep_return, next_obs, next_mask, done, discount))

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        obs, actions, returns, next_obs, next_masks, dones, discounts = zip(*batch)
        return (
            np.stack(obs),
            np.asarray(actions, dtype=np.int64),
            np.asarray(returns, dtype=np.float32),
            np.stack(next_obs),
            np.stack(next_masks),
            np.asarray(dones, dtype=np.float32),
            np.asarray(discounts, dtype=np.float32),
        )

    def __len__(self):
        return len(self.data)


# n-step returns (Sutton & Barto, Reinforcement Learning 2nd ed., Ch. 7).
# Forward view: a transition's target is sum_{k=0}^{m-1} gamma^k * r_{t+k}
# bootstrapped from the state m steps ahead with factor gamma^m. Windows are
# truncated at episode boundaries (terminal or T_max timeout) so they never
# cross episodes. The env is finite-horizon (T_max), so the timeout IS a true
# terminal: we do not bootstrap past it (done=True), which also matches the
# existing 1-step convention.
def emit_nstep(nstep_queue, next_obs_vec, next_mask, done, gamma, flush):
    """Yield (obs, action, return, next_obs, next_mask, done, discount) tuples.

    When `flush` is False, emit only the oldest transition once the window is
    full (len == n). When True (episode ended), drain the whole queue, emitting
    one (progressively shorter) truncated-return transition per remaining item.
    """
    while nstep_queue:
        obs_vec, action, _ = nstep_queue[0]
        ret = 0.0
        for k, (_, _, r) in enumerate(nstep_queue):
            ret += (gamma ** k) * r
        discount = gamma ** len(nstep_queue)
        yield (obs_vec, action, ret, next_obs_vec, next_mask, done, discount)
        nstep_queue.popleft()
        if not flush:
            break


def epsilon_by_step(step: int, cfg: dict) -> float:
    start = float(cfg["epsilon_start"])
    end = float(cfg["epsilon_end"])
    decay = max(int(cfg["epsilon_decay_steps"]), 1)
    frac = min(step / decay, 1.0)
    return start + frac * (end - start)


def choose_action(q_net, obs, cfg: Config, eps: float, device: torch.device, normalize_time: bool) -> int:
    mask = np.asarray(obs["action_mask"], dtype=bool)
    valid = np.flatnonzero(mask)
    if random.random() < eps:
        return int(np.random.choice(valid))
    x = torch.as_tensor(
        obs_to_vector(obs, cfg, normalize_time),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        q = q_net(x).squeeze(0).cpu().numpy()
    return int(np.argmax(np.where(mask, q, -np.inf)))


def episode_cost(stats: dict) -> float:
    cost = stats["energy"] + stats["late_cost"] + stats["drop_cost"] + stats["depletion_cost"]
    return cost / max(stats["delivered"], 1)


def greedy_eval(q_net, env_cfg: Config, device, normalize_time: bool, seeds):
    """Greedy (no-epsilon) evaluation via the shipped evaluate() util.

    Returns mean cost_per_order, mean episode_return (the aligned pair) plus the
    action mix, so we can confirm the optimized return and the graded cost move
    together and watch whether the passive charge/no-op collapse shrinks.
    """
    was_training = q_net.training
    policy = DQNPolicy(env_cfg, q_net, device=str(device), normalize_time=normalize_time)
    mean = evaluate(policy, env_cfg, seeds)["mean"]
    counts = {"assign": 0, "charge": 0, "noop": 0}
    for seed in seeds:
        env = DroneDispatchEnv(env_cfg)
        obs, _ = env.reset(seed=int(seed))
        done = False
        while not done:
            action = policy.act(obs)
            counts[env_cfg.decode(action)[0]] += 1
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    if was_training:
        q_net.train()
    return mean["cost_per_order"], mean["episode_return"], counts


def checkpoint_path(weight_path: str, step: int) -> Path:
    path = Path(weight_path)
    return path.with_name(f"{path.stem}_step_{step}{path.suffix}")


def train(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    seed = int(train_cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = Config.from_yaml(train_cfg["env_config"])
    env = DroneDispatchEnv(env_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q_net = build_q_net(env_cfg, train_cfg["hidden_sizes"], dueling=bool(train_cfg["dueling"])).to(device)
    target_net = build_q_net(env_cfg, train_cfg["hidden_sizes"], dueling=bool(train_cfg["dueling"])).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=float(train_cfg["learning_rate"]))
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(int(train_cfg["buffer_size"]))
    reward_scale = float(train_cfg.get("reward_scale", 1.0))
    if reward_scale <= 0:
        raise ValueError("reward_scale must be positive")
    normalize_time = bool(train_cfg.get("normalize_time", False))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 0))
    print_every_episodes = max(int(train_cfg.get("print_every_episodes", 1)), 1)
    n_step = max(int(train_cfg.get("n_step", 1)), 1)
    gamma = float(train_cfg["gamma"])
    eval_every = int(train_cfg.get("eval_every", 0))
    eval_seeds = [int(s) for s in train_cfg.get("eval_seeds", [])]

    Path(train_cfg["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(train_cfg["weight_path"]).parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    train_updates = 0
    episode = 0
    fieldnames = [
        "episode",
        "step",
        "episode_return",
        "cost_per_order",
        "delivered",
        "dropped",
        "epsilon",
        "loss",
        "assign_actions",
        "charge_actions",
        "noop_actions",
        "charge_open_steps",
        "train_updates",
    ]
    log_file = open(train_cfg["log_path"], "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()
    log_file.flush()

    # Aligned greedy-eval log: episode_return (optimized) and cost_per_order
    # (graded) side by side, plus the action mix, sampled every `eval_every` steps.
    eval_writer = None
    eval_file = None
    if eval_every > 0 and eval_seeds:
        eval_path = train_cfg.get("eval_log_path", str(Path(train_cfg["log_path"]).with_name(
            Path(train_cfg["log_path"]).stem + "_eval.csv")))
        eval_file = open(eval_path, "w", newline="", encoding="utf-8")
        eval_writer = csv.DictWriter(eval_file, fieldnames=[
            "step", "epsilon", "eval_cost_per_order", "eval_episode_return",
            "eval_assign", "eval_charge", "eval_noop",
        ])
        eval_writer.writeheader()
        eval_file.flush()

    total_steps = int(train_cfg["total_steps"])
    while global_step < total_steps:
        obs, _ = env.reset(seed=seed * 100000 + episode)
        ep_return = 0.0
        last_loss = ""
        done = False
        action_counts = {"assign": 0, "charge": 0, "noop": 0}
        charge_open_steps = 0
        nstep_queue = deque()  # holds (obs_vec, action, scaled_reward) within one episode
        for _ in range(int(train_cfg["max_steps_per_episode"])):
            eps = epsilon_by_step(global_step, train_cfg)
            charge_indices = [env_cfg.charge_index(d) for d in range(env_cfg.n_drones)]
            if np.asarray(obs["action_mask"], dtype=bool)[charge_indices].any():
                charge_open_steps += 1
            action = choose_action(q_net, obs, env_cfg, eps, device, normalize_time)
            action_counts[env_cfg.decode(action)[0]] += 1
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            nstep_queue.append((obs_to_vector(obs, env_cfg, normalize_time), action, reward / reward_scale))
            next_vec = obs_to_vector(next_obs, env_cfg, normalize_time)
            next_mask = np.asarray(next_obs["action_mask"], dtype=np.float32)
            # Emit completed n-step windows. On episode end, flush the whole queue
            # (truncated returns); otherwise emit the oldest once the window is full.
            if done:
                for tr in emit_nstep(nstep_queue, next_vec, next_mask, True, gamma, flush=True):
                    replay.add(*tr)
            elif len(nstep_queue) == n_step:
                for tr in emit_nstep(nstep_queue, next_vec, next_mask, False, gamma, flush=False):
                    replay.add(*tr)
            obs = next_obs
            ep_return += reward
            global_step += 1

            if (
                len(replay) >= int(train_cfg["learning_starts"])
                and global_step % int(train_cfg["train_every"]) == 0
            ):
                b_obs, b_actions, b_returns, b_next_obs, b_next_masks, b_dones, b_disc = replay.sample(int(train_cfg["batch_size"]))
                obs_t = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(b_actions, dtype=torch.int64, device=device).unsqueeze(1)
                returns_t = torch.as_tensor(b_returns, dtype=torch.float32, device=device)
                next_obs_t = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
                next_masks_t = torch.as_tensor(b_next_masks, dtype=torch.bool, device=device)
                dones_t = torch.as_tensor(b_dones, dtype=torch.float32, device=device)
                disc_t = torch.as_tensor(b_disc, dtype=torch.float32, device=device)

                q = q_net(obs_t).gather(1, actions_t).squeeze(1)
                with torch.no_grad():
                    if bool(train_cfg["double_dqn"]):
                        online_next = q_net(next_obs_t).masked_fill(~next_masks_t, -1e9)
                        next_actions = online_next.argmax(dim=1, keepdim=True)
                        next_q = target_net(next_obs_t).gather(1, next_actions).squeeze(1)
                    else:
                        next_q = target_net(next_obs_t).masked_fill(~next_masks_t, -1e9).max(dim=1).values
                    # n-step target: accumulated return + gamma**window * bootstrap.
                    target = returns_t + disc_t * (1.0 - dones_t) * next_q

                loss = loss_fn(q, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()
                last_loss = float(loss.item())
                train_updates += 1

            if global_step % int(train_cfg["target_update_every"]) == 0:
                target_net.load_state_dict(q_net.state_dict())

            if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_path(train_cfg["weight_path"], global_step),
                    env_cfg,
                    q_net,
                    train_cfg,
                )

            if eval_writer is not None and global_step % eval_every == 0:
                ev_cost, ev_return, ev_counts = greedy_eval(
                    q_net, env_cfg, device, normalize_time, eval_seeds
                )
                eval_writer.writerow({
                    "step": global_step,
                    "epsilon": epsilon_by_step(global_step, train_cfg),
                    "eval_cost_per_order": ev_cost,
                    "eval_episode_return": ev_return,
                    "eval_assign": ev_counts["assign"],
                    "eval_charge": ev_counts["charge"],
                    "eval_noop": ev_counts["noop"],
                })
                eval_file.flush()
                print(
                    f"[eval] step={global_step} cost={ev_cost:.2f} return={ev_return:.2f} "
                    f"assign={ev_counts['assign']} charge={ev_counts['charge']} noop={ev_counts['noop']}"
                )

            if done or global_step >= total_steps:
                break

        cost_per_order = episode_cost(env.stats)
        row = {
            "episode": episode,
            "step": global_step,
            "episode_return": ep_return,
            "cost_per_order": cost_per_order,
            "delivered": env.stats["delivered"],
            "dropped": env.stats["dropped"],
            "epsilon": epsilon_by_step(global_step, train_cfg),
            "loss": last_loss,
            "assign_actions": action_counts["assign"],
            "charge_actions": action_counts["charge"],
            "noop_actions": action_counts["noop"],
            "charge_open_steps": charge_open_steps,
            "train_updates": train_updates,
        }
        writer.writerow(row)
        log_file.flush()
        if episode % print_every_episodes == 0 or global_step >= total_steps:
            print(
                f"episode={episode} step={global_step} return={ep_return:.2f} "
                f"cost={cost_per_order:.2f} epsilon={row['epsilon']:.3f} "
                f"loss={last_loss} charge={action_counts['charge']}/{charge_open_steps} "
                f"updates={train_updates}"
            )
        episode += 1

    log_file.close()
    if eval_file is not None:
        eval_file.close()
    save_checkpoint(train_cfg["weight_path"], env_cfg, q_net, train_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dqn.yaml")
    args = parser.parse_args()
    train(args.config)
