"""DDPG — Deep Deterministic Policy Gradient (Lillicrap et al., 2016) for the
continuous DroneControl-v0 sub-env.

Off-policy actor-critic for continuous actions: a deterministic actor, a Q-critic,
target networks with soft (Polyak) updates, a replay buffer, and additive Gaussian
exploration noise. Graded against the GoStraight baseline (which stalls on no-fly
walls), so the learned controller should win on success rate and return.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

from drone_dispatch_env.env_control import DroneControlEnv

from .networks import DDPGActor, DDPGCritic
from .utils import CSVLogger, seed_everything


class ReplayBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, no, d
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch: int):
        idx = np.random.randint(0, len(self), size=batch)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next_obs[idx], self.done[idx])


def _clip_action(a: np.ndarray) -> np.ndarray:
    return np.array([np.clip(a[0], 0.0, 1.0), np.clip(a[1], -1.0, 1.0)], dtype=np.float32)


class OUNoise:
    """Ornstein-Uhlenbeck process (Lillicrap et al., 2016): temporally-correlated
    exploration so the drone commits to a heading long enough to make real progress
    toward the target, instead of spinning in place under i.i.d. noise."""

    def __init__(self, dim: int, sigma: float, theta: float = 0.15, seed: int = 0):
        self.mu = np.zeros(dim, dtype=np.float32)
        self.theta, self.sigma = theta, sigma
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        self.state = (self.state + self.theta * (self.mu - self.state)
                      + self.sigma * self.rng.standard_normal(self.state.shape).astype(np.float32))
        return self.state


@torch.no_grad()
def eval_ddpg(actor, cfg, seeds, device) -> dict:
    env = DroneControlEnv(cfg)
    rets, succ, steps = [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        done = False
        ret = 0.0
        n = 0
        last_term = False
        while not done:
            a = actor(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
            obs, r, term, trunc, _ = env.step(_clip_action(a))
            ret += r
            n += 1
            last_term = term
            done = term or trunc
        rets.append(ret)
        succ.append(1.0 if (last_term and obs[2] > 0.0) else 0.0)   # reached target w/ battery
        steps.append(n)
    return {"return": float(np.mean(rets)), "success_rate": float(np.mean(succ)),
            "mean_steps": float(np.mean(steps))}


def train(cfg, params: dict, seed: int, log_path: str, weight_path: str) -> dict:
    p = params
    device = torch.device("cpu")
    torch.set_num_threads(1)   # 1 core per process so seeds run in parallel
    seed_everything(seed)

    actor_lr = float(p.get("actor_lr", 1e-4))
    critic_lr = float(p.get("critic_lr", 1e-3))
    gamma = float(p.get("gamma", 0.99))
    tau = float(p.get("tau", 0.005))
    hidden = int(p.get("hidden", 256))
    buffer_size = int(p.get("buffer_size", 200_000))
    batch_size = int(p.get("batch_size", 256))
    warmup = int(p.get("warmup_steps", 10_000))
    noise_sigma = float(p.get("noise_sigma", 0.1))
    reward_scale = float(p.get("reward_scale", 0.1))
    total_steps = int(p.get("total_steps", 250_000))
    eval_every = int(p.get("eval_every_steps", 5_000))
    eval_seeds = list(p.get("eval_seeds", [200, 201, 202, 203, 204]))

    os.makedirs(os.path.dirname(weight_path) or ".", exist_ok=True)
    env = DroneControlEnv(cfg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = DDPGActor(obs_dim, hidden).to(device)
    critic = DDPGCritic(obs_dim, act_dim, hidden).to(device)
    actor_t = copy.deepcopy(actor)
    critic_t = copy.deepcopy(critic)
    a_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    buf = ReplayBuffer(buffer_size, obs_dim, act_dim)
    logger = CSVLogger(log_path)
    seed_rng = np.random.default_rng(seed)
    ou = OUNoise(act_dim, noise_sigma, seed=seed)

    def reset_new():
        return env.reset(seed=int(seed_rng.integers(10_000, 5_000_000)))[0]

    obs = reset_new()
    best_return = -float("inf")

    for step in range(1, total_steps + 1):
        if step < warmup:
            a = np.array([seed_rng.uniform(0, 1), seed_rng.uniform(-1, 1)], dtype=np.float32)
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
            a = _clip_action(a + ou.sample())

        nobs, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, r * reward_scale, nobs, 1.0 if term else 0.0)
        if term or trunc:
            obs = reset_new()
            ou.reset()
        else:
            obs = nobs

        if len(buf) >= batch_size and step >= warmup:
            o, ac, rw, no, dn = buf.sample(batch_size)
            o = torch.as_tensor(o, device=device)
            ac = torch.as_tensor(ac, device=device)
            rw = torch.as_tensor(rw, device=device)
            no = torch.as_tensor(no, device=device)
            dn = torch.as_tensor(dn, device=device)

            with torch.no_grad():
                target_q = rw + gamma * (1.0 - dn) * critic_t(no, actor_t(no))
            q = critic(o, ac)
            critic_loss = F.smooth_l1_loss(q, target_q)
            c_opt.zero_grad(); critic_loss.backward(); c_opt.step()

            actor_loss = -critic(o, actor(o)).mean()
            a_opt.zero_grad(); actor_loss.backward(); a_opt.step()

            with torch.no_grad():
                for pt, ps in zip(actor_t.parameters(), actor.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * ps.data)
                for pt, ps in zip(critic_t.parameters(), critic.parameters()):
                    pt.data.mul_(1 - tau).add_(tau * ps.data)

        if step % eval_every == 0 or step == total_steps:
            m = eval_ddpg(actor, cfg, eval_seeds, device)
            logger.log({"step": step, "return": m["return"],
                        "success_rate": m["success_rate"], "mean_steps": m["mean_steps"]})
            if m["return"] > best_return:
                best_return = m["return"]
                torch.save({"state_dict": actor.state_dict(), "hidden": hidden,
                            "obs_dim": obs_dim, "act_dim": act_dim,
                            "return": best_return}, weight_path)
            print(f"[ddpg seed={seed}] step {step}/{total_steps} "
                  f"return {m['return']:.2f} success {m['success_rate']:.2f} "
                  f"(best {best_return:.2f})", flush=True)

    logger.close()
    return {"best_return": best_return}
