"""A2C — synchronous advantage actor-critic (Mnih et al., 2016, the synchronous
variant of A3C).

Same factored policy/value net as REINFORCE, but instead of waiting for whole
episodes it collects fixed n-step rollouts and bootstraps the advantage with the
critic's value of the state after the rollout (GAE-lambda). Lower variance and
more sample-efficient than Monte-Carlo REINFORCE. One env, n-step rollouts;
episodes auto-reset mid-rollout with fresh training seeds.
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from drone_dispatch_env.env_dispatch import DroneDispatchEnv

from .features import RoutedCache, extract_features
from .networks import FactoredActorCritic
from .rollout import compute_gae, normalize
from .utils import (CSVLogger, batch_features, evaluate_dispatch_agent,
                    seed_everything, single_batch)
from .adapters import PolicyGradientAgent


def train(cfg, params: dict, seed: int, log_path: str, weight_path: str) -> dict:
    p = params
    device = torch.device("cpu")
    torch.set_num_threads(1)   # 1 core per process so seeds run in parallel
    seed_everything(seed)

    lr = float(p.get("lr", 7e-4))
    gamma = float(p.get("gamma", 0.99))
    lam = float(p.get("gae_lambda", 0.95))
    ent_coef = float(p.get("ent_coef", 0.01))
    vf_coef = float(p.get("vf_coef", 0.5))
    hidden = int(p.get("hidden", 128))
    max_grad_norm = float(p.get("max_grad_norm", 0.5))
    norm_adv = bool(p.get("norm_adv", True))
    reward_scale = float(p.get("reward_scale", 0.1))
    n_steps = int(p.get("n_steps", 32))
    total_steps = int(p.get("total_steps", 500_000))
    eval_every = int(p.get("eval_every_updates", 400))
    val_seeds = list(p.get("val_seeds", [200, 201, 202, 203, 204]))

    os.makedirs(os.path.dirname(weight_path) or ".", exist_ok=True)
    env = DroneDispatchEnv(cfg)
    net = FactoredActorCritic(hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    cache = RoutedCache(cfg.neighborhood)
    seed_rng = np.random.default_rng(seed)
    logger = CSVLogger(log_path)

    n_updates = max(1, total_steps // n_steps)
    ep_return_hist = deque(maxlen=50)
    best_cost = float("inf")

    def reset_new():
        s = int(seed_rng.integers(10_000, 5_000_000))
        obs, _ = env.reset(seed=s)
        return obs

    obs = reset_new()
    ep_ret = 0.0
    global_steps = 0
    last_loss = (0.0, 0.0, 0.0)

    for update in range(1, n_updates + 1):
        feats, actions, rewards, dones = [], [], [], []
        for _ in range(n_steps):
            feat = extract_features(obs, cfg, cache)
            b = single_batch(feat, device)
            with torch.no_grad():
                logits, _ = net(b)
            a = int(torch.distributions.Categorical(logits=logits[0]).sample().item())
            nobs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            feats.append(feat)
            actions.append(a)
            rewards.append(r * reward_scale)
            dones.append(1.0 if done else 0.0)
            ep_ret += r
            global_steps += 1
            if done:
                ep_return_hist.append(ep_ret)
                ep_ret = 0.0
                obs = reset_new()
            else:
                obs = nobs

        # bootstrap value of the state following the rollout
        with torch.no_grad():
            bf = single_batch(extract_features(obs, cfg, cache), device)
            _, last_v = net(bf)
        last_value = float(last_v.item())

        b = batch_features(feats, device)
        acts = torch.as_tensor(actions, dtype=torch.long, device=device)
        logits, values = net(b)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(acts)
        entropy = dist.entropy().mean()

        values_np = values.detach().cpu().numpy()
        adv, returns = compute_gae(rewards, values_np, dones, gamma, lam, last_value)
        if norm_adv:
            adv = normalize(adv)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        policy_loss = -(logp * adv_t).mean()
        value_loss = F.smooth_l1_loss(values, ret_t)
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        opt.step()
        last_loss = (float(policy_loss.item()), float(value_loss.item()), float(entropy.item()))

        if update % eval_every == 0 or update == n_updates:
            agent = PolicyGradientAgent(net, cfg, device, deterministic=True)
            m = evaluate_dispatch_agent(agent, cfg, val_seeds)
            net.train()
            row = {
                "update": update, "step": global_steps,
                "train_return": float(np.mean(ep_return_hist)) if ep_return_hist else 0.0,
                "cost_per_order": m["cost_per_order"],
                "success_rate": m["success_rate"],
                "ontime_rate": m["ontime_rate"],
                "depletion_events": m["depletion_events"],
                "n_delivered": m["n_delivered"],
                "eval_return": m["episode_return"],
                "policy_loss": last_loss[0],
                "value_loss": last_loss[1],
                "entropy": last_loss[2],
            }
            logger.log(row)
            if m["cost_per_order"] < best_cost:
                best_cost = m["cost_per_order"]
                torch.save({"state_dict": net.state_dict(), "hidden": hidden,
                            "cost_per_order": best_cost}, weight_path)
            print(f"[a2c seed={seed}] upd {update}/{n_updates} step {global_steps} "
                  f"cost/order {m['cost_per_order']:.3f} (best {best_cost:.3f}) "
                  f"deliv {m['n_delivered']:.0f}", flush=True)

    logger.close()
    return {"best_cost": best_cost}
