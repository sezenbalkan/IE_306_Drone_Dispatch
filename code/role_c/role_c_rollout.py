from __future__ import annotations

import numpy as np

from drone_dispatch_env import Config
from drone_dispatch_env.config import CHARGER
from drone_dispatch_env.world import Router


class RoutedDistance:
    """
    No-fly-aware routed distance cache.
    This uses only obs['grid'], so it respects the frozen policy interface.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._grid_key = None
        self._router = None
        self._fields = {}

    def _ensure_grid(self, grid):
        grid = np.asarray(grid)
        key = grid.tobytes()
        if key != self._grid_key:
            self._grid_key = key
            self._router = Router(grid, self.cfg.neighborhood)
            self._fields = {}

    def dist(self, grid, source, target) -> float:
        self._ensure_grid(grid)
        source = (int(source[0]), int(source[1]))
        target = (int(target[0]), int(target[1]))

        field = self._fields.get(source)
        if field is None:
            field = self._router.dist_field(source)
            self._fields[source] = field

        d = float(field[target[0], target[1]])
        if np.isfinite(d):
            return d

        # safety fallback
        return abs(source[0] - target[0]) + abs(source[1] - target[1])


class RoleCRolloutPlanner:
    """
    Role C: rollout-style planning policy.

    depth=0:
        Greedy-like: mainly pickup distance.
    depth=1:
        Adds delivery distance, deadline risk, and battery feasibility.
    depth=2:
        Adds post-delivery charging readiness as a shallow rollout proxy.

    This is not changing the simulator. It only implements act(obs) -> action.
    """

    def __init__(
        self,
        cfg: Config,
        depth: int = 2,
        reserve_soc: float = 0.08,
        emergency_soc: float = 0.22,
    ):
        self.cfg = cfg
        self.depth = int(depth)
        self.reserve_soc = float(reserve_soc)
        self.emergency_soc = float(emergency_soc)
        self.routed = RoutedDistance(cfg)

    def act(self, obs):
        c = self.cfg
        mask = np.asarray(obs["action_mask"])
        valid = np.flatnonzero(mask)

        if len(valid) == 0:
            return c.noop_index

        drones = obs["drones"]
        orders = obs["orders"]
        grid = obs["grid"]

        # 1) Emergency charging: do not risk losing very low-battery idle drones.
        for d in range(c.n_drones):
            charge_action = c.charge_index(d)
            if mask[charge_action] and drones[d, 2] < self.emergency_soc:
                return int(charge_action)

        best_action = c.noop_index
        best_score = float("inf")

        for action in valid:
            action = int(action)

            if action == c.noop_index:
                score = 1e6

            elif action >= c.n_drones * c.k_max:
                # charge action
                d = action - c.n_drones * c.k_max
                soc = float(drones[d, 2])
                drone_pos = (drones[d, 0], drones[d, 1])
                charger_dist = self._nearest_charger_distance(grid, drone_pos)

                # Low battery charging should be attractive.
                if soc < c.charge_threshold:
                    score = -25.0 + 50.0 * soc + 0.5 * charger_dist
                else:
                    # Do not overcharge healthy drones unless no safe assignment exists.
                    score = 80.0 + 80.0 * soc + 0.5 * charger_dist

            else:
                # assignment action
                d = action // c.k_max
                slot = action % c.k_max

                soc = float(drones[d, 2])
                drone_pos = (drones[d, 0], drones[d, 1])
                origin = (orders[slot, 0], orders[slot, 1])
                dest = (orders[slot, 2], orders[slot, 3])
                age = float(orders[slot, 4])

                pickup_dist = self.routed.dist(grid, drone_pos, origin)

                if self.depth == 0:
                    # Greedy-like ablation: only nearest pickup with basic battery guard.
                    if soc < c.charge_threshold:
                        score = 1e5 + pickup_dist
                    else:
                        score = pickup_dist

                else:
                    delivery_dist = self.routed.dist(grid, origin, dest)
                    eta = pickup_dist + delivery_dist

                    remaining = max(0.0, c.sla_steps - age)
                    lateness_risk = max(0.0, eta - remaining)

                    energy_need = c.e_move * eta
                    battery_shortfall = max(0.0, energy_need + self.reserve_soc - soc)

                    score = (
                        1.00 * pickup_dist
                        + 0.75 * delivery_dist
                        + 8.00 * lateness_risk
                        + 150.0 * battery_shortfall
                        - 0.15 * age
                    )

                    if self.depth >= 2:
                        # Shallow rollout proxy:
                        # after delivery, prefer actions that leave the drone closer to charging.
                        post_charge_dist = self._nearest_charger_distance(grid, dest)
                        score += 0.25 * post_charge_dist

                    # If low battery and this job is not safely feasible, penalize strongly.
                    if soc < c.charge_threshold and battery_shortfall > 0:
                        score += 100.0

            # deterministic tie-breaker
            score += 1e-6 * action

            if score < best_score:
                best_score = score
                best_action = action

        return int(best_action)

    def _nearest_charger_distance(self, grid, pos) -> float:
        chargers = np.argwhere(np.asarray(grid) == CHARGER)
        if len(chargers) == 0:
            return 0.0

        best = float("inf")
        for ch in chargers:
            d = self.routed.dist(grid, pos, (ch[0], ch[1]))
            if d < best:
                best = d

        return best if np.isfinite(best) else 0.0

    # Optional visualizer hooks. We skip them for this planner.
    def action_values(self, obs):
        return None

    def action_probs(self, obs):
        return None

    def state_values(self, obs):
        return None
