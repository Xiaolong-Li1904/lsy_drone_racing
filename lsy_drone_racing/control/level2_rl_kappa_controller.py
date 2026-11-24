"""Real-flight RL controller for feature-version-3 / kappa policies (numpy inference only, 2026-11-23).

Needs env.control_mode = "attitude". Policy npz from ADR_RL_POLICY (exported by train_ppo.py). Supports the
velocity history (n_vhist) of fv-3 policies and kappa policies (train_ppo --kappa):
- ADR_KAPPA (float, default 0.0): aggressiveness, 0 = safest .. 1 = fastest.
- ADR_KAPPA_FILE (optional path): a text file holding one float; it is re-read at every episode start
  (episode_callback / reset), so the aggressiveness can be changed between laps without restarting.
For kappa policies the roll/pitch command limit widens with kappa exactly as in training (kappa_k0,
kappa_tilt_hi stored in the npz). No artificial delay / thrust change (unlike level3_rl_stress_controller).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lsy_drone_racing.control.level2_rl_controller import Level2RLController
from lsy_drone_racing.rl.actions import to_env
from lsy_drone_racing.rl.features import policy_input


class Level2RLKappaController(Level2RLController):
    # Used when ADR_RL_POLICY is not set: the generalist policy (no kappa, tilt 1.0) as a safe fallback.
    default_policy = Path(__file__).parents[2] / "artifacts/race_init/general_tilt1.0.npz"
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self._vel_hist = np.zeros((1, max(self._policy.n_vhist, 1), 3))  # newest first
        self._kappa = self._read_kappa()

    def _read_kappa(self) -> float:
        k = float(os.environ.get("ADR_KAPPA", "0"))
        path = os.environ.get("ADR_KAPPA_FILE")
        if path and os.path.exists(path):
            try:
                k = float(open(path).read().strip())
            except ValueError:
                pass
        return float(np.clip(k, 0.0, 1.0))

    def compute_control(self, obs, info=None):
        batch = {k: np.asarray(v)[None] for k, v in obs.items()}
        if self._policy.kappa:
            batch["kappa"] = np.array([self._kappa], np.float32)
        x = policy_input(batch, self._prev_u, np, self._policy.n_gates, self._policy.n_pillars,
                         self._vel_hist if self._policy.n_vhist else None,
                         fv=self._policy.feature_version)
        raw = self._policy(x)
        u = np.clip(raw, -1.0, 1.0)
        tilt = self._policy.tilt_max
        wide = self._policy.kappa and self._policy.kappa_tilt_hi > tilt
        if wide:
            k0 = self._policy.kappa_k0
            lim = 1.0 + (self._policy.kappa_tilt_hi / tilt - 1.0) * np.clip(
                (self._kappa - k0) / max(1e-6, 1.0 - k0), 0.0, 1.0)
            u[:, :2] = np.clip(raw[:, :2], -lim, lim)
        self._prev_u = np.concatenate([u[:, None], self._prev_u[:, :-1]], axis=1)
        self._vel_hist = np.concatenate([x[:, None, 0:3], self._vel_hist[:, :-1]], axis=1)
        action = to_env(u, self._thrust_min, self._thrust_max, np, tilt)
        if wide:
            action[:, :2] = u[:, :2] * tilt
        return action[0].astype(np.float32)

    def episode_callback(self):
        super().episode_callback()
        self._vel_hist = np.zeros_like(self._vel_hist)
        self._kappa = self._read_kappa()
