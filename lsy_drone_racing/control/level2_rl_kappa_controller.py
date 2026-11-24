"""Real-flight RL controller for feature-version-3 / kappa policies (numpy inference only, 2026-11-23).

Needs env.control_mode = "attitude". Policy npz from ADR_RL_POLICY (exported by train_ppo.py). Supports the
velocity history (n_vhist) of fv-3 policies and kappa policies (train_ppo --kappa):
- ADR_KAPPA (float, default 0.0): aggressiveness, 0 = safest .. 1 = fastest.
- ADR_KAPPA_FILE (optional path): a text file holding one float; it is re-read at every episode start
  (episode_callback / reset), so the aggressiveness can be changed between laps without restarting.
For kappa policies the roll/pitch command limit widens with kappa exactly as in training (kappa_k0,
kappa_tilt_hi stored in the npz). No artificial delay / thrust change (unlike level3_rl_stress_controller).

Flight log: every control step (time, full observation, policy input, raw/clipped output, sent action) is
recorded in memory and saved to flight_logs/<date>_<time>_<policy>_k<kappa>.npz when the flight ends,
also on Ctrl+C. ADR_LOG_DIR=<dir> changes the folder, ADR_LOG_DIR= (empty) disables logging.
"""

from __future__ import annotations

import atexit
import json
import os
import time
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
        self._log_init(config)

    # ---- flight log -------------------------------------------------------------------------------
    def _log_init(self, config):
        self._log = None
        log_dir = os.environ.get("ADR_LOG_DIR", str(Path(__file__).parents[2] / "flight_logs"))
        if not log_dir:
            return
        policy = Path(os.environ.get("ADR_RL_POLICY", self.default_policy))
        now = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{int(now * 1000) % 1000:03d}"
        self._log_path = Path(log_dir) / f"{stamp}_{policy.stem}_k{self._kappa:.2f}.npz"
        self._log_meta = {
            "policy": str(policy), "kappa": self._kappa, "tilt_max": float(self._policy.tilt_max),
            "kappa_tilt_hi": float(getattr(self._policy, "kappa_tilt_hi", 0.0)),
            "thrust_min": self._thrust_min, "thrust_max": self._thrust_max,
            "freq": int(config.env.freq), "control_mode": str(config.env.control_mode),
            "drones": str(config.deploy.drones) if "deploy" in config else "",
            "start_unix_time": now,
        }
        self._log = {"t": [], "wall": [], "obs": [], "x": [], "raw": [], "u": [], "action": []}
        atexit.register(self._log_save)

    def _log_save(self):
        log = self._log
        if not log or not log["t"]:
            return
        out = {k: np.asarray(log[k]) for k in ("t", "wall", "x", "raw", "u", "action")}
        for k in log["obs"][0]:
            out["obs_" + k] = np.stack([o[k] for o in log["obs"]])
        out["meta"] = np.array(json.dumps(self._log_meta))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._log_path, **out)

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
        t_in = time.perf_counter()
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
        action = action[0].astype(np.float32)
        if self._log is not None:
            log = self._log
            log["t"].append(t_in)
            log["wall"].append(time.time())
            log["obs"].append({k: np.array(v) for k, v in obs.items()})
            log["x"].append(x[0].astype(np.float32))
            log["raw"].append(raw[0].astype(np.float32))
            log["u"].append(u[0].astype(np.float32))
            log["action"].append(action.copy())
        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        if terminated or truncated:
            self._log_save()
        return super().step_callback(action, obs, reward, terminated, truncated, info)

    def episode_callback(self):
        self._log_save()
        super().episode_callback()
        self._vel_hist = np.zeros_like(self._vel_hist)
        self._kappa = self._read_kappa()
