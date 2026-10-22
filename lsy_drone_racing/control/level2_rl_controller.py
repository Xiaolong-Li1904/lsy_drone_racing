from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from lsy_drone_racing.control import Controller
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.rl.actions import to_env
from lsy_drone_racing.rl.features import features
from lsy_drone_racing.rl.policy_np import NumpyPolicy

DEFAULT_POLICY = Path(__file__).with_name("level2_rl_policy.npz")


class Level2RLController(Controller):
    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        self._policy = NumpyPolicy(os.environ.get("ADR_RL_POLICY", DEFAULT_POLICY))
        space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
        self._thrust_min, self._thrust_max = float(space.low[3]), float(space.high[3])
        self._prev_u = np.zeros((1, 4))

    def compute_control(self, obs, info=None):
        batch = {k: np.asarray(v)[None] for k, v in obs.items()}
        u = np.clip(self._policy(features(batch, self._prev_u, np)), -1.0, 1.0)
        self._prev_u = u
        action = to_env(u, self._thrust_min, self._thrust_max, np, self._policy.tilt_max)
        return action[0].astype(np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return False

    def episode_callback(self):
        self._prev_u = np.zeros((1, 4))
