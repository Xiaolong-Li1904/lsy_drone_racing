"""Level 2: end-to-end RL policy (numpy inference only), see docs/rl_plan.md.

Needs env.control_mode = "attitude". The policy reads only relative geometry from the observation
(lsy_drone_racing/rl/features.py) and outputs [roll, pitch, yaw, collective thrust].
Weights: level2_rl_policy.npz next to this file, or the path in ADR_RL_POLICY.
"""

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
        return to_env(u, self._thrust_min, self._thrust_max, np)[0].astype(np.float32)

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        return False  # the base class returns True, which stops the flight after one step

    def episode_callback(self):
        self._prev_u = np.zeros((1, 4))
