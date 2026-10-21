"""Numpy inference of an exported policy (.npz from scripts/rl/export_policy.py).

The file holds obs_mean, obs_std, n_layers and W0, b0, ... W{n-1}, b{n-1}; hidden layers use ELU,
the output layer is linear (the Gaussian mean in policy units, clipped to [-1, 1] by the caller).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OBS_CLIP = 10.0


class NumpyPolicy:
    def __init__(self, path: str | Path):
        data = np.load(path)
        self.obs_mean = data["obs_mean"].astype(np.float64)
        self.obs_std = data["obs_std"].astype(np.float64)
        n = int(data["n_layers"])
        self.layers = [(data[f"W{i}"].astype(np.float64), data[f"b{i}"].astype(np.float64))
                       for i in range(n)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.clip((x - self.obs_mean) / self.obs_std, -OBS_CLIP, OBS_CLIP)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return x
