from __future__ import annotations

from pathlib import Path

import numpy as np

from lsy_drone_racing.rl.actions import TILT_MAX_RAD

OBS_CLIP = 10.0


class NumpyPolicy:
    def __init__(self, path: str | Path):
        data = np.load(path)
        self.obs_mean = data["obs_mean"].astype(np.float64)
        self.obs_std = data["obs_std"].astype(np.float64)
        n = int(data["n_layers"])
        self.layers = [(data[f"W{i}"].astype(np.float64), data[f"b{i}"].astype(np.float64))
                       for i in range(n)]
        self.tilt_max = float(data["tilt_max"]) if "tilt_max" in data.files else TILT_MAX_RAD

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.clip((x - self.obs_mean) / self.obs_std, -OBS_CLIP, OBS_CLIP)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return x
