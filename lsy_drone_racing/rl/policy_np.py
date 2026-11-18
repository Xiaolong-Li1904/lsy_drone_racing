from __future__ import annotations

from pathlib import Path

import numpy as np

from lsy_drone_racing.rl.actions import TILT_MAX_RAD
from lsy_drone_racing.rl.features import N_GATES, N_PILLARS, input_dim

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
        self.thrust_frac = float(data["thrust_frac"]) if "thrust_frac" in data.files else 1.0
        self.n_gates = int(data["n_gates"]) if "n_gates" in data.files else N_GATES
        self.n_pillars = int(data["n_pillars"]) if "n_pillars" in data.files else N_PILLARS
        self.n_hist = int(data["n_hist"]) if "n_hist" in data.files else 0
        self.n_vhist = int(data["n_vhist"]) if "n_vhist" in data.files else 0
        self.n_param = int(data["n_param"]) if "n_param" in data.files else 0
        assert self.obs_mean.shape[-1] == input_dim(self.n_gates, self.n_pillars, self.n_hist,
                                                    self.n_vhist, self.n_param), (
            path, self.n_gates, self.n_pillars, self.n_hist, self.n_vhist, self.n_param)

        # Optional residual safety layer: out = base + res_scale * tanh(residual MLP).
        self.res_layers = []
        if "res_n_layers" in data.files:
            self.res_layers = [(data[f"R{i}"].astype(np.float64), data[f"rb{i}"].astype(np.float64))
                               for i in range(int(data["res_n_layers"]))]
            self.res_scale = float(data["res_scale"])

    @staticmethod
    def _mlp(x: np.ndarray, layers) -> np.ndarray:
        for i, (w, b) in enumerate(layers):
            x = x @ w + b
            if i < len(layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.clip((x - self.obs_mean) / self.obs_std, -OBS_CLIP, OBS_CLIP)
        out = self._mlp(x, self.layers)
        if self.res_layers:
            out = out + self.res_scale * np.tanh(self._mlp(x, self.res_layers))
        return out
