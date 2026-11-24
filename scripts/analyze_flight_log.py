"""Summarise flight logs written by level2_rl_kappa_controller.py and estimate the control delay.

Run as:

    $ python scripts/analyze_flight_log.py                       # all files in flight_logs/
    $ python scripts/analyze_flight_log.py flight_logs/xyz.npz   # selected files / folders

Delay estimate: the measured roll/pitch (from the estimator quaternion) is fitted to a first-order
response to the commanded roll/pitch shifted by d control steps,
    att[t+1] - att[t] = a * (cmd[t - d] - att[t]) + b,
for d = 0 .. 4 steps in 0.1-step increments (fractional d = linear interpolation). The d with the
smallest residual is the raw lag; it includes a sim/attitude-loop baseline (SIM_BASELINE, measured on sim
logs of l2_robust kappa 0 on level2.toml, 6 seeds each: sim delay 0 -> 1.33, 1 -> 2.35, 2 -> 3.23 steps).
equiv_delay_steps = raw lag - SIM_BASELINE is the delay in the unit of train_ppo --delay-max
(1 step = 1/freq s = 20 ms). It is an estimate (+-0.3 steps): the real attitude loop is not the sim one.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
from scipy.spatial.transform import Rotation

SIM_BASELINE = 1.33  # raw lag of a sim flight without added delay (see docstring)
DELAYS = np.round(np.arange(0.0, 4.0001, 0.1), 2)


def _shift(cmd: np.ndarray, d: float) -> np.ndarray:
    """cmd[t - d] for fractional d (linear interpolation), first value held before the start."""
    n = len(cmd)
    t = np.arange(n) - d
    return np.interp(t, np.arange(n), cmd, left=cmd[0])


def fit_delay(cmd: np.ndarray, att: np.ndarray, skip: int = 10) -> tuple[float, np.ndarray]:
    """Best fractional delay (steps) and the residual curve over DELAYS."""
    res = []
    for d in DELAYS:
        c = _shift(cmd, d)[skip:-1]
        a0 = att[skip:-1]
        y = att[skip + 1 :] - a0
        X = np.stack([c - a0, np.ones_like(a0)], axis=1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res.append(np.mean((y - X @ coef) ** 2))
    res = np.asarray(res)
    return float(DELAYS[np.argmin(res)]), res


def analyse(path: Path) -> dict:
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    t = d["t"] - d["t"][0]
    dt = np.diff(d["t"]) * 1e3
    rpy = Rotation.from_quat(d["obs_quat"]).as_euler("xyz")
    cmd = d["action"]
    d_roll, _ = fit_delay(cmd[:, 0], rpy[:, 0])
    d_pitch, _ = fit_delay(cmd[:, 1], rpy[:, 1])
    n_gates = int(d["obs_n_gates_passed"][-1])
    n_total = int(d["obs_gate_sequence"].shape[-1])
    return {
        "file": path.name,
        "policy": Path(meta["policy"].replace("\\", "/")).stem,
        "kappa": meta["kappa"],
        "steps": len(t),
        "duration_s": round(float(t[-1]), 2),
        "gates": f"{n_gates}/{n_total}" if n_gates >= 0 else f"crash after {d['obs_n_gates_passed'].max()}",
        "loop_dt_ms_mean": round(float(dt.mean()), 2),
        "loop_dt_ms_max": round(float(dt.max()), 1),
        "loop_steps_over_25ms": int((dt > 25).sum()),
        "delay_roll_steps": d_roll,
        "delay_pitch_steps": d_pitch,
        "raw_lag_steps": round((d_roll + d_pitch) / 2, 2),
        "equiv_delay_steps": round((d_roll + d_pitch) / 2 - SIM_BASELINE, 2),
        "equiv_delay_ms": round(((d_roll + d_pitch) / 2 - SIM_BASELINE) * 1e3 / meta["freq"], 0),
        "max_tilt_cmd_rad": round(float(np.abs(cmd[:, :2]).max()), 2),
        "max_tilt_meas_rad": round(float(np.abs(rpy[:, :2]).max()), 2),
        "max_speed_mps": round(float(np.linalg.norm(d["obs_vel"], axis=1).max()), 2),
        "min_height_m": round(float(d["obs_pos"][:, 2].min()), 2),
    }


def main(*paths: str):
    """Analyse flight log files or folders (default: flight_logs/)."""
    root = Path(__file__).parents[1]
    paths = paths or (str(root / "flight_logs"),)
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    if not files:
        print("no logs found")
        return
    rows = [analyse(f) for f in files]
    for r in rows:
        print(json.dumps(r))
    delays = [r["equiv_delay_steps"] for r in rows]
    print(f"\n{len(rows)} logs, equivalent delay (sim steps) median {np.median(delays):.2f}, "
          f"min {min(delays):.2f}, max {max(delays):.2f}")


if __name__ == "__main__":
    fire.Fire(main)
