"""Policy action u in [-1, 1]^4 <-> env attitude action [roll, pitch, yaw, collective thrust].

Roll/pitch/yaw bounds are those of level2_mpc_controller.py (TILT_MAX_RAD, YAW_MAX_RAD), so the
MPC demonstrations map into [-1, 1]. Yaw is absolute (the env clips it to +-pi/2).
"""

from __future__ import annotations

TILT_MAX_RAD = 1.0
YAW_MAX_RAD = 0.5


def to_env(u, thrust_min, thrust_max, xp):
    """Env action from policy units; thrust_min/max are the env's total-thrust bounds."""
    u = xp.clip(u, -1.0, 1.0)
    thrust = thrust_min + 0.5 * (u[..., 3] + 1.0) * (thrust_max - thrust_min)
    return xp.stack([u[..., 0] * TILT_MAX_RAD, u[..., 1] * TILT_MAX_RAD, u[..., 2] * YAW_MAX_RAD,
                     thrust], axis=-1)


def from_env(a, thrust_min, thrust_max, xp):
    """Policy units of an env action (inverse of to_env inside the bounds)."""
    u = xp.stack([a[..., 0] / TILT_MAX_RAD, a[..., 1] / TILT_MAX_RAD, a[..., 2] / YAW_MAX_RAD,
                  2.0 * (a[..., 3] - thrust_min) / (thrust_max - thrust_min) - 1.0], axis=-1)
    return xp.clip(u, -1.0, 1.0)
