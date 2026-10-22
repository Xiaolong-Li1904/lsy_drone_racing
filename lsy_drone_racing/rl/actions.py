from __future__ import annotations

TILT_MAX_RAD = 1.0
YAW_MAX_RAD = 0.5


def to_env(u, thrust_min, thrust_max, xp, tilt_max=TILT_MAX_RAD):
    u = xp.clip(u, -1.0, 1.0)
    thrust = thrust_min + 0.5 * (u[..., 3] + 1.0) * (thrust_max - thrust_min)
    return xp.stack([u[..., 0] * tilt_max, u[..., 1] * tilt_max, u[..., 2] * YAW_MAX_RAD,
                     thrust], axis=-1)


def from_env(a, thrust_min, thrust_max, xp, tilt_max=TILT_MAX_RAD):
    u = xp.stack([a[..., 0] / tilt_max, a[..., 1] / tilt_max, a[..., 2] / YAW_MAX_RAD,
                  2.0 * (a[..., 3] - thrust_min) / (thrust_max - thrust_min) - 1.0], axis=-1)
    return xp.clip(u, -1.0, 1.0)
