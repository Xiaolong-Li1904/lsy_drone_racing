from __future__ import annotations

N_PILLARS = 2  # default number of nearest pillars in the observation
N_GATES = 2  # default number of upcoming gates in the observation
GATE_START = 15  # first gate feature
PARAM_NAMES = ("delay_steps", "thrust_scale")


def gate_block_dim(n_gates: int) -> int:
    """Number of gate features: 10 for the next gate, 11 for each further one."""
    return 10 + 11 * (n_gates - 1)


def obs_dim(n_gates: int = N_GATES, n_pillars: int = N_PILLARS) -> int:
    return GATE_START + gate_block_dim(n_gates) + 3 * n_pillars + 4


def input_dim(n_gates: int = N_GATES, n_pillars: int = N_PILLARS, n_hist: int = 0,
              n_vhist: int = 0, n_param: int = 0) -> int:
    return obs_dim(n_gates, n_pillars) + 4 * n_hist + 3 * n_vhist + n_param


OBS_DIM = obs_dim(N_GATES)


def quat_to_mat(q, xp):
    """Rotation matrices of xyzw quaternions, shape (..., 3, 3)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack([
        xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def _take(a, idx, xp):
    """a[b, idx[b]] for a of shape (B, K, ...) and idx of shape (B,)."""
    idx = idx.reshape(idx.shape + (1,) * (a.ndim - 1))
    return xp.take_along_axis(a, idx, axis=1)[:, 0]


def features(obs: dict, prev_action, xp, n_gates: int = N_GATES, n_pillars: int = N_PILLARS):
    """Policy input of shape (B, obs_dim(n_gates, n_pillars)) from batched env observations."""
    pos = obs["pos"]
    rot = quat_to_mat(obs["quat"], xp)
    yaw = xp.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    c, s = xp.cos(yaw), xp.sin(yaw)

    def heading(v):
        cc = c.reshape(c.shape + (1,) * (v.ndim - 2))
        ss = s.reshape(s.shape + (1,) * (v.ndim - 2))
        return xp.stack([cc * v[..., 0] + ss * v[..., 1], -ss * v[..., 0] + cc * v[..., 1],
                         v[..., 2]], axis=-1)

    n_passes = obs["gate_sequence"].shape[-1]
    passed = obs["n_gates_passed"].astype(xp.int32)
    sequence = obs["gate_sequence"].astype(xp.int32)
    direction = obs["gate_sequence_direction"]

    def gate(k):
        slot = xp.minimum(passed + k, n_passes - 1)
        gate_id = _take(sequence, slot, xp)
        sign = _take(direction, slot, xp).astype(pos.dtype)[:, None]
        rot_g = quat_to_mat(_take(obs["gates_quat"], gate_id, xp), xp)
        sensed = _take(obs["gates_visited"], gate_id, xp).astype(pos.dtype)[:, None]
        rel = _take(obs["gates_pos"], gate_id, xp) - pos
        return xp.concatenate([heading(rel), heading(rot_g[..., 0] * sign),
                               heading(rot_g[..., 1] * sign), sensed], axis=-1)

    gates = [gate(0)]
    for k in range(1, n_gates):
        has = (passed + k < n_passes).astype(pos.dtype)[:, None]
        gates += [gate(k) * has, has]

    rel_p = obs["obstacles_pos"][..., :2] - pos[:, None, :2]
    order = xp.argsort(xp.sum(rel_p * rel_p, axis=-1), axis=1)[:, :n_pillars]
    near = xp.take_along_axis(rel_p, order[..., None], axis=1)
    near = heading(xp.concatenate([near, xp.zeros_like(near[..., :1])], axis=-1))[..., :2]
    sensed_p = xp.take_along_axis(obs["obstacles_visited"], order, axis=1).astype(pos.dtype)
    pillars = xp.concatenate([near, sensed_p[..., None]], axis=-1).reshape(pos.shape[0], -1)

    return xp.concatenate([
        heading(obs["vel"]), obs["ang_vel"], heading(rot[..., 0]), heading(rot[..., 2]),
        xp.stack([s, c], axis=-1), pos[:, 2:3],
        *gates,
        pillars, prev_action,
    ], axis=-1).astype(xp.float32)


def policy_input(obs: dict, prev_actions, xp, n_gates: int = N_GATES, n_pillars: int = N_PILLARS,
                 vel_hist=None, params=None):
    """features() followed by the older actions; prev_actions has shape (B, 1 + n_hist, 4),
    newest (u_{t-1}) first. Optional vel_hist (B, n_vhist, 3), newest first, and params
    (B, n_param) are appended in that order."""
    x = features(obs, prev_actions[:, 0], xp, n_gates, n_pillars)
    b = prev_actions.shape[0]
    parts = [x, prev_actions[:, 1:].reshape(b, -1).astype(xp.float32)]
    if vel_hist is not None:
        parts.append(vel_hist.reshape(b, -1).astype(xp.float32))
    if params is not None:
        parts.append(params.astype(xp.float32))
    return xp.concatenate(parts, axis=-1)
