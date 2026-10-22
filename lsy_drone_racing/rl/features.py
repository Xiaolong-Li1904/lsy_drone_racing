from __future__ import annotations

N_PILLARS = 2
OBS_DIM = 46


def quat_to_mat(q, xp):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack([
        xp.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        xp.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        xp.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def _take(a, idx, xp):
    idx = idx.reshape(idx.shape + (1,) * (a.ndim - 1))
    return xp.take_along_axis(a, idx, axis=1)[:, 0]


def features(obs: dict, prev_action, xp):
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

    has_second = (passed + 1 < n_passes).astype(pos.dtype)[:, None]

    rel_p = obs["obstacles_pos"][..., :2] - pos[:, None, :2]
    order = xp.argsort(xp.sum(rel_p * rel_p, axis=-1), axis=1)[:, :N_PILLARS]
    near = xp.take_along_axis(rel_p, order[..., None], axis=1)
    near = heading(xp.concatenate([near, xp.zeros_like(near[..., :1])], axis=-1))[..., :2]
    sensed_p = xp.take_along_axis(obs["obstacles_visited"], order, axis=1).astype(pos.dtype)
    pillars = xp.concatenate([near, sensed_p[..., None]], axis=-1).reshape(pos.shape[0], -1)

    return xp.concatenate([
        heading(obs["vel"]), obs["ang_vel"], heading(rot[..., 0]), heading(rot[..., 2]),
        xp.stack([s, c], axis=-1), pos[:, 2:3],
        gate(0), gate(1) * has_second, has_second,
        pillars, prev_action,
    ], axis=-1).astype(xp.float32)
