"""Policy observation: relative geometry only, shared by training (jax.numpy) and the controller (numpy).

All vectors are expressed in the heading frame (world frame rotated by the drone's yaw), so no
world position or gate number reaches the policy. Inputs are batched env observations
(leading axis B); the controller adds a batch axis of 1.

Layout for n_gates upcoming gates and n_pillars pillars (OBS_DIM = obs_dim(2, 2) = 46,
obs_dim(4, 2) = 68, obs_dim(2, 4) = 52):
  0:3   velocity                         3:6   angular velocity (as the env reports it)
  6:9   body x axis                      9:12  body z axis (thrust direction)
  12:14 sin/cos yaw (yaw is commanded absolutely, see actions.py)
  14    height above ground
  15:25 next gate: rel pos 3, normal 3 (flipped for reverse passes), lateral axis 3, sensed 1
  then for each further gate k = 1 .. n_gates - 1: the same 10 values (zero past the last
        gate) and 1 flag "gate k exists"; for n_gates = 2 this is 25:35 and 35
  then  n_pillars nearest pillars (horizontal): rel xy 2, sensed 1 each (3 * n_pillars)
  last 4 previous action in policy units [-1, 1]
The numbers of gates and pillars are stored with the policy (n_gates, n_pillars, default 2 each).
policy_input appends n_hist older actions (u_{t-2}, u_{t-3}, ...; default 0) after these features,
so a policy trained with action delay can tell which commands are still in flight. Then optionally
n_vhist older velocities (features 0:3 of the last n_vhist steps, newest first; zeros before the
episode start) and n_param plant parameters (PARAM_NAMES: action delay in control steps, thrust
scale of the motors), so a policy can adapt to latency and thrust either by inferring them from
the history or by being told the measured values.
"""

from __future__ import annotations

N_PILLARS = 2  # default number of nearest pillars in the observation
N_GATES = 2  # default number of upcoming gates in the observation
GATE_START = 15  # first gate feature
PARAM_NAMES = ("delay_steps", "thrust_scale")
KAPPA_DEFAULT = None  # kappa appended to policy inputs whose obs has no "kappa" (set by train_ppo --kappa)


def gate_block_dim(n_gates: int) -> int:
    """Number of gate features: 10 for the next gate, 11 for each further one."""
    return 10 + 11 * (n_gates - 1)


def obs_dim(n_gates: int = N_GATES, n_pillars: int = N_PILLARS) -> int:
    return GATE_START + gate_block_dim(n_gates) + 3 * n_pillars + 4


def input_dim(n_gates: int = N_GATES, n_pillars: int = N_PILLARS, n_hist: int = 0,
              n_vhist: int = 0, n_param: int = 0, fv: int = 1) -> int:
    return obs_dim_fv(n_gates, n_pillars, fv) + 4 * n_hist + 3 * n_vhist + n_param


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
                 vel_hist=None, params=None, fv: int = 1):
    """features() followed by the older actions; prev_actions has shape (B, 1 + n_hist, 4),
    newest (u_{t-1}) first. Optional vel_hist (B, n_vhist, 3), newest first, and params
    (B, n_param) are appended in that order."""
    if fv == 1:
        x = features(obs, prev_actions[:, 0], xp, n_gates, n_pillars)
    elif fv == 3:
        x = features_v3(obs, prev_actions[:, 0], xp, n_gates, n_pillars)
    else:
        x = features_v4(obs, prev_actions[:, 0], xp, n_gates, n_pillars, world=fv == 5)
    b = prev_actions.shape[0]
    parts = [x, prev_actions[:, 1:].reshape(b, -1).astype(xp.float32)]
    if vel_hist is not None:
        parts.append(vel_hist.reshape(b, -1).astype(xp.float32))
    if params is not None:
        parts.append(params.astype(xp.float32))
    if isinstance(obs, dict) and "kappa" in obs:
        parts.append(xp.asarray(obs["kappa"]).reshape(b, 1).astype(xp.float32))
    elif KAPPA_DEFAULT is not None:
        parts.append(xp.full((b, 1), KAPPA_DEFAULT, dtype=xp.float32))
    return xp.concatenate(parts, axis=-1)


# --------------------------------------------------------------------------------------------
# Feature version 3 (obs-v3, 2026-11-18): the v1 self + gate block, plus inputs that show the
# collision risk directly instead of leaving it to the MLP to derive from raw geometry.
#   0:15            self (as v1)
#   15:15+GB        upcoming gates (as v1, GB = gate_block_dim(n_gates))
#   then 9          next target gate in its own frame (x = crossing normal, flipped for reverse
#                   passes): drone position 3, velocity 3, where the straight velocity line meets the
#                   gate plane (lateral, vertical offset) 2, time to the plane 1 (0 if not approaching)
#   then 7          gate just passed, same frame: position 3, velocity 3, flag 1 (0 before gate 1)
#   then 5 * n_p    pillars sorted by threat (closest approach along the velocity line within
#                   FV3_T s): rel xy (heading frame) 2, sensed 1, closest approach (m, axis) 1,
#                   time of closest approach / FV3_T 1
#   then 5 * FV3_BARS  most threatening gate-frame bars (all gates, observed poses): vector to the
#                   nearest point of the bar (heading frame) 3, closest approach 1, its time / FV3_T 1
#   last 4          previous action
# Closest approach: the drone flies straight at its current velocity for FV3_T s, sampled at
# FV3_K points. Distances are clipped to FV3_DMAX.

FV3_T = 0.6
FV3_K = 7
FV3_BARS = 2
FV3_DMAX = 2.0
# Gate-frame collision boxes in the gate frame (gate.xml), same as reward.py: bar centres (y, z),
# half sizes (y, z), half thickness along the normal.
_FV3_BAR_C = ((0.0, 0.28), (0.0, -0.28), (-0.28, 0.0), (0.28, 0.0))
_FV3_BAR_H = ((0.36, 0.08), (0.36, 0.08), (0.08, 0.36), (0.08, 0.36))
_FV3_BAR_TH = 0.01


def obs_dim_v3(n_gates: int = N_GATES, n_pillars: int = 4) -> int:
    return GATE_START + gate_block_dim(n_gates) + 9 + 7 + 5 * n_pillars + 5 * FV3_BARS + 4


FV4_HALF = 0.2  # half inner opening of a gate (m): corners at +-FV4_HALF along the gate y / z axes
FV4_GATES = 2  # gates whose corners are given (next target and the one after)


def obs_dim_fv(n_gates: int, n_pillars: int, fv: int = 1) -> int:
    if fv == 1:
        return obs_dim(n_gates, n_pillars)
    d = obs_dim_v3(n_gates, n_pillars)
    if fv >= 4:
        d += 12 * FV4_GATES
    if fv == 5:
        d += 6
    return d


def features_v4(obs: dict, prev_action, xp, n_gates: int = N_GATES, n_pillars: int = 4,
                world: bool = False):
    """Feature versions 4 / 5 (2026-11-19): the v3 features with, before the previous action,
    the 4 inner corners of the next FV4_GATES gates of the sequence (position relative to the drone
    in the heading frame, corner order (+y+z, +y-z, -y-z, -y+z) in the gate frame, zeros past the
    last gate) and, for version 5 (world=True), the world-frame position and velocity."""
    x = features_v3(obs, prev_action, xp, n_gates, n_pillars)
    pos = obs["pos"]
    rot = quat_to_mat(obs["quat"], xp)
    yaw = xp.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    c, s = xp.cos(yaw), xp.sin(yaw)

    def heading(v):  # (B, 3)
        return xp.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], axis=-1)

    n_passes = obs["gate_sequence"].shape[-1]
    passed = obs["n_gates_passed"].astype(xp.int32)
    seq = obs["gate_sequence"].astype(xp.int32)
    parts = []
    for k in range(FV4_GATES):
        slot = xp.minimum(passed + k, n_passes - 1)
        gid = _take(seq, slot, xp)
        has = (passed + k < n_passes).astype(pos.dtype)[:, None]
        gp = _take(obs["gates_pos"], gid, xp)
        R = quat_to_mat(_take(obs["gates_quat"], gid, xp), xp)
        for sy, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            corner = gp + FV4_HALF * (sy * R[..., 1] + sz * R[..., 2]) - pos
            parts.append(heading(corner) * has)
    if world:
        parts += [pos, obs["vel"]]
    return xp.concatenate([x[:, :-4], *[p.astype(x.dtype) for p in parts], x[:, -4:]],
                          axis=-1).astype(xp.float32)


def features_v3(obs: dict, prev_action, xp, n_gates: int = N_GATES, n_pillars: int = 4):
    """Policy input of feature version 3, shape (B, obs_dim_v3(n_gates, n_pillars))."""
    pos, vel = obs["pos"], obs["vel"]
    B = pos.shape[0]
    dt = pos.dtype
    base = features(obs, prev_action, xp, n_gates, 1)[:, :GATE_START + gate_block_dim(n_gates)]
    rot = quat_to_mat(obs["quat"], xp)
    yaw = xp.arctan2(rot[:, 1, 0], rot[:, 0, 0])
    c, s = xp.cos(yaw), xp.sin(yaw)

    def h2(v):  # (B, ..., 2) world xy -> heading frame
        cc = c.reshape(c.shape + (1,) * (v.ndim - 2))
        ss = s.reshape(s.shape + (1,) * (v.ndim - 2))
        return xp.stack([cc * v[..., 0] + ss * v[..., 1], -ss * v[..., 0] + cc * v[..., 1]], axis=-1)

    def h3(v):
        return xp.concatenate([h2(v[..., :2]), v[..., 2:3]], axis=-1)

    n_passes = obs["gate_sequence"].shape[-1]
    passed = obs["n_gates_passed"].astype(xp.int32)
    sequence = obs["gate_sequence"].astype(xp.int32)
    direction = obs["gate_sequence_direction"]

    def gate_frame(slot):
        gate_id = _take(sequence, slot, xp)
        sign = _take(direction, slot, xp).astype(dt)
        R = quat_to_mat(_take(obs["gates_quat"], gate_id, xp), xp)
        flip = xp.stack([sign, sign, xp.ones_like(sign)], axis=-1)[:, None, :]
        return _take(obs["gates_pos"], gate_id, xp), R * flip

    def local(R, v):  # R^T v
        return xp.einsum("bji,bj->bi", R, v)

    # next target gate
    gp, R = gate_frame(xp.minimum(passed, n_passes - 1))
    lp, lv = local(R, pos - gp), local(R, vel)
    approach = (lp[:, 0] < 0) & (lv[:, 0] > 0.05)
    tc = xp.where(approach, -lp[:, 0] / xp.maximum(lv[:, 0], 0.05), 0.0)
    tc = xp.clip(tc, 0.0, 2.0)
    cross = xp.clip(lp[:, 1:] + lv[:, 1:] * tc[:, None], -FV3_DMAX, FV3_DMAX)
    nxt = xp.concatenate([lp, lv, cross, tc[:, None]], axis=-1)

    # gate just passed
    gp0, R0 = gate_frame(xp.maximum(passed - 1, 0))
    flag0 = (passed > 0).astype(dt)[:, None]
    prv = xp.concatenate([local(R0, pos - gp0) * flag0, local(R0, vel) * flag0, flag0], axis=-1)

    ts = xp.linspace(0.0, FV3_T, FV3_K).astype(dt)  # (K,)
    traj = pos[:, None, :] + vel[:, None, :] * ts[None, :, None]  # (B, K, 3)

    # pillars (vertical axes): horizontal closest approach
    op = obs["obstacles_pos"]
    P = op.shape[1]
    dp = traj[:, :, None, :2] - op[:, None, :, :2]  # (B, K, P, 2)
    dist = xp.sqrt(xp.sum(dp * dp, axis=-1))  # (B, K, P)
    dmin = xp.min(dist, axis=1)
    tmin = ts[xp.argmin(dist, axis=1)] / FV3_T  # (B, P)
    rel = h2(op[..., :2] - pos[:, None, :2])  # (B, P, 2)
    sensed = obs["obstacles_visited"].astype(dt)
    per = xp.concatenate([rel, sensed[..., None], xp.minimum(dmin, FV3_DMAX)[..., None],
                          tmin[..., None]], axis=-1)  # (B, P, 5)
    # ties (equal distances) broken by index so that numpy and JAX pick the same objects
    order = xp.argsort(dmin + 1e-4 * xp.arange(P, dtype=dt), axis=1)[:, :min(P, n_pillars)]
    pil = xp.take_along_axis(per, order[..., None], axis=1).reshape(B, -1)
    if P < n_pillars:
        pil = xp.concatenate([pil, xp.zeros((B, 5 * (n_pillars - P)), dt)], axis=-1)

    # gate-frame bars of all gates (observed poses)
    bc = xp.asarray(_FV3_BAR_C, dt)
    bh = xp.asarray(_FV3_BAR_H, dt)
    Rg = quat_to_mat(obs["gates_quat"], xp)  # (B, G, 3, 3)
    gpos = obs["gates_pos"]
    G = gpos.shape[1]
    lk = xp.einsum("bgji,bkgj->bkgi", Rg, traj[:, :, None, :] - gpos[:, None])  # (B, K, G, 3)
    dx = xp.maximum(xp.abs(lk[..., 0]) - _FV3_BAR_TH, 0.0)[..., None]  # (B, K, G, 1)
    dyz = xp.maximum(xp.abs(lk[..., None, 1:] - bc) - bh, 0.0)  # (B, K, G, 4, 2)
    d = xp.sqrt(dx * dx + xp.sum(dyz * dyz, axis=-1) + 1e-9)  # (B, K, G, 4)
    bmin = xp.min(d, axis=1).reshape(B, G * 4)
    btmin = (ts[xp.argmin(d, axis=1)] / FV3_T).reshape(B, G * 4)
    l0 = lk[:, 0]  # (B, G, 3) current position in each gate frame
    cp_yz = xp.clip(l0[:, :, None, 1:], bc - bh, bc + bh)  # (B, G, 4, 2)
    cp_x = xp.broadcast_to(xp.clip(l0[:, :, None, 0:1], -_FV3_BAR_TH, _FV3_BAR_TH), cp_yz[..., :1].shape)
    cp = xp.concatenate([cp_x, cp_yz], axis=-1)  # (B, G, 4, 3)
    cw = gpos[:, :, None, :] + xp.einsum("bgij,bgkj->bgki", Rg, cp)
    brel = h3((cw - pos[:, None, None, :]).reshape(B, G * 4, 3))
    bper = xp.concatenate([brel, xp.minimum(bmin, FV3_DMAX)[..., None], btmin[..., None]], axis=-1)
    border = xp.argsort(bmin + 1e-4 * xp.arange(G * 4, dtype=dt), axis=1)[:, :FV3_BARS]
    bars = xp.take_along_axis(bper, border[..., None], axis=1).reshape(B, -1)

    return xp.concatenate([base, nxt, prv, pil, bars, prev_action], axis=-1).astype(xp.float32)
