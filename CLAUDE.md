# CLAUDE.md — context for Claude sessions in this repo

The user writes in Chinese; answer in Chinese. They fly a real Crazyflie in the lab with RL policies and
prefer step-by-step commands they run themselves. The full operator manual is `LAB_DEPLOY.md` (Chinese) —
read it first; this file is background for you.

## What this repo is

- Official `learnsyslab/lsy_drone_racing` at commit 769cc13, byte-identical except:
  - `config/level2.toml`: `controller.file = "level2_rl_kappa_controller.py"`, `env.control_mode = "attitude"`,
    `[[deploy.drones]] id = 2, channel = 80, drone = "cf21B_500"` (the drone is **cf2**).
  - Added: `lsy_drone_racing/control/level2_rl_controller.py`, `level2_rl_kappa_controller.py`,
    `lsy_drone_racing/rl/{__init__,actions,features,policy_np}.py`, `artifacts/race_init/*.npz + README.txt`,
    `fly.sh`, `scripts/analyze_flight_log.py`, `kappa.txt`, `LAB_DEPLOY.md`, this file.
- Do **not** modify other official files, and never the estimator (`drone-estimators` package in `.pixi/`).
  Choosing `--estimator_type` / `--drone_name` on the command line is allowed (official docs).
- Branches: `adr_project` repo branch `real-flight` (code) and the fork `Xiaolong-Li1904/lsy_drone_racing`
  `main` hold the same code. Flight logs go to `adr_project` branch `flight-logs` only.
- Files must stay LF (`.sh` break on Linux with CRLF); `tools/*.sh` and `fly.sh` are mode 755.

## Controller and policies

- `Level2RLKappaController`: numpy MLP inference (feature_version 3, n_hist 2, n_vhist 1, 4 gates, 4 pillars),
  outputs [roll, pitch, yaw, collective thrust] in attitude mode at 50 Hz.
- Policy: env `ADR_RL_POLICY` (path) overrides `default_policy` = `artifacts/race_init/general_tilt1.0.npz`.
- kappa (0 safest .. 1 fastest): `ADR_KAPPA_FILE` (re-read each run) > `ADR_KAPPA` > 0. For kappa policies the
  roll/pitch limit widens from 1.0 rad to `kappa_tilt_hi` with kappa (as in training).
- Policies (level2 track, seeds 1–100, training-side numbers):
  - `l2_robust_tilt1.15` (= kFclr_4500): delay 0–2 trained. k1: ideal 97 %/3.94 s, d1 98 %/4.07, d2 90 %/4.29.
  - `l2_delay1_tilt1.15` (= kD1_6000): delay fixed 1 step. d1: k0 100 %/4.11 .. k1 99 %/3.88. At delay 0
    k1 only 62 % (it anticipates the delay) — do not judge it at delay 0. Delay 2 not evaluated.
  - `l2_kp1_tilt1.15` (= kP1_12000, newest): fractional delay 0.4-0.8 step (real equivalent delay measured ~0.6),
    thrust 0.9-1.1. Delay 0.6: k0 98 %/3.84 .. k1 93 %/3.64; held-out 101-200 k1 93 %/3.68. Fall back to l2_robust
    if thrust is far off. Sim check needs a fractional-delay wrapper ((1-f) u[t-k] + f u[t-k-1]).
  - `general_tilt1.0` (= f10_1500): generalist, no kappa, ~5.5 s.
  - `l2_risky_tilt1.4` (= kDdrk_3000): tilt up to 1.4 rad > ~1.15 rad hover limit; may lose altitude.
- Recommended plan: l2_robust k0 ×1–2 → l2_kp1 k0 → raise kappa 0.3/0.5/0.7/1.0 on the better one.

## Flight flow (official, unchanged)

`deploy.py`: reset (track/start-position checks → connect → arm) → control loop starts immediately (no
confirmation, takes off directly) → ends when all 5 gate passages are done (`gate_order = [1,2,3,4,2]`),
then return-to-start and stop. Ctrl+C = immediate emergency stop (no return). `Track time` = first command
to last gate.

## Flight logs and delay

- The controller records every step and saves `flight_logs/<stamp>_<policy>_k<kappa>.npz` at the end / on
  Ctrl+C (atexit). Keys: t, wall, x (policy input), raw, u, action, obs_* (all observation keys), meta (json).
- `fly.sh` runs deploy, then commits new logs to local branch `flight-logs` with a separate index (code branch
  untouched) and pushes in the background to adr_project. `FLY_DEPLOY_CMD=true ./fly.sh` = upload only.
- `scripts/analyze_flight_log.py`: fits measured roll/pitch to commanded roll/pitch shifted by d steps;
  `equiv_delay_steps = raw lag - 1.33` (sim baseline). Sim check: added delay 0/1/2 → 0.00/1.02/1.90.
  ~0.4-0.8 → l2_kp1; ~1 → l2_delay1; ~0 or ≥1.5 → l2_robust. The estimator type (`diff` default, `ukf` planned) affects it.

## Status (2026-11-25)

- First lab session: 2 flights were made with the official `python scripts/deploy.py` command; no logs reached
  GitHub (unknown whether they were flown from `~/adr_project`). If `~/adr_project/flight_logs/` has files on
  the lab PC, upload them with `FLY_DEPLOY_CMD=true ./fly.sh`.
- The training session reports a measured real equivalent delay of ~0.6 step (2026-11-25); kP1 was trained for it.
  Confirm with our own logs (`scripts/analyze_flight_log.py`).
