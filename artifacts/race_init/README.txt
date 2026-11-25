=== File names in this folder (renamed 2026-11-24; the notes below use the original training names) ===
general_tilt1.0.npz       = f10_policy_1500.npz           (generalist, no kappa, tilt 1.0; controller default)
l2_robust_tilt1.15.npz    = kFclr_level2_policy_4500.npz  (kappa, tilt 1.15, delay 0-2)
l2_delay1_tilt1.15.npz    = kD1_level2_policy_6000.npz    (kappa, tilt 1.15, delay fixed 1 step)
l2_risky_tilt1.4.npz      = kDdrk_level2_policy_3000.npz  (kappa, tilt up to 1.4 -- may lose altitude on the real drone)
l2_kp1_tilt1.15.npz       = kP1_level2_policy_12000.npz   (kappa, tilt 1.15, fractional delay 0.4-0.8 step, thrust 0.9-1.1; newest)
Chain: f10 -> kF115 -> kFc -> robust (kFclr) -> delay1 (kD1) -> kE2 -> kG3 -> kH3 -> kp1 (kP1); only general, robust, delay1, kp1 (and risky) are in this repo. Training checkpoints (.pkl, D_r12 / K_r14 meta inits) are not in this repo (flight only).

Race-day starting points (2026-11-22).
D_r12_step100.pkl + .flags : meta init (Reptile+ES, r12 step 100) - fastest training (~300 it to reach f10-finetune@750), less reliable in ideal cond.
K_r14_step100.pkl + .flags : meta init (r14 step 100, fail weight 20) - success >= f10-finetune in delay conditions, faster in all.
f10_ckpt_1500.pkl / f10_policy_1500.npz : generalist (no fine-tune fallback, L3 sim).
Fine-tune: train_ppo.py --bc X.pkl --config <track> --iterations 750 --actor-lr-cosine --actor-lr-warmup 20 --kl-bc-decay 300 --r-crash-first-gate 100 --r-pillar-clear 0.25 --delay-max 1 --thrust-scale-min 0.85 --thrust-scale-max 1.1 $(cat X.flags)

--- 2026-11-23: kappa (safe -> fast) policy for the 20-min real session ---
kDdrk_level2_policy_3000.npz : kappa policy trained on level2.toml (example). Recipe for a NEW track (from D):
  train_ppo.py --bc D_r12_step100.pkl --config <track> --iterations 1500 --actor-lr-cosine --actor-lr-warmup 20     --delay-max 2 --thrust-scale-min 0.8 --thrust-scale-max 1.15 --kl-bc-decay 300 --r-crash-first-gate 100     --r-pillar-clear 0.25 $(cat D_r12_step100.flags) --kappa --kappa-safe crash_penalty=5000,gate_time=60,time=0.01     --kappa-fast crash_penalty=200,gate_time=300,time=0.3 --kappa-tilt 0.0:1.4 --kappa-dr 0.5
  (1500 it cosine ~5 min on a 5080 ~= 3000 it minus 0.03 s; on 8 new tracks speed is monotone in kappa:
   ideal kappa 0 96 %/4.46 s .. kappa 1 91 %/4.14 s)
Fly with lsy_drone_racing/control/level2_rl_kappa_controller.py: ADR_RL_POLICY=<npz>, ADR_KAPPA=<0..1> or
  ADR_KAPPA_FILE=<file with one float, re-read every episode start> to change kappa between laps.
  NOTE kappa > 0.5 widens the roll/pitch limit above 1.0 rad (up to 1.4 at kappa 1): above ~1.15 rad the real
  drone cannot hold altitude at full thrust -- use high kappa only if the real flights look safe.

--- 2026-11-23 evening: tilt-capped kappa policy for the real drone ---
kF115_level2_policy_1500.npz : kappa policy on level2.toml FROM f10 (default hp, no flags), same kappa recipe as above but
  --kappa-tilt 0.0:1.15 (never above the ~1.15 rad hover limit), 1500 it cosine. level2 seeds 1-100 (kappa 0 .. 1):
  ideal 98 %/4.28 .. 92 %/4.04 | d1t09 100 %/4.43 .. 97 %/4.18 | d1t11 100 %/4.37 .. 93 %/4.16 | d2t085 96 %/4.60 .. 88 %/4.44
  = as fast as kDdrk (tilt 1.4) and more reliable at delay 2 -> preferred real-flight kappa policy.
kFc_level2_policy_3000.npz   : kF115 continued +1500 it (same recipe). -0.06..-0.12 s vs kF115 at every kappa, success same/higher.
kFclr_level2_policy_4500.npz : kFc continued +1500 it with --actor-lr 1e-4. Slightly better again (ideal k1 97 %/3.94 s,
  d1t09 k1 98 %/4.07, d2t085 k1 90 %/4.29). kFc2 (+1500 at lr 3e-5) = no gain -> speed saturates ~3.95 s at tilt 1.15.

--- 2026-11-24: delay-1 specialist (real latency ~20 ms = 1 control step) ---
kD1_level2_policy_6000.npz : kappa policy, tilt cap 1.15 rad (kappa_tilt_hi 1.15, k0 0.0). kFclr continued +1500 it with the
  action delay FIXED at 1 step: --delay-min 1 --delay-max 1 (--delay-min is a new train_ppo flag on the cluster), thrust 0.8-1.15,
  --actor-lr 1e-4, otherwise the kF115 recipe (--kappa-safe crash_penalty=5000,gate_time=60,time=0.01 --kappa-fast
  crash_penalty=200,gate_time=300,time=0.3 --kappa-tilt 0.0:1.15 --kappa-dr 0.5). Chain: f10 -> kF115 -> kFc -> kFclr -> kD1 (6000 it).
  level2 seeds 1-100, delay 1 (kappa 0 .. 1): thrust 1.0 100 %/4.11 .. 99 %/3.88 | thrust 0.9 99 %/4.16 .. 98 %/3.93 |
  thrust 1.1 100 %/4.11 .. 97 %/3.88  (-0.14..-0.16 s vs kFclr at every kappa, success unchanged).
  Delay 0 (no latency): kappa 0 96 %/4.04 .. kappa 1 62 %/3.80 -- specialised for 1-step delay, do not judge it at delay 0.
  Delay 2 not evaluated. Aggressive fast-end rewards on top (kB1-kB4) gave <= 0.02 s more at lower success.

--- 2026-11-25: real equivalent delay ~0.6 step (measured on the real drone, per the training session) ---
kP1_level2_policy_12000.npz (= l2_kp1_tilt1.15.npz) : kH3 continued +1500 it (cumulative 12000) with a FRACTIONAL delay drawn
  per episode in [0.4, 0.8] step (train_ppo --delay-max 1 --delay-frac 0.4,0.8: env gets (1-f) u[t-k] + f u[t-k-1]),
  thrust 0.9-1.1, lr 1e-4, kappa-fast crash_penalty=10,gate_time=500,time=1.5, kappa-safe as before, kappa-dr 0.5, tilt cap 1.15.
  Intermediate runs kE2 / kG3 / kH3 (delay fixed 1, more aggressive fast end, kH3 thrust narrowed to 0.9-1.1) are not in this repo.
  Seeds 1-100, delay 0.6 (kappa 0 .. 1): thrust 1.0 98 %/3.84 .. 93 %/3.64 | thrust 0.9 97 %/3.96 .. 95 %/3.74 |
  thrust 1.1 99 %/3.86 .. 94 %/3.66. Delay 0.4: k0 99 %/3.84, k1 93 %/3.62. Delay 0.8: k0 97 %/3.88, k1 93 %/3.68.
  Held-out seeds 101-200, delay 0.6: thrust 1.0 99 %/3.88 .. 93 %/3.68 | thrust 0.9 99 %/3.98 .. 92 %/3.76.
  For comparison the delay-1 line (kH3) at delay 0.6: k1 only 82 % (68 % at delay 0.4).
  Trained for +-10 % thrust only: if the real thrust is far off, fall back to l2_robust (kFclr, 97-98 % at delay 0.4-0.8, ~0.3 s slower).
  Not evaluated at delay 0. Sim reproduction needs a fractional-delay wrapper (an integer 1-step delay is not its training condition);
  the training repo uses level3_rl_stress_controller with ADR_STRESS_DELAY=0.6 (not part of this flight repo).
