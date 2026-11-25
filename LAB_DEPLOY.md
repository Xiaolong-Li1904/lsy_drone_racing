# 实验室实飞手册（kappa 网络）

飞机：**cf2**（`config/level2.toml`：`id = 2`，`channel = 80`，`drone = "cf21B_500"`）。
控制器只有一个：`lsy_drone_racing/control/level2_rl_kappa_controller.py`（`level2.toml` 已设为默认，
`control_mode = "attitude"`）。换网络、调激进程度都**不用改代码**，只改环境变量和 `kappa.txt`。

## 0. 第一次在一台电脑上用（只做一次）

```bash
cd ~
git clone -b real-flight --single-branch https://github.com/Xiaolong-Li1904/adr_project.git
cd ~/adr_project

git config --global user.name  "Xiaolong-Li1904"
git config --global user.email "Xiaolong-Li1904@users.noreply.github.com"
git config --global credential.helper store
```

- 没有 pixi：`curl -fsSL https://pixi.sh/install.sh | bash`，然后重开终端。
- 第一次 `pixi run -e deploy ...` 会自动安装环境（含 ROS2）并编译 `ros_ws`，需要联网，等几分钟。
- 第一次上传飞行记录时会问用户名和密码：用户名 `Xiaolong-Li1904`，密码填 **GitHub token**（不是登录密码）。
  之后不再问（token 明文存在本用户的 `~/.git-credentials`）。
- 换另一台电脑登同一个用户：`ls ~/.git-credentials ~/adr_project` 都在 → 什么都不用做；
  不在（用户目录不共享）→ 重新做本节。
- 已经 clone 过：`cd ~/adr_project && git pull` 更新代码。

可选，确认这台电脑能跑：`pixi run python scripts/sim.py --config level2.toml --n_runs 1`，最后 `Finished: True` 即可。

## 1. 三个终端（每个都先 `cd ~/adr_project`）

```bash
# 终端1: Vicon → ROS2（一直开着）
pixi run -e deploy mocap
```

```bash
# 终端2: 状态估计器（一直开着）。名字 = cf + id，不补 0
pixi run -e deploy estimator --drone_name cf2 --estimator_type ukf
```

- 不加 `--estimator_type` 时默认是 `diff`。可选 `ukf` / `ekf` / `diff`（官方允许的命令行参数）。
  **一天里只用同一种**，换了的话记下哪几圈用的哪种（飞行记录里不会存这一项）。
- 估计器是官方依赖包（drone-estimators，装在 `.pixi/` 里），**不改它的代码**。
- 一直停在 `Waiting for initial measurement` = Vicon 没看到飞机（Vicon 软件里勾选 cf2）。

```bash
# 终端3: 起飞。前 4 行每次打开终端只做一次
pixi shell -e deploy
export ADR_RL_POLICY=artifacts/race_init/l2_robust_tilt1.15.npz
export ADR_KAPPA_FILE=$PWD/kappa.txt
echo 0 > kappa.txt

./fly.sh                     # 每一圈只输入这一行
```

- `pixi shell -e deploy` = 让这个终端进入 deploy 环境（提示符前出现 `(lsy_drone_racing:deploy)`）。
  只在打开终端时做一次；换网络、改 kappa、连飞多圈都**不用重新进**。关了终端或 `exit` 后要重新进并重新 `export`。
- `./fly.sh` = 官方 `python scripts/deploy.py --config level2.toml` + 飞完自动上传飞行记录。
  两者二选一，**不要写在一起**。用官方命令也能飞，记录只存本地（事后补传见第 5 节）。
- 注意是 `./fly.sh`（点、斜杠），不是 `.fly.sh`。

## 2. 按下回车后会发生什么（和官方完全一样）

1. 检查门/柱子位置（摆偏太多 → 报错退出，不起飞）
2. 检查飞机是否在起点附近（不在 → 报错退出，不起飞）
3. 连接飞机、解锁电机（1–2 秒）
4. **网络立刻开始控制，飞机直接起飞并比赛**（没有确认，也没有先悬停）
5. 结束：
   - 过完最后一个门 → 自动飞回起点上方，然后停桨。终端打印 `Track time: x.xxx s`
   - **Ctrl+C → 立即急停，电机停转，飞机会掉下来**（不飞回起点）
   - 撞了 / 飞出边界 / 断线 → `Task not completed`

用时 = 从网络发出第一个指令到穿过**最后一个门**（门顺序 `gate_order = [1, 2, 3, 4, 2]`，共 5 次过门，2 号门过两次）。

**按回车前**：飞机在起点、电池够（红灯闪 = 够，红灯常亮 = 没电）、场地没人、手放在 Ctrl+C 上。

**命令一次只粘贴一行。** 本手册里连着写的几行 `./fly.sh` 是“每圈输入什么”，整段粘贴会一圈结束马上开始下一圈。

## 3. 切换网络

在终端3改 `ADR_RL_POLICY`，下一次 `./fly.sh` 生效（不用重启终端1/2，不用重新 `pixi shell`）：

```bash
export ADR_RL_POLICY=artifacts/race_init/l2_delay1_tilt1.15.npz
echo 0 > kappa.txt
./fly.sh
```

| 网络 (`artifacts/race_init/`) | 说明 |
|---|---|
| `l2_kp1_tilt1.15.npz` | **最新、最快**。按真机实测等效延迟约 **0.6 步**训练（延迟 0.4–0.8 步、推力 ±10 %）。延迟 0.6：kappa 0 98 %/3.84 s，kappa 1 93 %/3.64 s（比 l2_delay1 快约 0.25 s）。推力偏差超过 ±10 % 时退回 l2_robust |
| `l2_robust_tilt1.15.npz` | **推荐起步**。延迟 0–2 步都训练过（kappa 1：无延迟 97 %/3.94 s，延迟1 98 %/4.07 s，延迟2 90 %/4.29 s） |
| `l2_delay1_tilt1.15.npz` | 专门针对 **1 步延迟**（20 ms）（kappa 0→1：100 %/4.11 → 99 %/3.88 s）；**无延迟时 kappa 1 只有 62 %**；延迟 0.6 时这一系 kappa 1 约 82 % |
| `general_tilt1.0.npz` | 不设 `ADR_RL_POLICY` 时的默认。通用保底，最慢（约 5.5 s），kappa 对它无效 |
| `l2_risky_tilt1.4.npz` | 倾角上限 1.4 rad，超过真机能维持高度的 1.15，kappa > 0.5 可能掉高，慎用 |

- 确认当前设置：`echo $ADR_RL_POLICY; cat kappa.txt`（`./fly.sh` 第一行也会打印 `[fly] policy=... kappa=...`）
- 只飞一次某个网络：`ADR_RL_POLICY=artifacts/race_init/general_tilt1.0.npz ./fly.sh`
- 回到默认网络：`unset ADR_RL_POLICY`
- 改代码里的默认网络：`level2_rl_kappa_controller.py` 的 `default_policy`（环境变量优先）。
- 文件名里的 `tilt` = 最大倾角（rad）。网络都是在 `level2.toml` 的赛道上训练的，**不要改门和柱子的位置**。
- 原始训练名（kFclr、kD1、kP1 等）和完整测试数据见 `artifacts/race_init/README.txt`。

## 4. 调激进程度 kappa（0 = 最稳，1 = 最快）

```bash
echo 0.3 > kappa.txt && ./fly.sh
```

`kappa.txt` 每次 `./fly.sh` 开始时读取。建议 0 → 0.3 → 0.5 → 0.7 → 1.0，每档飞稳再加；**换网络时 kappa 回到 0**。
优先级：`ADR_KAPPA_FILE` 文件 > 环境变量 `ADR_KAPPA` > 默认 0。

### 建议飞行顺序

| 圈 | 网络 | kappa | 目的 |
|---|---|---|---|
| 1–2 | l2_robust | 0 | 确认整套系统、测延迟 |
| 3 | l2_kp1 | 0 | 和 robust 对比 |
| 之后 | 飞得好的那个 | 0.3 → 0.5 → 0.7 → 1.0 | 提速 |

延迟（第 5 节 `equiv_delay_steps`）约 0.4–0.8 步 → `l2_kp1`；约 1 步 → `l2_delay1`；接近 0 或 ≥ 1.5 → `l2_robust`。
新网络晃动、过冲或掉高（推力不对）→ 退回 `l2_robust`。

## 5. 飞行记录和上传

- 每次飞行自动记录全部数据（每步：时间、位置/速度/姿态/角速度、门和柱子、网络输入/输出、发出的指令），
  飞完或 Ctrl+C 时存到 `flight_logs/<日期>_<时间>_<网络>_k<kappa>.npz`（约 80 KB）。不影响控制（每步 +0.05 ms）。
  用官方命令飞也会记录。`export ADR_LOG_DIR=` 置空可关闭。
- `./fly.sh` 飞完把新记录提交到本地 `flight-logs` 分支，并在后台推到
  **adr_project 的 `flight-logs` 分支**（固定这个仓库，不管从哪 clone 的；代码分支不受影响）。
- 上传失败（没网、token 不对）不影响飞行，下次 `./fly.sh` 一起补传。状态：`cat .git/flight-logs-push.log`。
- 只上传不起飞（例如用官方命令飞过的）：`FLY_DEPLOY_CMD=true ./fly.sh`（在 `pixi shell -e deploy` 里）
- 手动补传：`git push https://github.com/Xiaolong-Li1904/adr_project.git flight-logs`

看结果 / 估计真机延迟：

```bash
python scripts/analyze_flight_log.py              # flight_logs/ 下全部记录，每次飞行一行 + 中位数
```

`equiv_delay_steps` = 相当于仿真里几步延迟（训练 delay 同一单位，1 步 = 20 ms，误差约 ±0.3 步；
仿真验证 0/1/2 步 → 0.00/1.02/1.90）。`duration_s` 比 `Track time` 少约 1 步。
`loop_steps_over_25ms` > 0 = 控制循环卡顿。

## 6. 起飞前检查

- `level2.toml`：`id = 2`，`channel = 80`
- RViz 里能看到 `cf2`（看不到 = Vicon 里没勾选）
- 终端2 在打印频率信息
- 电池、场地、Ctrl+C
- `echo $ADR_RL_POLICY` 和 `cat kappa.txt`
- `pwd` 是 `~/adr_project`（在别的文件夹起飞 = 不是我们的网络，也没有记录）
- 不摆门空飞练习：`level2.toml` 里 `real_track_objects = false`
- 摆门参考：`python scripts/nominal_frame_publisher.py --config level2.toml`，RViz 添加 MarkerArray `/nominal_frames`

## 7. 换飞机

改 `config/level2.toml` 的 `id` / `channel`，终端2 的 `--drone_name` 改成 `cf<id>`（不补 0：`id = 2` → `cf2`）。

## 8. 本地仿真（Windows 也能跑）

```powershell
$env:ADR_RL_POLICY = "artifacts/race_init/l2_robust_tilt1.15.npz"
$env:ADR_KAPPA = "0.5"
pixi run python -W ignore scripts/sim.py --config level2.toml --render True --n_runs 3
```

仿真用时按“步数 ÷ 50 Hz”，也停在最后一个门。真机部署（`-e deploy`）只能在 Linux 上。
