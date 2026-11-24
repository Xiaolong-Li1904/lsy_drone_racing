# 实验室实飞手册（f10 / kappa 网络）

控制器只有一个：`lsy_drone_racing/control/level2_rl_kappa_controller.py`（`config/level2.toml` 已设为默认，
`control_mode = "attitude"`）。换网络、调激进程度都**不用改控制器**，只改网络文件和 kappa。

## 0. 克隆

```bash
git clone <仓库地址> && cd <仓库名>
```

第一次运行 `pixi run -e deploy ...` 时会自动安装环境并编译 `ros_ws`（需要联网，等一会儿）。

## 1. 选飞机：`config/level2.toml`

```toml
[[deploy.drones]]
id = 10          # 机身上的十进制编号
channel = 100    # 这架飞机的无线电信道
drone = "cf21B_500"
```

代码里的飞机名是 `cf{id}`，**不补 0**：`id = 2` → `cf2`，`id = 10` → `cf10`。
estimator 的 `--drone_name` 必须和它完全一致。

## 2. 三个终端

```bash
# 终端1: Vicon → ROS2
pixi run -e deploy mocap

# 终端2: 状态估计器（名字 = cf + id）
pixi run -e deploy estimator --drone_name cf10

# 终端3: 起飞
pixi shell -e deploy
export ADR_RL_POLICY=artifacts/race_init/l2_robust_tilt1.15.npz
export ADR_KAPPA_FILE=$PWD/kappa.txt
echo 0 > kappa.txt
python scripts/deploy.py --config level2.toml
```

摆门时可以在终端3先运行 `python scripts/nominal_frame_publisher.py --config level2.toml`，
在 RViz 里添加 MarkerArray `/nominal_frames` 对照名义位置。

**不稳就 Ctrl+C**：会立即急停（不会先飞回起点）。

## 3. 切换网络

方法 B（推荐，不改代码）：在终端3设置环境变量，下一次 `deploy.py` 生效。

```bash
export ADR_RL_POLICY=artifacts/race_init/l2_robust_tilt1.15.npz
```

方法 A（改代码）：`level2_rl_kappa_controller.py` 里的 `default_policy`，只改文件名。
没设 `ADR_RL_POLICY` 时才用这个默认值；两者都有时**环境变量优先**。

```python
default_policy = Path(__file__).parents[2] / "artifacts/race_init/general_tilt1.0.npz"
```

| 网络 (`artifacts/race_init/`) | 说明 |
|---|---|
| `l2_delay1_tilt1.15.npz` | 最新，专门针对 **1 步延迟**训练，1 步延迟下最快（kappa 0→1：4.11→3.88 s）；无延迟时反而变差 |
| `l2_robust_tilt1.15.npz` | **推荐**，延迟 0–2 步都训练过，适应面最广（kappa 1：理想 97 %/3.94 s，延迟1 98 %/4.07 s，延迟2 90 %/4.29 s） |
| `general_tilt1.0.npz` | **代码里的默认网络**。通用保底（不是专门为这条赛道训练的），最慢但最保守，kappa 对它无效 |
| `l2_risky_tilt1.4.npz` | kappa > 0.5 可能掉高，慎用 |

训练链：general → l2_robust → l2_delay1（中间两个旧版本已删除）。
原始训练名（kF115、kFclr、kD1 等）的对照见 `artifacts/race_init/README.txt` 开头。

文件名里的 `tilt` 是最大倾角（rad）。1.15 是真机能维持高度的上限，`tilt1.4` 超过了它。
这些网络都是在 `level2.toml` 的赛道上训练的，**不要改门和柱子的位置**。

## 4. 调激进程度 kappa（0 = 最稳，1 = 最快）

`kappa.txt` 在每次 `deploy.py` 开始时重新读取，改文件即可：

```bash
echo 0.3 > kappa.txt && python scripts/deploy.py --config level2.toml
```

建议 0 → 0.3 → 0.5 → 0.7 → 1.0，每档飞稳了再往上加。

优先级：`ADR_KAPPA_FILE` 文件 > 环境变量 `ADR_KAPPA` > 代码默认值 0。

## 5. 起飞前检查

- `level2.toml` 的 `id` / `channel` 是这架飞机的
- RViz 里能看到 `cf<id>`（看不到 = Vicon 里没勾选这架飞机）
- 终端2 的 estimator 在打印频率信息（停在 `Waiting for initial measurement` = Vicon 没看到飞机）
- 电池：红灯常亮 = 没电，红灯闪 = 电量够
- `echo $ADR_RL_POLICY` 和 `cat kappa.txt` 确认网络和档位
- 不摆门空飞练习：`level2.toml` 里 `real_track_objects = false`

## 5b. 飞行记录（自动）

每次 `deploy.py` 都会自动记录全部飞行数据（每步：时间、位置/速度/姿态/角速度、门和柱子、网络输入、
网络输出、发出的指令），飞完或 Ctrl+C 时存到 `flight_logs/<日期>_<时间>_<网络>_k<kappa>.npz`。
不影响控制（每步多约 0.05 ms）。`export ADR_LOG_DIR=` 置空可关闭。

飞完看结果 / 估计真机延迟：

```bash
python scripts/analyze_flight_log.py              # 分析 flight_logs/ 下全部记录
```

每次飞行一行：用时、过门数、控制循环是否超时（`loop_steps_over_25ms`）、最大倾角/速度、
`equiv_delay_steps` = 相当于仿真里的几步延迟（和训练的 delay 同一单位，1 步 = 20 ms，误差约 ±0.3 步）。
最后一行是所有记录的中位数。约 1 → 用 `l2_delay1`；接近 0 或 ≥ 2 → 用 `l2_robust`。
记录要带回来：`git add -f flight_logs && git commit -m "flight logs" && git push`。

## 6. 本地仿真（Windows 也能跑）

```powershell
$env:ADR_RL_POLICY = "artifacts/race_init/l2_robust_tilt1.15.npz"
$env:ADR_KAPPA = "0.5"
pixi run python -W ignore scripts/sim.py --config level2.toml --render True --n_runs 3
```
