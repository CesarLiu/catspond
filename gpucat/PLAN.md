# 在 GPUDrive 上跑 CAT 的实验（分支 `gpudrive-ppo`）

**目标：** MetaDrive 上的一个训练要 14–19 小时，而且 GPU 基本闲着。把 CAT 的设定搬到 GPUDrive（Madrona，几百个世界在 GPU 上并行），用 PPO 训练，以便：
- 每个设置跑 5–10 个种子；
- 补齐消融实验；
- 在第二个模拟器上复现 MetaDrive 的结论。

MetaDrive 的结果仍然是忠实复现 CAT 的主线；这里的结果换了观测、动力学和算法，不能直接和 CAT 论文或 MetaDrive 的结果比较。

**环境：** GPUDrive 装在 `~/gpudrive`，安装和编译过程见 `responsibility/GPUDRIVE_NOTES.md`。运行方式：

```bash
PYTHONPATH=~/gpudrive/build:~/gpudrive LD_LIBRARY_PATH=~/cuda-12.4/lib64 ~/gpudrive/.venv/bin/python -m scripts.gpucat.<脚本>
```

场景导出（`scripts/gpucat/export_scenes.py`）在本仓库的 Python 3.9 环境里跑。

## 里程碑

| 里程碑 | 内容 | 估计 |
|---|---|---|
| M0 可行性 | 只控制自车；运行时注入对手计划；碰撞信息；单独重置部分世界；带网络时的吞吐量 | **完成（2026-10-07）**，见下文 |
| M1 GPU 上的 CAT | 每个场景预先算好 32 条候选（DenseTNT），以及公平对手需要的量（对手的运动集合、自车的运动集合）；每个回合根据自车最近的轨迹，用 torch 批量选出对手；和 CAT 原版逐场景核对选择一致 | 1–2 天 |
| M2 环境和 PPO | 自动重置；仿照 MetaDrive 写奖励（沿路线前进、到达、出界、碰撞）；PPO；评测循环 | 2–3 天 |
| M3 验证 | 用 replay 和 cat 训练做合理性检查，看 MetaDrive 上的主要结论能不能复现 | 1 天 |
| M4 实验 | replay、cat、cat_fair、cat_rss 各 5–10 个种子，加消融；cat_share 需要批量的碰撞归因，放到后面 | – |

## M0 的结果（`scripts/gpucat/m0_feasibility.py`，测试场景 400–463，`logs/gpucat/m0.json`）

**场景：** `gpucat/scenes.py` 和 `scripts/gpucat/export_scenes.py` 导出了 500 个场景，在 `logs/gpucat/scenes`：
- 物体的顺序是自车、对手，其余按离自车的距离排；
- 除自车以外全部标为专家，按日志回放；
- 对手的终点挪到 5 km 外，否则 GPUDrive 会在它接近日志里的终点时把它移走。

GPUDrive 每个世界最多 64 个智能体，有 161 个场景超过这个数，离自车最远的被丢掉。

| 检查 | 结果 |
|---|---|
| 只控制自车 | 64/64 个世界正好控制 1 个智能体，而且就是自车（按轨迹 id 核对） |
| 找到对手 | 64/64。默认的 `all_non_trivial` 只创建第 0 步就有效的智能体，会漏掉 7 个晚出现的对手（第 1–10 步出现），要用 `init_mode="all_objects"` |
| 运行时注入对手计划 | 把计划写进 `expert_trajectory_tensor`（零拷贝的 GPU 张量）。对手出现之前的步保留原数据，有效标志设为 0。回放和计划完全一致（误差 0.0 m），晚一步：第 k 步之后对手在计划的第 k−1 行，和 CAT 在 MetaDrive 里一样 |
| 自车按专家动作回放（delta_local） | 到达终点之前，误差最大 3 mm |
| 单独重置一半世界 | 写入的计划保留；被重置的对手回到计划的第一行（误差 0.0 m）；其他 32 个世界完全不动 |
| 碰撞 | GPUDrive 报 43/64 次；离线用三圆近似算是 62 次，用实际大小的矩形算是 60 次。原因是 GPUDrive 把车辆的碰撞框长宽都乘了 0.7（`src/consts.hpp:25` 的 `vehicleLengthScale`）。离线用 0.7 倍的矩形重算，也是 43 次，53/64 个世界的第一次碰撞步数完全相同，其余基本只差 1 步 |

**吞吐量（L4；带一个 PPO 规模的网络，actor 和 critic 各两层 256 单元；观测 2,984 维；另有 1 个 MetaDrive 训练在跑）：**

| 世界数 | 每秒世界步 | 显存 |
|---|---|---|
| 64 | 974 | 8.2 GB |
| 128 | 1,625 | 10.8 GB |
| 256 | 2,352 | 15.9 GB |
| 384 | 3,085 | 20.9 GB |

按 256 个世界算，100 万步的仿真约 7 分钟；MetaDrive 上一个 100 万步的训练要 14–19 小时。

**待定：碰撞框的缩放。** GPUDrive 默认 0.7 倍，是为了减少 WOMD 标注框带来的误报，但这让碰撞比 MetaDrive 和离线分析宽松（测试场景里 43 对 60 次）。有两个选择：
- 改成 1.0 再重新编译 GPUDrive，和 MetaDrive 以及不可避免性等离线分析一致；
- 保留 0.7，并在结论里说明。
