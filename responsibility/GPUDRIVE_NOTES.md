# GPUDrive 可行性摸底（L4 机器，2026-10-04）

**问题：** 能不能用 GPUDrive（github.com/Emerge-Lab/gpudrive，ICLR 2025）代替 MetaDrive，加快 RL 训练？

**结论：**
- 能装，CAT 的场景也能转过去并在 GPUDrive 里跑起来。
- 速度上，SMART 跑完后重测：256 个世界时 CAT 场景约 2,100 世界步/s，即每秒 18,700 车辆步，约是现在 5 个 MetaDrive 训练加起来（约 175 步/s）的 12 倍，按受控车辆算约 100 倍。远不到宣传的每秒百万步：显存最多放下 256 个世界，而且 5 个 RL 训练仍占着 CPU 和 5.8 GB 显存。
- 最大的语义差别是**信号灯**：GPUDrive 没有信号灯模型，它自己的数据转换会直接丢掉所有带信号灯的场景。CAT 的 500 个场景里有 265 个带信号灯。

## 安装（都在用户目录下，没有改动系统）

| 步骤 | 做法 | 遇到的问题 |
|---|---|---|
| CUDA | GPUDrive 只支持 CUDA 12.2–12.4，系统装的是 12.9。用 NVIDIA 的 runfile 只装 toolkit，装到 `~/cuda-12.4`（不需要 root，驱动 580 兼容） | – |
| Python 依赖 | `uv sync --frozen --no-install-project --no-default-groups`，得到 `~/gpudrive/.venv`（Python 3.11，torch 2.6.0+cu124） | 默认的依赖组里有 pufferlib，编译失败；基准测试用不到它 |
| 编译 | `cmake -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5`，指向 CUDA 12.4，`nice -n 15 ninja -j2`，不和实验抢 CPU | 机器上没有 `make`，所以用 pip 装的 Ninja；GPUDrive 固定用 cmake 4.0，而子模块 meshoptimizer 声明的最低版本低于 3.5，要加这个 policy 参数 |
| 使用 | 不用 `pip install -e .`（会用全部核重新编译），而是 `PYTHONPATH=~/gpudrive/build:~/gpudrive`，再加上 `LD_LIBRARY_PATH=~/cuda-12.4/lib64`（Madrona 运行时要用 NVRTC 编译 GPU 内核） | 第一次启动编译内核约 150 s，之后用缓存（`~/gpudrive/gpudrive_cache`） |
| 数据 | GPUDrive_mini：1000 个训练场景、300 个测试和验证场景，2.4 GB，在 `~/gpudrive/data/processed` | Hugging Face 匿名下载被限流（HTTP 429），等一会儿续传即可 |
| 场景转换 | `trimesh` 加 `python-fcl`：GPUDrive 用它们标记"专家"车辆 | – |

## CAT 场景的转换

`responsibility/gpudrive_export.py` 和 `scripts/responsibility/export_gpudrive.py` 逐字段照搬 GPUDrive 自己的 WOMD 转换（`waymo_to_scenario`）：
- 对象：91 步的位置、朝向、速度和有效标志（无效的步填 -1e4），尺寸，终点，类型，id；
- 道路：多段线或多边形，带 WOMD 的路网类型编号；
- 元数据：SDC、objects of interest、tracks to predict；
- 专家标记：车辆一开始就和别的车或路沿重叠，或者 logged 路径穿过路沿时，GPUDrive 就照日志回放它，不交给策略控制。

单元测试有 3 个；在 GPUDrive 的环境里跑全部通过，没有 trimesh 的环境跳过专家标记那一项。

**全部 500 个场景已转换**（`logs/gpudrive/cat_scenes`，1.1 GB）。按 GPUDrive 的专家标记，40% 的对象（每个场景中位数是 41 个里有 12 个）照日志回放、不交给策略控制，大多是一开始就压着路沿的车，比如停着的车。

**和 GPUDrive 自己的数据的两个差别**（按场景记在 `<目录>_index.json` 里）：

| | CAT 的 500 个场景 | GPUDrive 的做法 | 这里的做法 |
|---|---|---|---|
| 带信号灯 | 265 个 | 整个场景丢掉，模拟器里没有信号灯 | 保留场景，去掉信号状态。后果：logged 的车照样按信号灯停车，受控的车却看不到信号灯 |
| 有立体路面（立交桥等） | 18 个 | 丢掉 | 保留并标记 |

## 速度（测量时有 5 个 RL 训练和 SMART 在跑：显存已占 13.6 GB，GPU 利用率约 85%，CPU 负载约 8.3/8）

| 数据 | 世界数 | 受控的车 | 世界步/s | 车辆步/s | 显存 |
|---|---|---|---|---|---|
| GPUDrive_mini | 1 | 1 | 57 | 57 | 5.8 GB |
| GPUDrive_mini | 16 | 76 | 154 | 730 | 6.4 GB |
| GPUDrive_mini | 64 | 510 | 466 | 3,713 | 8.2 GB |
| CAT 场景 | 1 | 10 | 26 | 257 | 5.8 GB |
| CAT 场景 | 16 | 147 | 159 | 1,456 | 6.4 GB |
| CAT 场景 | 64 | 459 | 545 | 3,907 | 8.2 GB |
| CAT 场景，只推进仿真 | 64 | 459 | 587 | 4,210 | 8.2 GB |

- 128 和 256 个世界时显存不够：GPUDrive 约有 5.8 GB 的基础占用，每个世界再加约 38 MB，而当时只剩约 9 GB。整块 L4 空出来的话，估计能跑约 400 个世界。
- 只推进仿真（不读观测、奖励和完成标志）几乎一样快，所以瓶颈在仿真本身。而仿真此时在和 SMART 抢 GPU。
- 对比：一个 CAT 的 MetaDrive 训练约 35 步/s，占一个 CPU 核；5 个加起来约 175 步/s。

### SMART 跑完后重测（2026-10-05 04:50，`logs/gpudrive/benchmark_free.jsonl`）

GPU 上没有 SMART 了，但 5 个 RL 训练还在跑：重测开始时它们占 5.8 GB 显存，CPU 也被占满。

| 数据 | 世界数 | 受控的车 | 世界步/s | 车辆步/s | 显存 |
|---|---|---|---|---|---|
| GPUDrive_mini | 64 | 510 | 828 | 6,594 | 8.2 GB |
| CAT 场景 | 64 | 615 | 695 | 6,677 | 8.2 GB |
| GPUDrive_mini | 128 | 947 | 1,282 | 9,484 | 10.7 GB |
| CAT 场景 | 128 | 1,239 | 1,167 | 11,297 | 10.7 GB |
| GPUDrive_mini | 256 | 1,865 | 2,072 | 15,093 | 15.5 GB |
| CAT 场景 | 256 | 2,262 | 2,116 | 18,694 | 15.5 GB |

- 512 个世界在两种数据上都显存不够（Madrona 报 CUDA_ERROR_OUT_OF_MEMORY）。显存大约是 5.8 GB 基础占用加上每个世界约 38 MB；L4 有 22 GB，再减去 RL 训练占的部分，256 个世界就是上限。
- 世界数翻倍，速度涨 1.6–1.8 倍，还没饱和。所以显存更大的卡（如 H200）能跑得更快。
- 受控的车数（"受控的车"一列）和 SMART 运行期间那次测量不同，因为 GPUDrive 每次随机抽场景。

## 对迁移的判断

光换模拟器不够，还要重做这些：
1. CAT 的对手注入，以及每个回合一次的 DenseTNT 生成。几百个并行世界时，生成会成为新的瓶颈，需要批量生成或者预先生成。
2. TD3 的观测和奖励：换到 GPUDrive 之后，`cat` 这一组就不再是 CAT 原版，结果不能和 CAT 论文直接对比。
3. 训练中的碰撞归因：碰撞的数量会多出几个数量级，必须在 GPU 上批量算。
4. 信号灯：GPUDrive 不模拟信号灯，CAT 的场景有一半带信号灯。

建议：这篇论文继续用 MetaDrive。GPUDrive 适合后续做多智能体自博弈、加种子、扩大场景规模，前提是 GPU 空闲时的重测确认它能快一个数量级以上。
