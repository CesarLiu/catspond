# 用 CAT-K 的 SMART 作为第二个预测模型：实施计划

2026-10-04 定。替代 `UNITRAJ_PLAN.md` 里训练 MTR 的计划（U2–U5）。MTR 的适配器代码保留，计划暂停。

**目标：** 回答审稿人一定会问的问题：换一个预测模型，责任值、激进/胆怯判定和碰撞归因的结论是否不变。

**为什么改用 SMART：**
- catk 分支（`CesarLiu/catk4test`，本机在 `~/catk`）已经有训练好的 SMART 检查点和完整的责任实现。所以不用训练模型，也不用 TB 级的磁盘。
- 本仓库的 `geometry.py`、`risk.py`、`hmm.py` 就是从那里原样复制来的，两边的指标定义一致。

**本机上的 catk：**
- 检查点在 `~/catk/ckpts/`：`pre_bc_E31.ckpt`（行为克隆预训练）和 `clsft_E9.ckpt`（CAT-K 闭环微调）。
- 环境：Python 3.11、torch 2.4.1/cu121、PyG 2.6.1，用 catk 自带的 `install/setup_server.sh` 装在 `~/venvs/catk`。和本仓库的 Python 3.9 环境分开。

## 做法：复用 catk 的整条流程，只在数据上架桥

SMART 的反事实做法和 DenseTNT 很不一样：
- DenseTNT 对单个车预测；SMART 让所有车一起按 0.5 s 的 token 自回归地往前滚；
- 反事实是在滚动中把其他车强制成它们的 logged token，或者把某辆车从场景里去掉；
- courtesy 是按步累加的 token 分布 KL（`exact`）。

这些套不进本仓库面向单车预测的模型接口（`distribution`、`sample`、`with_and_without`）。所以不在本仓库重写，而是把 CAT 的场景转换成 catk 的数据格式，用 catk 原有的 `compute_responsibility` 计算，再把结果转回本仓库的格式，与 DenseTNT 对照。

| 里程碑 | 内容 | 在哪里跑 |
|---|---|---|
| S0 环境 | catk 环境；载入两个检查点；跑 catk 自己的测试。**完成（2026-10-04）**：catk 自带的安装脚本装好了环境（为了不动系统，跳过了用 apt 装 ffmpeg 的那一步）；catk 的 163 个测试全部通过；两个检查点都能按它们自己的解码器配置载入到 GPU 上，各 7.0M 参数 | `~/venvs/catk` |
| S1 导出 | `scripts/responsibility/export_catk.py`：把 `Scene`（logged 场景，或者从 rollout 重建的场景）转成 catk 的缓存格式。**完成（2026-10-04）**：`responsibility/catk_export.py` 重建 catk 解码器那一段的输出（有 3 个单元测试），第二段直接调用 catk 自己的函数 | catk 环境 |
| S2 核对 | **结构检查通过（2026-10-04，前 5 个场景）**，见下文。导出的场景能被 catk 读入和 token 化；logged 轨迹的 token 化重建误差小；logged 运动在 SMART 下的 next-token NLL 处于正常范围；有条件的话，再与 catk 用原始 WOMD 转出的同一个场景逐字段对照 | catk 环境 |
| S3 计算 | 在 500 个场景上用 SMART 计算 SDC 和对手的责任。**运行中（2026-10-04 19:28 起）**：`scripts/responsibility/run_smart.sh`，与 RL 正式训练同时跑。500 个场景已导出；每次模型调用 32 份场景副本（默认 128，在 61 辆车的场景里显存不够；32 时峰值 6.3 GB）；catk 估计约 4.7 小时 | catk 环境 |
| S4 对照 | 把 catk 的结果转成本仓库的 `windows.csv`，用现有的 `compare_models.py` 与 DenseTNT 对照；之后对回放参照和策略的 rollout 也做一遍 | 本仓库环境 |

### S1 导出的设计

catk 的预处理分两段：
1. `decode_*_from_proto` 把 WOMD 的 protobuf 读成数组和列表；
2. `get_agent_features`、`get_map_features`、`process_dynamic_map`、`preprocess_map` 把它们变成 SMART 用的张量。

导出只替换第 1 段：从 `Scene`（以及它带的 MetaDrive 地图和信号灯字典）构造同样的中间数组，第 2 段原样调用 catk 的函数。CAT 的场景就是从同一批 WOMD protobuf 转出来的，字段可以一一对应：
- 轨迹：位置、尺寸、朝向、速度、有效标志；
- 角色：`sdc_id`、`objects_of_interest`、`tracks_to_predict`；
- 地图类型：按 catk 的编号，车道 0/1/3（有停车标志的车道是 2），路沿 4/5，道路线 6/7/8，人行横道、减速带、车道入口 9；
- 信号灯：每一步的车道 id 和状态。

场景 pickle 是纯字典和列表，不依赖 MetaDrive，Python 3.11 能直接读。

### S2 的结果（catk 的 `verify_counterfactuals`，`clsft_E9`，场景 0–4）

- **全部通过：** 强制回放复现 token 化的日志（误差 0.00 m；相对原始日志只差 token 量化，平均 2.8 cm）；用掩码移除一辆车，等于直接删掉它；批量计算等于逐个计算；`hidden` 模式确实堵住了信息泄漏，`logged` 模式确实会泄漏；滚动结果可以重放；打包计算等于逐对计算。
- **场景 1 和 3 没通过一项"courtesy 能检测到邻车"：** KL 只有约 0.0002 nats。这项检查挑的是离得最近的一对车，在停车多的场景里就是两辆停着的车（速度都是 0.00 m/s）。停着的车的计划本来就不受旁边车的影响，所以 KL 接近 0 是正确的。这是检查挑车的方式造成的，不是导出的问题。
- **与 catk 原生缓存的逐字段对照（2026-10-04，`scripts/responsibility/verify_catk_export.py`）：** catk 的 `data/womd_cache/validation`（WOMD v1.2.1）里正好有 4 个 CAT 场景（21、84、225、275）。导出与原生缓存的差别全部来自源数据：CAT 的场景是 WOMD v1.1 的 interactive 划分，catk 用的是 v1.2.1。对照原始的 v1.2.1 protobuf 逐项确认过：
  - **agent：** 源数据一致的地方，导出完全一致：场景 21、84、225 的 id、有效标志、位置、朝向、速度、尺寸的差都是 0.000。场景 275 的两辆车轨迹完全相同，只是 v1.2.1 给它们重新编了号（868/918 对 3234/726）。
  - **地图：** v1.2 新增了车道入口（driveway），每个场景 290–480 个 token，CAT 的场景里一个都没有；v1.2.1 还重新处理了地图：场景 21 的地图 id，v1.2.1 有 276 个，CAT 的有 214 个，共有的只有 165 个，而且共有的车道点也不同。场景 84、225、275 的车道、路沿、道路线 token 大多在 0.5 m 以内对得上。
  - **角色：** interactive 划分只预测那两个交互的 agent，v1.2.1 的标准划分预测更多。
  - **信号灯：** 随着地图重处理，车道 id 和状态都变了；导出的信号灯状态与 CAT 的源文件一致。
- **对 S4 的含义：** DenseTNT 和 SMART 看到的是同一份 CAT 源数据，所以对照是公平的。但 SMART 是在带车道入口的 v1.2.1 地图上训练的，CAT 的地图里没有车道入口，对 SMART 是一处轻微的分布偏移，写进局限。
- **v1.2.1 原生缓存（2026-10-04）：** 为了量出这处偏移有多大，又按 CAT 的 497 个 scenario id 从 WOMD v1.2.1 的 validation_interactive（150 个分片）里取出原始场景：
  - `scripts/responsibility/extract_womd_scenarios.py` 不依赖 TensorFlow，在存数据的机器上运行，只拷贝这 497 条记录，得到一个 430 MB 的 tfrecord；
  - `scripts/responsibility/cache_womd_for_cat.py` 再用 catk 自己的预处理生成缓存，在 `logs/catk/v121`：**497 个全部找到，对应 CAT 的 500 个场景**（有 3 个场景 CAT 重复用了）。agent 数中位数是 21，范围 2–129；
  - v1.2.1 重新分配了场景所在的分片：服务器上原有的 14 个分片里只有 47 个 CAT 场景。
  - S3 在 v1.1 导出上的运行结束后，同样的设置在这份缓存上再跑一遍（`EXPORT=0 OUT=logs/catk/v121 bash scripts/responsibility/run_smart.sh`）。两次运行的差别就是数据版本（车道入口、重处理的地图、重新编号的 agent）对 SMART 结果的影响。
- **还没做的：** logged 运动在 SMART 下的 NLL，与 catk 在 WOMD 验证集上的水平对比。

### S3 的设置：与 DenseTNT 的运行对齐

| 设置 | 取值 | 原因 |
|---|---|---|
| `--query` | `interest` | CAT 场景的两个 objects of interest 就是 SDC 和对手，正是 DenseTNT 运行的两个 agent |
| `--neighbor-future` | `hidden` | DenseTNT 只看到 k 时刻之前的历史，对应的就是 `hidden` 这种开环读法。默认的 `logged` 会让被查询的车看到邻车真实的未来，抬高 β_s（catk 文档里有说明） |
| `--courtesy-estimator` | `exact` | 本仓库对 DenseTNT 用的是目标点上精确的 KL，`exact` 对应 SMART 的精确 KL；`boltzmann` 是论文用奖励函数做的近似 |
| 样本数、CVaR α、d_sat、度量时长、窗口间隔 | 40、0.1、10 m、2 s、0.5 s | 两边的默认值本来就相同 |
| 检查点 | `clsft_E9.ckpt` | CAT-K 的最终模型 |

**待定：** 是否也用 `--route-consistent`，让安全 CVaR 只保留沿 logged 路线的备选动作。DenseTNT 的运行没有这个过滤，先不用，以便对照。

## 验收

- S2：导出的场景能被 catk 读入、token 化和滚动；重复运行的结果完全一致；logged 运动的 NLL 和 catk 在 WOMD 验证集上的水平相当。
- S4：一张稳健性表，列出 β_s、β_c 的 Spearman 相关，激进和胆怯标记的一致率和 Cohen's κ，场景判定的一致率，以及碰撞归因的一致率，还有 SMART 是否复现论文 B 的核心结论（CAT 对手的 β 远高于 logged 对手）。

## 风险

| 风险 | 缓解 |
|---|---|
| MetaDrive 转换过的地图与 WOMD 原始数据有出入（点的间距、类型） | S2 的检查；有条件就拿原始 WOMD 场景逐字段对照 |
| catk 的模型是在 WOMD 训练集上训练的，CAT 的场景来自验证集或 testing_interactive，应该不重叠 | 用 CAT 场景的 scenario id 和训练集核对一次 |
| SMART 的计算量大：每个查询的 agent 要 40 次滚动，每个邻车的 courtesy 还要再加 | 先在 20 个场景上测速，再决定分片和并行 |
