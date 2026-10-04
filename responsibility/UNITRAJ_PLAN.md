# 接入 UniTraj 的 MTR 作为第二个预测模型：实施计划

> **暂停（2026-10-04）：** 第二个预测模型改用 CAT-K 的 SMART，见 `SMART_PLAN.md`。SMART 已经有训练好的检查点和完整的责任实现，不用训练 MTR，也不用 TB 级的磁盘。下文的适配器代码（`responsibility/unitraj.py`）保留，以后如果需要第三个模型可以再用。

**目标：**
- 把 UniTraj 里的 MTR 接进责任计算框架，作为 DenseTNT 之外的第二个预测模型。
- 回答一个审稿人一定会问的问题：换一个预测模型，责任值、激进/胆怯判定和策略对比的结论是否不变？
- 如果 MTR 更可靠，就把它作为主模型。

**范围：**
- 只替换"计算责任"用的模型：`compute_responsibility`、记录、方向 3 的评测、方向 2 的碰撞归因。
- CAT 的对抗生成（方向 1）继续用 DenseTNT。它是 CAT 原版的一部分，换掉就不再是同条件的对比。

标 🖥 的里程碑需要 GPU（MTR 的 CUDA 算子），只能在服务器上开发和验证；其余的可以离线开发。

## 状态

| 里程碑 | 状态 |
|---|---|
| U0 🖥 环境 | 工具已就绪：`setup_unitraj_env.sh`，以及 GPU 冒烟测试 `verify_unitraj --checkpoint random`。待服务器执行（runbook 11a） |
| U1 数据接口 | **完成**。在真实场景上，第 10、40、70 步的历史和未来与日志对齐，误差 < 3e-6 m |
| U2 🖥 训练 | 工具已就绪：`train_unitraj.py`（只导入 MTR，不需要 natten、torch_geometric、torch_cluster）、`MTR_womd.yaml`、试点流程。待服务器执行（runbook 11b–11d） |
| U3 🖥 校准和一致性检查 | 脚本完成，逻辑用替身模型测试过。待有检查点后运行（runbook 11e） |
| U4 接入 | **完成**：`--model densetnt\|mtr`、`--motion-set sampled\|weighted`、记录和可视化 |
| U5 🖥 稳健性研究 | 对照脚本 `compare_models.py` 完成。待有 MTR 的运行结果（runbook 11f–11g） |

实现与下文设计的几处不同：
- **不修补 UniTraj 的代码。** 由适配器把场景裁剪到窗口 [k − 10, k + 80]，片段外的步标为无效，并令 `starting_frame = 0`。这样 UniTraj 往前补零的问题不会触发。
- **时间戳重新从 0 开始。** 时间戳是模型输入的一部分，训练时总是 0–9 s。
- **训练用自己的启动脚本。** UniTraj 的 `train.py` 会导入所有模型及其依赖。启动脚本的训练器设置与之相同，唯一差别是每个 epoch 重新打乱训练数据。
- **CAT 的场景 id 列表**写在 `unitraj_configs/cat_scenario_ids.txt`：500 个文件，497 个不同的 id（有 3 个场景重复）。校准时排除这些场景。

---

## 设计决定

### 1. 代码位置和环境
- 适配器写在 `cat/responsibility/unitraj.py`。UniTraj 以包的形式安装（`python setup.py develop`），代码本身不改；需要修补的地方在适配器里覆盖。
- UniTraj 用**单独的 venv**（`unitraj39`）。原因：
  - UniTraj 的数据代码会导入 `scenarionet` 和较新的 `metadrive`，而 CAT 仓库根目录自带一份旧版 `metadrive/`。脚本把仓库根目录插在 `sys.path` 最前面，旧版会覆盖新版。
  - 推理其实不需要 `scenarionet`：只用到 `MetaDriveType` 这几个常量，`read_scenario` 用不到。所以适配器在导入 UniTraj 之前，如果真实模块不可用，就往 `sys.modules` 注入一个最小替身。
  - 这样推理环境里不装 `scenarionet` 也能跑；训练环境照常安装。
- `responsibility/` 里的指标、场景和 rollout 代码只依赖 numpy 和 torch，两个环境都能导入。
- `compute_responsibility.py` 目前在文件顶部导入 `DenseTNT`，会连带导入 TensorFlow。改成按 `--model` 延迟导入。

### 2. 模型配置：与 Waymo 标准和 DenseTNT 对齐
- 历史 **1.1 s（`past_len: 11`）**，未来 **8 s（`future_len: 80`）**，意图点文件用 `cluster_64_center_dict_8s.pkl`。
- 选 1.1 s 而不是 UniTraj 默认的 2.1 s，是为了：
  - 符合 Waymo 挑战赛的标准设定，MTR 原论文也是这个设定；
  - 第一个可用的上下文时刻与 DenseTNT 一样是第 10 步，两个模型的窗口可以逐一对照。
- 训练目标包含三类：车辆、行人、骑车人（`object_type: [VEHICLE, PEDESTRIAN, CYCLIST]`）。意图点文件按类型分别给出。这样 courtesy 可以扩展到对行人和骑车人（DenseTNT 只能算对车辆的）。

### 3. 预测分布：保留全部 64 个意图点
- 推理时设 `NUM_MOTION_MODES: 64`，跳过 NMS。
- MTR 训练时的分类损失，是在 64 个固定意图点上做交叉熵，目标是离真实终点最近的意图点（`MTR.py:666–693`）。所以 softmax 后的 64 个分数，就是固定意图点上的类别分布。
- 在框架里把它包成与 `GoalDistribution` 同形的对象：
  - `log_prob [64]`：64 个意图点的对数概率；
  - `goals`：各模态的预测终点（中心车坐标系），加上 `to_global`，供记录和可视化使用；
  - 每个模态的轨迹 `[64, 80, 2]`，以及逐步的方差（暂时不用）。

### 4. courtesy：精确 KL
- 邻车的 64 维分布在加入和移除自车时定义在同一组意图点上，直接用现有的 `goal_kl`，不需要蒙特卡洛估计。
- 分辨率比 DenseTNT 的 50176 个网格点粗：同一聚类里的细微变化感知不到。在论文里说明这一点。

### 5. safety 的备选轨迹集合，两种做法都实现
- **(a) 采样**，默认：按 64 维分布有放回地抽 N = 40 个意图点，取对应模态的均值轨迹。接口与 DenseTNT 相同，指标代码不用改。
- **(b) 精确加权**：直接用 64 个模态和它们的概率，计算加权 CVaR，没有采样噪声。需要在 `risk.py` 里加 `weighted_cvar`，并在 `metrics.responsibility_at` 里加一条分支：模型提供 `motion_set(dist)` 时走加权计算。
- 用 U5 的对照决定默认用哪种。

### 6. 构造模型输入
- `Scene.to_description()`：`from_description` 的逆过程，把 rollout 重建出来的场景也写回 ScenarioNet 格式。
- 调用 UniTraj 的 `preprocess → process`，不经过它的数据加载：
  - 在构造 `MTRDataset` 对象时跳过 `load_data`；
  - 用 `starting_frame = k − 10` 指定上下文时刻 k；
  - 用 `metadata.tracks_to_predict` 指定被预测的车。
- **必须修补的地方：** `preprocess` 在片段不够长时把补零加在序列前面（`base_dataset.py:183–184`），会让整个时间轴错位。91 步的 Waymo 片段里，k > 10 就会出现。改成补在后面，未来超出片段的部分标为无效。
- 输出在中心车坐标系下，用中心车在 k 时刻的位置和朝向转回全局坐标。

### 7. 反事实移除：用掩码，不删除
- 先构造"有该车"的输入。然后把该车所在槽位的 `obj_trajs_mask` 置 0、数据清零，得到"没有该车"的输入。其余 63 个槽位和顺序完全不变。
- 如果改成删除 track 再预处理：
  - UniTraj 只取最近的 64 辆车，删掉一辆会补进第 65 辆，两次输入的差别就不只是这一辆车；
  - 删除自车还会让 `sdc_track_index` 查找失败。
- U3 验证：车辆总数不超过 64 时，掩码与删除的输出一致（在数值误差内）。

### 8. 批处理
- 一个窗口需要的所有实例放在同一个 batch 里前向一次：被查询车 1 个，每个邻车加入/移除各 1 个。
- UniTraj 的输入是定长张量加掩码，训练时就是这样补齐的。U3 会验证单独前向和批量前向的结果一致（DenseTNT 在这一点上有问题，所以才必须逐个编码）。

---

## 里程碑

### U0 🖥 环境（半天）
- 在服务器上建 `unitraj39`：Python 3.9，torch 2.4.1 + cu121，与 CAT 环境相同。
- 编译 MTR 的 CUDA 算子（knn、attention），需要 nvcc 12.1。
- 用 UniTraj 自带的 nuScenes 样例数据跑 1 个 epoch 的 `train.py method=MTR`。
- 确认在 cat 仓库目录下导入 UniTraj 时，`metadrive` 被覆盖的问题是否出现，以及替身方案是否奏效。
- **验收：** 样例训练能跑通；适配器能在 cat 仓库里导入 UniTraj 的数据代码。

### U1 数据接口：`responsibility/unitraj.py` 的实例构造（2 天，离线）
- 实现 `Scene.to_description()`，并验证 `Scene.from_description(scene.to_description())` 与原场景逐字段一致。
- 实现 `instance(scene, step, target)`，用 UniTraj 的 `preprocess/process`，加上补零修补、`starting_frame` 设置和目标车指定。
- 实现 `mask_agent(batch, track_id)` 做反事实移除。
- 实现中心车坐标到全局坐标的变换。
- 本地 WSL 装一个只含数据代码的 CPU 环境（不需要 CUDA 算子）。
- **验收（单元测试）：**
  - 任意 k（10、40、70）的历史和未来与场景数组逐步对齐；
  - 目标车是中心车；
  - 坐标往返误差 < 1e-4 m；
  - 移除后其余槽位不变；
  - 重建自 rollout 的场景也能构造实例。

### U2 🖥 在 Waymo 上训练 MTR（数据转换半天，训练 3–6 天，与 U1、U3、U4 并行）
- 用 `scenarionet.convert_waymo` 把 WOMD 的 tfrecord 转成 ScenarioNet 格式。
- **防止数据泄漏：** 先核对 `raw_scenes_500` 的 scenario id 属于 WOMD 的哪个划分，确保这些场景不进训练集。否则在训练场景上评测，模型对 logged 行为过于自信，β 会被系统性地压低。
- 新配置 `unitraj/configs/method/MTR_womd.yaml`：`past_len: 11`、`future_len: 80`、8 s 意图点、三类目标。
- 训练成本高的话，可以减少 epoch，或者用 UniTraj 集成的 TAROT 做数据筛选。
- **验收：** 验证集上的 minADE、minFDE、mAP 与 MTR 论文在 WOMD 上的数字（约 0.60 / 1.22 / 0.41）相差不超过 10%。

### U3 🖥 模型接口和一致性检查：`scripts/responsibility/verify_unitraj.py`（2 天）
- `MTRModel` 实现框架需要的三个方法：`distribution`、`sample`、`with_and_without`；另外实现 `motion_set`，用于第 5 条的做法 (b)。
- **概率校准：** 在验证集上统计真实意图点分到的概率质量，以及期望校准误差（ECE）。校准不好就做温度缩放：在验证集上拟合一个温度 T，使真实意图点的负对数似然最小；T 存进检查点旁的 json。
- **检查项**（形式与 `verify_densetnt` 相同，在 `raw_scenes_500` 上跑）：
  - 重复前向结果完全一致（KL 为 0）；
  - 单独前向与批量前向一致；
  - 掩码移除与删除 track 一致（车辆 ≤ 64 时）；
  - 移除近处车辆引起的 KL 大于移除远处车辆；
  - 我们的 64 个模态做 NMS 后，与 UniTraj 原生的 6 模态输出一致；
  - 校准指标（温度缩放前后）。
- **验收：** 以上全部通过，并把校准结果写进报告。

### U4 接入流程（1 天，离线，用替身模型测试）
- `compute_responsibility.py`：
  - 加 `--model densetnt|mtr` 和 `--checkpoint`，写进 `config.json`，不同模型的结果不会混进同一个目录；
  - 适配器支持的话，加 `--motion-set sampled|weighted`。
- `risk.weighted_cvar`，以及 `metrics` 里的加权分支，加单元测试。
- 记录和可视化：目标点改为 64 个模态终点，外加概率。`visualize_responsibility` 按点画，不画网格热图。
- `blame.py`、`compare_policies` 不需要改，它们只经过 `model` 接口。
- `run_h200.sh` 加 `MODEL` 变量；runbook 加第 11 步。
- **验收：** 用 CPU 上的替身 UniTraj 模型，在 3 个场景上端到端跑通，包括 records、离线可视化和 `--rollouts`。

### U5 🖥 换模型的稳健性研究（1–2 天，服务器上跑）
- 在 500 个场景上，对 SDC 和对手车都用 MTR 计算责任，与 DenseTNT 的结果逐窗口对照：
  - β_s、β_c 的 Spearman 相关；
  - 激进/胆怯标记的一致率和 Cohen's κ；
  - 场景判定的一致率；
  - 两个模型各自的 HMM 等级结构是否相似（等级数、各等级的均值，以及各自在哪个维度上偏高）。
- 方向 3 的策略对比表用两个模型各算一遍，看结论（策略排序、相对参照的倍数）是否一致。这是论文里最关键的一句话。
- 方向 2 的碰撞归因：比较两个模型的判定一致率。
- 做法 (a) 和 (b) 对照：采样噪声有多大，选定默认做法。
- **产出：** 一张稳健性表和一段结论。如果两个模型的结论一致，论文可以写"结论不依赖预测模型"；如果不一致，要分析差异出在哪里（例如 64 个意图点的分辨率，或者 1.1 s 历史下的多模态程度）。

---

## 风险

| 风险 | 缓解 |
|---|---|
| 新旧 `metadrive` 冲突 | 单独的 venv；推理时用 `sys.modules` 替身，不装 `scenarionet` |
| MTR 的 CUDA 算子在 torch 2.4 / cu121 下编译失败 | 环境是独立的，可以换成 UniTraj 验证过的 torch 版本 |
| 训练时间过长 | 减少 epoch 或用 TAROT 筛数据；U1、U3、U4 与训练并行开发 |
| 64 维分数校准不好 | 在验证集上做温度缩放（U3） |
| 64 个意图点分辨率不足以体现 courtesy 的细微差别 | 在论文中说明；必要时用 128 或 256 个聚类重训 |
| CAT 场景出现在 MTR 训练集里 | U2 第一步核对 scenario id 并排除 |
| 1.1 s 历史下的预测不如 2.1 s | 这是 Waymo 的标准设定，与 DenseTNT 可比；需要的话再训一个 2.1 s 版本作对照 |

## 时间线（估计）

| 天 | 服务器 | 离线 |
|---|---|---|
| 1 | U0 环境；开始 U2 数据转换 | U1 |
| 2–6 | U2 训练 | U1 收尾，U4 |
| 5–7 | U3（用训练中间检查点先调试） | – |
| 7–9 | U3 定稿（最终检查点、校准），U5 | U5 分析 |

每个里程碑单独 commit，并补单元测试。
