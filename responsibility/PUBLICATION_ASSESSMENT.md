# 发表价值评估：用反事实责任改进对抗训练

2026-10-04（同日更新：加入 500 个场景的结果）· 基于 `counterfactual-responsibility` 分支和五篇参考论文（CAT、Hsu et al.、UniTraj、CtRL-Sim、CAT-K）

## 结论

最值得写的是**一篇**论文，用同一个反事实责任指标做三件事：生成对手、给训练中的碰撞定责、评测训练出的策略。三个方向单独投都偏薄，合在一起才完整。

- **方向 1（公平对手）单独投，新意不够。** 2026 年已有好几篇工作做"危险但可解"的对抗场景（第 2 节）。
- **卖点应该放在两处：**
  - 按责任加权碰撞惩罚（方向 2）：查新没有找到先例；
  - 反事实、数据驱动的责任定义：对手的责任 β 以米为单位，用 logged 人类驾驶标定；自车可避免性来自预测模型的反事实样本。
- **500 个场景的离线基准已经完成**（第 6 节），结果支持论文 B 的论点：CAT 的对手 40% 是不可避免的碰撞，平均 β 是 logged 对手 q90 的 20 多倍。
- **最大的风险：** 方向 1 和方向 2 的核心假设要靠 RL 训练回答，而训练还没开始。

## 1. 现状

| 部分 | 状态 |
|---|---|
| 指标移植（Hsu et al. 的 β_s、β_c，用 CAT 的 DenseTNT） | 完成并验证：`verify_densetnt` 在 3 个真实场景上复现 CAT 的 32 条对手轨迹，误差 3.6e-12 m，全部检查通过 |
| 方向 1：公平对手 `fair`（对手 β ≤ τ，且自车可避免性 ≥ ρ） | 代码完成；500 个场景的离线基准完成（含 τ = ∞ 的消融） |
| 方向 2：按责任份额加权碰撞惩罚（`--blame_weighting share`） | 代码完成，未训练 |
| 方向 3：双向评测（激进/胆怯）和碰撞归因 | 代码完成；500 个 logged 场景的责任、阈值和 HMM 等级已算完，作为评测策略的参照；策略本身未评测 |
| RSS 基线（归因，加训练惩罚 `--blame_weighting rss`） | 2026-10-04 完成 |
| 第二个预测模型 MTR（UniTraj） | 适配器完成；模型待训练 |

单元测试共 91 个：90 个通过，1 个需要 UniTraj 环境而跳过。

## 2. 查新：方向 1 已经拥挤

| 工作 | 做法 | 与本项目的区别 |
|---|---|---|
| [CARS](https://arxiv.org/abs/2605.13751)（2026-05） | 生成"可归责"的碰撞场景；责任按法规里的谨慎驾驶员模型判定 | 只用于测试，不用于训练；责任是规则化的 |
| [AlignADV](https://arxiv.org/abs/2606.14032)（2026-06） | 用 DPO 把生成器对齐到"危险但可解"的场景，加课程学习；数据也是 WOMD | 动机与方向 1 几乎相同；"可解"是学出来的偏好；摘要中没有评测胆怯 |
| [KG-ASG](https://arxiv.org/abs/2605.18895)（2026-05） | 给参与碰撞的车分配"主要/辅助"角色 | 分配的是角色，不判责任；只用于生成 |
| [SEAL](https://arxiv.org/abs/2409.10320)（2024）、STRIVE（2022） | 生成更像人的对手；STRIVE 用规划器检查场景是否可解 | 必须在相关工作里区分清楚 |
| [CtRL-Sim](https://arxiv.org/abs/2403.19918)（2024） | 回报条件的离线 RL；通过"指数倾斜"期望回报，生成反应式、可控的行为，包括对抗行为 | 控制的是回报，不直接约束对手的责任或自车能否躲开 |

"概率最大化的生成器会产出无解场景，用它训练会损害完成率"这个论点，已有工作提出过。

方向 2 没有找到先例：相关工作都是用 RSS 做奖励塑形或动作屏蔽，没有"先归因，再按归因结果加权碰撞惩罚"的做法。

方向 3 里，现有工作评测策略基本只看碰撞率和完成率，没有把"胆怯"量化成指标。

注：2026 年的三篇只读了摘要。

## 3. 与五篇参考论文的关系

- **CAT**：框架和主基线。CAT 选对手只看碰撞概率，不看这个碰撞该由谁负责。500 个场景上：
  - CAT 选中的对手平均 β = +5.00 m，而 logged 对手的 q90 只有 +0.23 m；
  - 40% 的场景是自车无论怎么开都躲不开的碰撞，logged 对手里只有 2%。
- **Hsu et al.（IROS 2023）**：指标的来源。原文用责任来解释轨迹预测，本项目把它用于对抗生成、奖励设计和策略评测。这个用法是新的，也是论文的方法论核心。
- **UniTraj / MTR**：第二个预测模型。用它回答审稿人一定会问的问题：换一个预测模型，结论是否不变。
- **CAT-K（SMART）**：本项目的责任代码就是从 `catk` 仓库移植的。SMART 可以作为第三个预测模型；它也是闭环、反应式交通的候选。
- **CtRL-Sim**：最接近的"可控对手"路线。审稿人可能会问"为什么不用回报倾斜来生成合理的对手"。区别在于：倾斜改变的是对手追求的回报，不保证对手的行为是人类会负责任地开出来的，也不保证自车躲得开。它可以作为对比基线，或者写进未来工作。
- CtRL-Sim 还指出，回放日志的交通不会对自车做出反应。这也是本项目的局限：CAT 的背景交通和自车的反事实样本都不会反应。

## 4. 候选论文

**A. 主论文：用反事实责任做对抗训练（2×2 训练 + 交叉评测）**
- 论点："CAT 的攻击成功有相当一部分来自不可避免、错在对手的碰撞。去掉这部分（fair），并且只按自车的责任份额惩罚碰撞（share），策略更少胆怯，安全性不下降。"
- 实验：
  - 训练设置：对手 `cat`/`fair` × 惩罚 `none`/`share`，另加 RSS 惩罚基线；每组 3 个种子；
  - 测试对手：无、`cat`、`fair`。
- 合适的投稿方向：CoRL、ICRA 或 RA-L。
- 风险：
  - RL 的方差可能盖过效果；
  - `fair` 降低攻击成功率，对抗信号可能不够。

**B. 低风险的分析论文："对抗场景生成器有多不公平？"**
- 只需要离线基准（第 6 步），用不到仿真器和 RL。
- 500 个场景证实了 20 个场景上的发现：像 logged 司机一样负责的对手（τ ≈ q90 = 0.23 m），对 logged 自车的碰撞率只有 8–16%，而 CAT 是 95%。也就是说，CAT 的攻击成功率主要来自不现实的对手行为。
- 加上 DenseTNT 与 MTR 的对照，可以说明结论不依赖预测模型。
- 合适的投稿方向：IV、ITSC 或 workshop。可以先投，也可以作为 A 的第一部分。

**C. 碰撞归因的方法验证**
- 比较反事实归因、追尾规则和 RSS，最好再加少量人工标注。
- 单独成文偏弱，适合作为 A 中支撑方向 2 的一节。

## 5. 审稿人会问的问题

| 问题 | 状态 |
|---|---|
| 和 RSS 比怎么样？ | **已实现**（`responsibility/rss.py`）。归因结果写进 `crashes.csv`，`compare_policies` 报告一致率、覆盖率和 RSS 判定的自车责任占比；训练基线是 `--blame_weighting rss`。RSS 只覆盖同向碰撞，对向和交叉碰撞需要路权规则，给 `n/a`。单元测试里有一个急刹的例子：RSS 判前车负责，追尾规则判后车。真实碰撞上的一致率待第 10 步 |
| 对手责任 β 在可避免性之外还有用吗？ | 这是和 AlignADV 这类工作区分的关键。**离线消融已完成**：只约束可避免性（τ = ∞）就能去掉几乎所有不可避免的碰撞（0.4%），碰撞率还有 49–78%，但对手的 β 仍是 +3.4 到 +4.4 m。离线结果只能说明两个约束管的是不同的东西；β 对训练有没有用，要靠 RL 回答 |
| 和近期的"可解场景"方法比呢？ | 待做。至少在正文里讨论 AlignADV、CARS、SEAL 和 CtRL-Sim；有开源代码的话做成基线 |
| 结论依赖预测模型吗？ | MTR 适配器已完成，模型待训练（runbook 第 11 步）；SMART（CAT-K）可以作为第三个模型 |
| 可避免性是开环的 | 写进局限：自车的反事实样本不会对对手做出反应，所以低估了能躲开的比例 |
| RL 策略的状态会偏离预测模型的训练分布 | 统计每个窗口自车处于分布内的程度；写进局限 |
| 测试集只有 100 个场景、3 个种子 | 报告均值 ± 标准差，并做显著性检验 |

## 6. 实验顺序和进度

在当前机器上运行：一块 L4（23 GB），8 个 CPU 核，与另一个训练任务共用。状态更新于 2026-10-04 12:30。

| 步骤 | 内容 | 状态 |
|---|---|---|
| runbook 2 | 吞吐量探测 | 完成：16 个场景 × 2 个 agent 用时 113 s，每个场景约 17 s |
| runbook 3 | 500 个场景上 SDC 和对手的责任，含 records、阈值判定和 HMM 等级 | 完成，约 2 h：阈值是 safety > 0.64 m、courtesy > 0.61 nats；BIC 选出 7 个等级 |
| runbook 6 | 对抗生成的离线基准：τ ∈ {0.25, 0.5, 1, 2, ∞}，ρ ∈ {0.1, 0.3, 0.5} | 完成：`cat` 规则在 500/500 个场景上与 CAT 原版一致；结果见下 |
| runbook 7 | 安装 MetaDrive | 完成：三处环境问题已修复并写进 runbook |
| runbook 8 | RL 训练：7 组设置 × 3 个种子 × 1e6 步 | 设置已定（第 5 节、runbook 8b），正式训练放到更大的服务器上 |
| runbook 9a | 在这台机器上试点：2×2 加 RSS 惩罚，1 个种子，2e5 步 | **中断**：跑到约 3.5 万步时机器出了问题，11:29 被关机，没有保存模型。截断前已确认两种定责在训练里都能跑通（每个训练 230–260 次碰撞，没有失败）。需要重跑 |
| runbook 10a | 回放的参照 rollout（4 种测试对手） | 抽样验收通过，并修复了 CAT 回放对手的错位；完整的 100 个场景待采集 |
| runbook 10–11 | 策略的 rollout 评测和交叉评测；MTR 训练和模型对照 | 待做 |

**第一个决策点（第 6 步之后）的结果：** 有限的 τ 都达不到 30% 的攻击率，`fair` 最高是 25%（τ = 2 m，ρ = 0.1）；只约束可避免性能达到 49–78%，但对手不负责任。所以：
- B 的论据已经足够，可以先写。
- A 仍然值得做，但要改一下实验设计：同时训练 `fair@2,0.1`（负责任，攻击率 25%）和 `fair@inf,0.5`（只保证可解，攻击率 49%），两者都和 `cat` 比较。这样 RL 结果能直接回答"β 在可避免性之外有没有用"。

## 7. 主要风险

- 现实的 τ 和足够的攻击率不能兼得，500 个场景已经证实：有限的 τ 都达不到 30%。负责任的对手提供的对抗信号可能太弱，训练可以考虑 τ 的课程（先松后紧）。
- 方向 2 的效果可能被 RL 的噪声淹没。
- RSS 的横向参数（ad-rss-lib 默认值，0.2 / 0.8 m/s²）很严格，可能把很多碰撞判成双方都有责任。要在真实碰撞上检查，并报告参数的敏感性。

## 参考文献

- Zhang et al., [CAT: Closed-loop Adversarial Training for Safe End-to-End Driving](https://arxiv.org/abs/2310.12432), CoRL 2023
- Hsu et al., [Interpretable Trajectory Prediction for Autonomous Vehicles via Counterfactual Responsibility](https://kaichiehhsu.github.io/research/responsibility), IROS 2023
- Feng et al., [UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](https://arxiv.org/abs/2403.15098), ECCV 2024
- Rowe et al., [CtRL-Sim: Reactive and Controllable Driving Agents with Offline Reinforcement Learning](https://arxiv.org/abs/2403.19918), CoRL 2024
- Zhang et al., [Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models (CAT-K)](https://arxiv.org/abs/2412.05334), CVPR 2025
- Shalev-Shwartz et al., [On a Formal Model of Safe and Scalable Self-driving Cars (RSS)](https://arxiv.org/abs/1708.06374), 2017
- Xiao et al., [Learning Responsibility-Attributed Adversarial Scenarios for Testing Autonomous Vehicles (CARS)](https://arxiv.org/abs/2605.13751), 2026
- Mei et al., [From Attacks to Curricula: Learnability-Guided Adversarial Training (AlignADV)](https://arxiv.org/abs/2606.14032), 2026
- Wang et al., [KG-ASG: Collision-Knowledge-Guided Closed-Loop Adversarial Scenario Generation](https://arxiv.org/abs/2605.18895), 2026
- Stoler et al., [SEAL: Skill-Enabled Adversary Learning](https://arxiv.org/abs/2409.10320), 2024
