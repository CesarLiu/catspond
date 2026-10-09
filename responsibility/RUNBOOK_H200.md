# H200 服务器实验命令清单

按顺序执行；所有命令都在仓库根目录（`cat/`）下运行。每一步写明了产出和怎么检查。
标 ⚠ 的步骤依赖 CAT 的 MetaDrive 仿真栈，这部分我在本地无法运行（没有 MetaDrive），第一次运行时请留意报错。

时间预估基于本地 CPU（每个场景、每个 agent，1 s 窗口约 1 分钟）；H200 上请以第 2 步的探测结果为准。

---

## 0. 准备代码和数据

分支 `counterfactual-responsibility` 目前只在本地。推到你自己的 remote，或者直接同步整个目录。下面的数据文件 git 不跟踪，需要单独拷贝：

```bash
# 在本机执行：USER@SERVER 和目标路径换成你的
rsync -av --exclude logs/ --exclude '__pycache__' cat/ USER@SERVER:~/cat/
```

服务器上确认这些文件都在：

```bash
cd ~/cat
git branch --show-current                  # counterfactual-responsibility
ls raw_scenes_500 | wc -l                  # 500
ls -la advgen/pretrained/densetnt.bin      # ~125 MB
ls advgen/utils_cython*.so                 # cpython-39 扩展
ls metadrive/__init__.py                   # 只有第 7-8 步（MetaDrive 实验）需要
```

后面的长任务都建议放在 tmux 里：

```bash
tmux new -s resp
```

## 1. 环境（Python 3.9 + torch 2.4.1/cu121）

```bash
bash scripts/responsibility/setup_env.sh    # 创建 ~/venvs/cat39；想用 conda 就加 BACKEND=conda
source ~/venvs/cat39/bin/activate
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1   # 屏蔽 TF 日志；不改写 CAT 跟踪的 .pyc
```

检查（必须全部通过才能继续）：

```bash
nvidia-smi
python -m pytest tests/responsibility -q                                 # 38 passed
python -m scripts.responsibility.verify_densetnt --n 5 --device cuda     # 最后一行应为 ALL CHECKS PASSED
```

`verify_densetnt` 在 GPU 上逐项对照 CAT 的原始流程：特征、32 条轨迹、采样、移除 agent、partner。

## 2. 吞吐量探测（约 10–20 分钟）

先用 16 个场景试跑，决定分片数 `SHARDS`：

```bash
OUT=logs/probe N=16 SHARDS=8 bash scripts/responsibility/run_h200.sh
grep -h "windows," logs/probe/sdc/logs/shard_*.log | head      # 每个场景的耗时（s）
nvidia-smi                                                       # 跑的时候另开窗口看 GPU 利用率
```

- 大多数时间花在 CPU（构造 DenseTNT 的输入），所以一块 GPU 可以同时跑很多进程。
- GPU 利用率低、CPU 核也没占满：增大 `SHARDS`。一般取 `SHARDS ×2`（sdc 和 adv 两个 agent）≈ CPU 核数的一半。
- 预估：总时间 ≈ 每场景耗时 × 1000（500 场景 × 2 个 agent）÷ 并行进程数。

## 3. 主实验：500 个场景上 SDC 和对手车的责任

```bash
RECORDS=1 SHARDS=32 bash scripts/responsibility/run_h200.sh 2>&1 | tee logs/run_h200.log
```

- 中途断了，重跑同一条命令即可续跑（已完成的场景会跳过）。
- 进度：`ls logs/responsibility/sdc/obs | wc -l`，数到 500 就是完成。
- 默认设置：窗口间隔 0.5 s（`STRIDE=5`），每窗口 40 个样本，度量时长 2 s，D_g 在 10 m 饱和。

产出（全部在 `logs/responsibility/` 下）：

| 路径 | 内容 |
|---|---|
| `sdc/`、`adv/` 下的 `windows*.csv` | 每个场景、每个 t_k 的 β_s、β_c，以及它们分别来自哪辆邻车 |
| `sdc/obs/`、`adv/obs/` | 每个场景每辆邻车的详细数据 |
| `sdc/records/`、`adv/records/` | 离线可视化用的记录（约 0.5–1.5 MB/场景） |
| `sdc/summary/`、`adv/summary/` | 阈值判定。对手车用 SDC 分布标定的阈值来判定 |
| `levels/`、`levels_log/` | HMM 责任等级。`levels_log` 的 courtesy 取 log(1+β_c) |

需要看的结果：
- `run_h200.log` 末尾的阈值（safety、courtesy）和被标为激进的场景数；
- 等级表：每个等级 "elevated in" 哪一维，sdc 和 adv 各自落在每个等级的窗口比例；
- 图：`sdc/summary/responsibility.png`（β_s–β_c 散点）、`levels/levels.png`（按等级着色的散点 + BIC 曲线）。

## 4. 离线可视化最激进的场景

不需要 GPU 或 TensorFlow，直接读第 3 步的 records：

```bash
# 按阈值判定最激进的 10 个场景（scenes.csv 已按被标记的窗口数排序）
for s in $(tail -n +2 logs/responsibility/sdc/summary/scenes.csv | head -10 | cut -d, -f1); do
  python -m scripts.responsibility.visualize_responsibility \
    --record logs/responsibility/sdc/records/$s.pkl \
    --levels logs/responsibility/levels/levels.csv \
    --out-dir logs/videos/sdc_$s
done

# 按等级判定：激进窗口占比最高的 10 个场景
python - <<'PY'
import csv
rows = [r for r in csv.DictReader(open("logs/responsibility/levels/scenes_levels.csv")) if r["run"] == "sdc"]
rows.sort(key=lambda r: -float(r["aggressive_share"]))
print(" ".join(r["scene"] for r in rows[:10]))
PY
```

每个场景输出 `frames/*.png`、`responsibility.gif` 和 `responsibility.mp4`。对手车的视频把路径换成 `adv/records/`、输出目录前缀换成 `adv_`。

## 4b. CVaR α 对照（论文的两种读法，必跑）

对 safety responsibility 的 CVaR，Hsu 论文的正文和公式给出相反的尾部方向：
- 正文说 "α → 1 时为最大值"。本实现按这个约定，默认 α = 0.1，等于取安全余量减少量上尾 90% 的均值，接近平均值。
- 复现指南（按公式）把 α = 0.1 读作 "40 个样本中最大 4 个的均值"（上尾 10%）。在本实现里这对应 `--cvar-alpha 0.9`。

两者的数值会明显不同，所以要用同样的设置再跑一遍后一种：

```bash
OUT=logs/alpha09 EXTRA="--cvar-alpha 0.9" RECORDS=1 SHARDS=32 bash scripts/responsibility/run_h200.sh 2>&1 | tee logs/run_alpha09.log
```

两组的 β_s 分布并排比较（courtesy 不受 α 影响，两组应该一致）：

```bash
python - <<'PY'
import numpy as np
from responsibility.results import read_windows
for agent in ("sdc", "adv"):
    for run in ("logs/responsibility", "logs/alpha09"):
        rows = [r for r in read_windows(f"{run}/{agent}") if r["speed"] >= 1.0]
        s = np.array([r["safety"] for r in rows]); c = np.array([r["courtesy"] for r in rows])
        q = np.quantile(s, [0.5, 0.75, 0.9, 0.99])
        print(f"{agent:>3} {run:>20}: {len(rows)} windows  safety p50/p75/p90/p99 "
              + " ".join(f"{v:+.2f}" for v in q) + f"  share > 0: {100 * (s > 0).mean():.0f}%"
              + f"  | courtesy p90 {np.quantile(c, 0.9):.3f}")
PY
```

然后分别对照论文 Fig. 2/3：大部分质量在 0 附近，safety 轴最多到约 1.2 m。
- 看哪种读法的分布更像论文；
- 看判定结果变化多大：比较两组 `summary/scenes.csv` 里被标为激进的场景，以及 `levels/` 的等级表。

```bash
# scenes.csv 是 \r\n 换行，先去掉 \r
flagged() { tail -n +2 "$1" | tr -d '\r' | awk -F, '$NF==1{print $1}' | sort; }
diff <(flagged logs/responsibility/sdc/summary/scenes.csv) <(flagged logs/alpha09/sdc/summary/scenes.csv)
```

之后选定一种读法，在报告里写明。第 6 步对抗选择的 β 固定用 α = 0.1（`ResponsibleAdvGenerator` 的默认值）；选定 0.9 的话，第 6 步的阈值要在同样的读法下重新标定，目前需要改 `responsibility/adversarial.py` 里的 `ResponsibilityConfig`。

## 5. （可选）敏感性分析

每组设置单独一个输出目录；`--horizon`、`--d-sat` 等参数通过 `EXTRA` 传入：

```bash
OUT=logs/sens_h30   EXTRA="--horizon 30"  SHARDS=32 bash scripts/responsibility/run_h200.sh   # 3 s 度量时长
OUT=logs/sens_dsat5 EXTRA="--d-sat 5"     SHARDS=32 bash scripts/responsibility/run_h200.sh   # 更近的交互范围
OUT=logs/sens_s80   SAMPLES=80            SHARDS=32 bash scripts/responsibility/run_h200.sh   # 样本数翻倍（看 CVaR 的稳定性）
```

比较各组的 `summary` 和等级表。样本数翻倍后数值变化很小，说明 40 个样本够用。

## 6. 对抗生成：离线对比与阈值标定

对比 CAT 原版、责任约束版和公平版（`fair`：对手 β ≤ τ，且自车可避免性 ≥ ρ，方向 1 的 M1.3）的选择规则。这一步不需要 MetaDrive，按 logged ego 开环评估，分 10 份并行跑：

```bash
mkdir -p logs/advgen
for i in $(seq 0 9); do
  python -m scripts.responsibility.benchmark_advgen --first $((i*50)) --n 50 \
    --thresholds 0.25 0.5 1 2 --avoid 0.1 0.3 0.5 --out logs/advgen/part_$i.csv --device cuda \
    > logs/advgen/part_$i.log 2>&1 &
done
wait
grep -h "reproduces" logs/advgen/part_*.log             # 每份都应是 50/50：cat 规则与 CAT 原版一致
python -m scripts.responsibility.benchmark_advgen --summarize logs/advgen/part_*.csv --plot logs/advgen/tradeoff.png
```

结果怎么读：
- 每条规则的列：预测碰撞率、被选中对手轨迹的 β（均值和中位数）、自车可避免性的均值、"不可避免的碰撞"的占比（撞上且可避免性 < 0.1）。
- 倒数第二行是 logged 对手自身 β 的分布。τ 在这里取，**推荐取 q90**。
- 最后一行是 logged 对手的自车可避免性分布。真实的对手几乎都能被躲开，ρ 取得比它的 q10 低一些即可。
- 选 (τ, ρ) 的标准：`fair@τ,ρ` 的"不可避免的碰撞"接近 0，同时预测碰撞率不低于约 30%。汇总的最后几行按这个标准列出推荐的两组，并直接给出 `TAU=… RHO=…`（门槛用 `--min-collision` 修改）。
- `tradeoff.png` 是权衡曲线（M1.3）：左图是碰撞率 vs 被选中对手的平均 β，右图是碰撞率 vs 不可避免碰撞占比；每条线是一条规则随 τ 的变化，点旁标着 τ。
- 500 个场景的结果（2026-10-04，L4 机器）：
  - 有限的 τ 都达不到 30% 的碰撞率，`fair` 最高是 25%（τ = 2，ρ = 0.1）；
  - logged 对手的 q90 是 0.23 m，那附近所有规则只有 8–16%；
  - 只约束可避免性（τ = ∞）有 49–78%，但对手的 β 和 CAT 的差不多。
  - 汇总推荐的 `fair@inf` 是消融，不是原设计的公平对手。
- 选定（2026-10-04）：两组都训练。`fair@2,0.1` 是负责任的公平对手；`fair@inf,0.5` 只保证可解，是 β 的消融：

```bash
TAU=2     # 有限 τ 里碰撞率最高的
RHO=0.1
FAIR="--adv_selection fair --resp_threshold $TAU --resp_avoid $RHO"   # 实验名 cat_fair2_0.1
ABL="--adv_selection fair --resp_threshold inf --resp_avoid 0.5"       # 实验名 cat_fairinf_0.5
```

## 7. ⚠ 安装 CAT 的 MetaDrive 仿真依赖（同一个 venv）

CAT 的 `requirements.txt` 固定了 `torch==1.12.0+cu116`，在 H200 上不能用，所以要排除它和 TF 相关的包：

```bash
grep -v -E "^(torch|torchvision|waymo-open-dataset|pickle5|opencv-python)" requirements.txt > /tmp/cat_requirements.txt
echo "numpy<1.24" >> /tmp/cat_requirements.txt   # 不让 pip 升级 numpy（TF 2.12 需要 < 1.24）
python -m pip install "pip<24.1"        # gym==0.22.0 的元数据在新版 pip 下安装会报错
python -m pip install -r /tmp/cat_requirements.txt
python -m pip install "setuptools<81"   # 见下方说明
# pickle5 是 Python < 3.8 的 backport（需要编译）；3.9 的 pickle 自带 protocol 5，CAT 只是按名字导入
echo "from pickle import *  # noqa" > "$(python -c 'import site; print(site.getsitepackages()[0])')/pickle5.py"
python -c "import cv2, metadrive, gym, panda3d; print('metadrive ok')"   # 仓库自带的 metadrive/ 从根目录直接导入
export SDL_VIDEODRIVER=dummy             # 无显示器时 pygame 的 top-down 渲染需要
```

在 L4 机器上实际安装时遇到的三个问题（2026-10-04），上面的命令已经避开：
- **OpenCV：** 不要装 `opencv-python`。它和 `setup_env.sh` 装的 `opencv-python-headless` 都提供 `cv2`，后装的会覆盖前者，而它需要系统的 `libGL.so.1`，服务器上通常没有。headless 版本对 CAT 已经够用。
- **setuptools < 81：** Panda3D 通过 `pkg_resources` 找到 `panda3d-gltf` 的加载插件，而 setuptools 81 起去掉了 `pkg_resources`。没有它的时候，MetaDrive 预加载行人模型会改用 Assimp，报错 `mismatched number of frames`。注意：用错误的加载器载入过一次之后，Panda3D 会把结果缓存到 `~/.cache/panda3d`，之后报的是 `get_anim_control` 的 `AssertionError`。这时要删掉这个缓存目录。
- **场景目录：** `raw_scenes_500/` 里只能放场景文件。MetaDrive 会把目录里的每个文件都当成场景读取，例如 `.gitignore` 会触发 `.gitignore is not .pkl file`。要让 git 忽略这个目录，在仓库根目录的 `.gitignore` 里加规则（现在已经加了）。

安装后确认 torch 仍然是 2.4.1+cu121：

```bash
python -c "import torch; print(torch.__version__)"
```

如果被改了，重跑 `setup_env.sh` 中安装 torch 的那一行。

## 8. ⚠ MetaDrive 闭环实验

### 8a. 对抗生成的攻击成功率（CAT 的 `cat_advgen.py`，500 个场景，EgoReplay）

```bash
python cat_advgen.py --adv_selection cat                                     2>&1 | tee logs/advgen/closed_cat.log
python cat_advgen.py --adv_selection constrained --resp_threshold $TAU       2>&1 | tee logs/advgen/closed_constrained.log
python cat_advgen.py --adv_selection penalized --resp_penalty 1.0            2>&1 | tee logs/advgen/closed_penalized.log
python cat_advgen.py --adv_selection fair --resp_threshold $TAU --resp_avoid $RHO 2>&1 | tee logs/advgen/closed_fair.log
```

看进度条里的 `avg_attack_success_rate` 和 `avg_compute_time`。责任约束版在最后还会打印选中对手轨迹的平均 β。

### 8b. RL 训练（TD3，1e6 步，每组 3 个种子）

七组设置（2026-10-04 定）：
- 对手：`cat`、`fair`、只约束可避免性，加上没有对手的 `replay`（方向 1，M1.4）；
- 碰撞惩罚：原版、按责任份额（方向 2）、按 RSS（规则基线）；
- `cat`、`cat_share`、`cat_fair2_0.1`、`cat_fair2_0.1_share` 组成 2×2（M2.3）。

| 名字 | 对手 | 碰撞惩罚 |
|---|---|---|
| `replay` | 没有对手 | 原版 |
| `cat` | CAT 原版 | 原版 |
| `cat_share` | CAT 原版 | 按责任份额（`--blame_weighting share`） |
| `cat_rss` | CAT 原版 | 按 RSS（`--blame_weighting rss`，规则基线） |
| `cat_fair2_0.1` | 公平（τ = 2，ρ = 0.1） | 原版 |
| `cat_fairinf_0.5` | 只约束可避免性（τ = ∞，ρ = 0.5；β 的消融） | 原版 |
| `cat_fair2_0.1_share` | 公平 | 按责任份额 |

先在小机器上跑过一次试点（每组 1 个种子，2e5 步），确认整条流程能跑通，见 9a。

用脚本启动 17 个训练：5 组核心设置（`replay`、`cat`、`cat_share`、`cat_rss`、`cat_fair2_0.1`）各 3 个种子，两个消融（`cat_fairinf_0.5`、`cat_fair2_0.1_share`）只跑种子 0（`ABLATION_SEEDS`，默认 `0`）。消融的种子 0 和 `cat` 差别明显时，再用 `ABLATION_SEEDS="0 1 2"` 补跑另外两个种子，已完成的会跳过。

```bash
DRY=1 bash scripts/responsibility/run_rl.sh                 # 先看会启动哪些训练、同时跑几个
nohup bash scripts/responsibility/run_rl.sh > logs/run_rl.log 2>&1 &
```

脚本做的事：
- 同时跑的训练数默认按可用内存（每个 4.5 GB）和核数算，可以用 `PARALLEL=…` 指定。多块 GPU 时轮流分配。
- 默认加 `--no_store_map`。内存很大的机器可以设 `NO_STORE_MAP=0`，地图缓存更快，但每个训练会涨到 5 GB 以上。
- 日志不缓冲，写到 `logs/rl/<名字>_s<种子>.log`。正常结束的训练会留下 `.done`，重跑脚本时跳过。
- 资源监控写到 `logs/rl/resources.log`：可用内存连续 2 分钟低于 2 GB 时，停掉最新启动的训练；磁盘低于 5 GB 时全部停掉。原来只要一次检查低于 2 GB 就动手，2026-10-06 另一个程序造成一次短暂的内存尖峰，因此停掉了跑到 87% 的 cat_share_s1。`WATCHDOG=0` 关掉监控，用于在已有 run_rl.sh 的机器上再启动一个（比如单独重跑某个训练）。
- 2026-10-04 在 L4 机器上用 3000 步试过 `cat_fairinf_0.5` 和 `replay`，都正常结束。

时间：试点时每个训练约 30 步/s（5 个并行），1e6 步约 13 小时，再加上评测（默认每 2.5 万步一次）。

- 进度：`tail -f logs/run_rl.log logs/rl/*.log`。
- 曲线数据：`logs/<名字>_MDWaymo-seed<seed>/logger.csv`。各组的名字见上表，分开记录，不会互相覆盖。
- 模型：`models/<名字>_s<seed>*`（加了 `--save_model` 才会保存；每个种子单独一份，第 10 步要用）。
- `_share` 的两组每次碰撞多两次 DenseTNT 推理（GPU 上约 1–2 s），日志里每次碰撞有一行 `collision at step ... penalty weight ...`。`cat_rss` 不需要 DenseTNT，每次归因约 0.15 s。
- 每次评测时打印 `N collisions attributed: mean penalty weight ...`。
- 归因明细写到 `logs/blame/<名字>_s<seed>.csv`，第 8d 步汇总。
- 先确认 `_share` 能正常跑：看最初几次碰撞的那一行，不应有 `warning: ... not attributed`。偶尔一两次无妨（该次碰撞保留全部惩罚），频繁出现就停下来告诉我。

### 8c. 画学习曲线

```bash
for key in route_completion_normal route_completion_adv crash_rate_normal crash_rate_adv; do
  PYTHONPATH=. python scripts/plot.py --log_dir logs/ --title "$key" --ykey $key --xlim 1000000 --save
done
```

比较各组在正常场景和对抗场景下的完成率与碰撞率。关键问题有三个：
- 用"合理的"对手训练出来的策略，泛化是否更好（`cat_fair2_0.1` 对 `cat`）；
- β 在可避免性之外有没有用（`cat_fair2_0.1` 对 `cat_fairinf_0.5`）；
- 只惩罚自车有责任的碰撞，能否减少过度保守（`cat_share` 对 `cat`，`cat_fair2_0.1_share` 对 `cat_fair2_0.1`），以及反事实的定责是否比 RSS 好（`cat_share` 对 `cat_rss`）。

### 8d. 训练中的碰撞归因（M2.2、M2.3）

```bash
python -m scripts.responsibility.summarize_blame --logs logs/blame/*.csv --out logs/blame/summary.md
```

每个 `_share` 和 `_rss` 训练一行（两种模式都会记录 RSS 的判定）：
- 碰撞次数、各判定的占比、平均惩罚权重 w；
- `full penalty`：保留全部惩罚（w = 1）的占比；
- `w < 0.5`：主要是对方责任的占比；
- 与追尾规则和 RSS 的一致率，以及 RSS 判自车、对方、双方的占比；
- 每次归因的耗时；
- w 在训练过程中的变化（按训练步数分 4 段）。

`cat_share` 的 `w < 0.5` 应明显高于 `cat_fair_share`，因为 CAT 的对手常常直接撞上来。w 随训练下降，说明剩下的碰撞越来越是自车的错。

## 9. 汇总清单

| 实验 | 命令所在步骤 | 关键产出 |
|---|---|---|
| 环境与一致性验证 | 1 | `ALL CHECKS PASSED` |
| SDC / 对手车责任（500 场景） | 3 | `summary/`、`levels/`、records |
| 激进场景视频 | 4 | `logs/videos/*/responsibility.mp4` |
| CVaR α 对照（两种读法） | 4b | `logs/alpha09`，与主实验并排的分位数 |
| 敏感性（可选） | 5 | `logs/sens_*` |
| 对抗选择离线对比与 τ 标定 | 6 | `--summarize` 表格、TAU |
| 闭环攻击成功率 ⚠ | 8a | `logs/advgen/closed_*.log` |
| RL 训练对比 ⚠ | 8b–8d | `logs/*_MDWaymo-seed*/logger.csv` 和曲线、`logs/blame/summary.md` |
| 策略的双向风格评测与交叉评测 ⚠ | 10 | `rollouts/`、`logs/responsibility/policies/compare/comparison.md`、`comparison_seeds.md` |
| UniTraj MTR 训练与模型对照 ⚠ | 11 | 检查点和校准文件、`logs/responsibility/model_comparison/*/model_comparison.md` |

### 9a. 试点（小机器上，正式训练之前）

在 L4 机器上跑通整条流程：2×2 加 RSS 惩罚，每组 1 个种子，2e5 步，每 5 万步评测一次：

```bash
mkdir -p logs/rl_pilot
export SDL_VIDEODRIVER=dummy
for setting in "--mode cat" "--mode cat --blame_weighting share" "--mode cat --blame_weighting rss" \
               "--mode cat $FAIR" "--mode cat $FAIR --blame_weighting share"; do
  name=$(echo "$setting" | tr -d ' -')
  nice -n 10 python cat_RLtrain.py $setting --seed 0 --max_timesteps 200000 --eval_freq 50000 --save_model \
    > logs/rl_pilot/$name.log 2>&1 &
done
```

**第一次试点（2026-10-04 10:13 开始）没有跑完：**
- 每个训练跑到约 3.5–4 万步时，机器出了问题：约 10:40 起系统日志写不进文件，snapd 卡死，11:29 虚拟机被正常关机。
- 五个训练都没有 Python 报错，但也都没到第一次评测（5 万步），所以没有保存模型。
- 原因无法确定。很可能是这台机器和另一个任务共用，磁盘或内存不够了。

截断前已经确认的：
- 速度：每个训练 17–19 步/s（5 个并行，和另一个任务共用 8 个核）。2e5 步的试点约需 3–4 小时，加上评测。
- 两种定责在训练里都能跑通，没有 `not attributed` 警告：
  - 前约 3.5 万步，每个训练有 230–260 次碰撞被归因；
  - 每次归因的耗时：`share` 0.4 s，`rss` 0.1 s；
  - `cat_share` 的平均惩罚权重是 0.72，24% 的碰撞主要是对方的责任；
  - RSS 对 56–65% 的碰撞给出判定，其余大多不是同向碰撞。在两者都给出判定的碰撞里，与反事实定责一致的比例是：`cat_share` 75%，`cat_fair2_0.1_share` 52%。
- 这段数据主要来自随机探索（前 1 万步是随机动作），只能说明流程能跑通，不能用来检验假设。

**还没验证的：** 评测、保存模型、用训练出的模型采集 rollout。

**内存是瓶颈（重跑时查明）：** CAT 的 `ScenarioEnv` 默认 `store_map=True`，会把建过的每张地图都留在内存里，没有上限。
- 3 个训练并行时，每个训练 8 分钟就占 5 GB，而且每分钟还涨约 0.33 GB。这很可能就是第一次试点时机器出问题的原因（5 个训练，加上另一个任务）。
- `cat_RLtrain.py --no_store_map` 改为每个回合重建地图，结果不变：8 分钟时每个训练只占 2.2 GB，之后每分钟涨约 50 MB。
- 代价是速度：5 个训练同时跑，每个约 35 步/s；用缓存、3 个训练时约 52 步/s。总吞吐量反而更高。
- 内存少于每个训练约 10 GB 的机器，都应该加上这个选项。

**第二次试点（2026-10-04 12:42–15:31）：全部跑通。**
- 5 个训练同时跑，都加了 `--no_store_map`，并用 `python -u`，跑满 2e5 步，都正常退出（exit 0），用时 2 小时 35–46 分钟。
- 每个训练在 5、10、15 万步各评测一次，模型已保存。
- 内存稳定在每个训练 3.3–4.2 GB，最低还有 12.8 GB 可用。资源监控一次都没出手。
- 同时采集了回放的参照 rollout（第 10a/10b 步）。

试点的评测（种子 0）。normal 是测试集上不放对手的完成率 / 碰撞率：

| 设置 | normal，50k → 100k → 150k |
|---|---|
| `cat` | 0.55/0.27 → 0.69/0.19 → 0.62/0.21 |
| `cat_share` | 0.55/0.24 → 0.56/0.29 → 0.60/0.29 |
| `cat_rss` | 0.41/0.34 → 0.50/0.34 → 0.57/0.31 |
| `cat_fair2_0.1` | 0.52/0.26 → 0.68/0.20 → 0.67/0.16 |
| `cat_fair2_0.1_share` | 0.55/0.29 → 0.61/0.29 → 0.66/0.18 |

- 只有一个种子，每次评测 100 个回合，标准误约 ±4–5 个百分点，所以这些差别还不能下结论。
- `eval_policy` 的对抗评测用的是各组自己的训练对手，不同设置之间不能比较，所以这里没有列出来。交叉评测要用第 10 步的 rollout。

训练中的归因（`summarize_blame`）：

| 训练 | 碰撞次数 | 平均惩罚权重 w | w < 0.5 | 与 RSS 的一致率 |
|---|---|---|---|---|
| `cat_share` | 1623 | 0.65 | 32% | 71% |
| `cat_fair2_0.1_share` | 1518 | 0.73 | 23% | 70% |
| `cat_rss` | 1526 | 0.89 | 11% | – |

- 每次归因 0.1–0.3 s，没有一次失败。
- 符合计划的预期：`cat_share` 的 `w < 0.5`（主要是对方的责任）高于 `cat_fair2_0.1_share`。
- RSS 只给出约一半碰撞的判定，所以只有 11% 的碰撞被去掉了惩罚。

**整条流程的检查：** 用训练好的 `cat` 模型（2e5 步）在 10 个测试场景上采集 rollout，计算责任，再做对比和碰撞归因，全部跑通。
- 这个策略还很弱：完成率 56.5%，常常开出道路；激进窗口是参照的 1.8 倍，胆怯窗口是 2.1 倍。
- 它和 CAT 对手的 4 次碰撞，归因判对手 3 次、双方 1 次。

**正式训练的估算：**
- 1e6 步、5 个训练并行时，每个约 13 小时（试点估算）。实际在这台 L4 机器上和 SMART 一起跑时，每个约 19 小时。
- 21 个训练要 5 批，约 4 天。2026-10-05 改成 17 个，4 批，约 3.2 天，省下约 19 小时（见 8b）。
- 核更多的服务器可以同时跑更多个，每个训练按 4.5 GB 内存估算。

## 10. ⚠ 策略的双向驾驶风格评测（方向 3，M3.0–M3.4）

依赖第 8b 步保存的模型（`models/<名字>_s<seed>*`）。`replay` 基线不需要模型，可以先跑，顺便完成第 10a 步的核查。

用脚本跑完 10a–10c，对 `models/` 里的每个模型和回放的 logged 驾驶都做一遍：

```bash
nohup bash scripts/responsibility/run_eval.sh > logs/run_eval.log 2>&1 &
cat logs/responsibility/policies/compare/comparison_seeds.md      # 跑完后：按种子平均的表和矩阵
```

- 每个策略在测试集上面对四种测试对手：无、CAT、`fair2_0.1`、`fairinf_0.5`。
- 计算责任和碰撞归因，在回放的无对手驾驶上拟合等级，最后输出对比表。
- 每一步都可以续跑。同时跑的进程数默认按可用内存（每个 3 GB）和核数算，可以用 `PARALLEL=…` 指定。
- 2026-10-04 在 L4 机器上用 2 个试点模型、3 个场景试过整个脚本。
- 下面的 10a–10c 说明它做了什么，以及结果怎么读。

### 10a. 采集 rollout，并核查 CAT 的评测流程（M3.0、M3.1）

```bash
mkdir -p logs/rollouts
export SDL_VIDEODRIVER=dummy
# 基线：logged ego 回放，正常场景 + CAT 对手（测试集 400–499）
python -m scripts.responsibility.collect_rollouts --policy replay --adversary --out_dir rollouts \
    2>&1 | tee logs/rollouts/replay.log
tail -4 logs/rollouts/replay.log
# 另外两种测试对手下的回放（交叉评测的参照）
python -m scripts.responsibility.collect_rollouts --policy replay --adversary $FAIR --out_dir rollouts \
    2>&1 | tee logs/rollouts/replay_fair.log
python -m scripts.responsibility.collect_rollouts --policy replay --adversary $ABL --out_dir rollouts \
    2>&1 | tee logs/rollouts/replay_abl.log
```

最后几行是核查结果：
- `replay: max ego error ... m`：应 < 0.1 m。这同时确认了 rollout 的第 i 个状态对应场景的第 i 步。如果误差很大，先停下来告诉我。
- `adversary: X m off its log`：X 应明显大于 0。如果接近 0，说明对手轨迹没有生效。代码层面已经确认 `eval_policy` 用全局 `env` 没有问题（`env.engine` 是全局单例），这里是实测确认。
- `plan error lag 0 / lag 1`：对手实际位置与计划轨迹的偏差。预期 lag 1 接近 0，也就是对手比计划晚一步执行。rollout 记录的是实际位置，所以不影响责任的计算，只作记录。
- 回放的自车不会报 `crash_vehicle`：MetaDrive 在碰撞检查之后又调用了回放车辆的 `before_step`，把当步的标志清掉了。所以 `collect_rollouts` 改为读 `ego_crash_flag`，CAT 的 `cat_advgen.py` 也是这么做的（提交见下）。在这之前采集的回放 rollout 里一次碰撞都没有，要重新采集。
- 2026-10-04 修复了 CAT 回放对手计划的一个错位（提交 0b94e88，见 README）：对手在日志里出现得晚的场景，原来会被放到原点若干步，然后整段计划都晚执行。这类场景在 500 个里有 24 个，测试集里有 8 个。修复前采集的对抗 rollout 要重新采集。

再采集训练好的策略：17 个模型（第 8b 步：5 组核心设置 × 3 个种子，加上两个消融的种子 0），每个模型在三种测试对手下各跑一次：

```bash
for seed in 0 1 2; do
  for name in replay cat cat_share cat_rss cat_fair2_0.1 cat_fairinf_0.5 cat_fair2_0.1_share; do
    [ -e models/${name}_s${seed}_actor ] || continue   # the ablations have only seed 0
    for adv in "--adv_selection cat" "$FAIR" "$ABL"; do
      tag=$(echo "$adv" | awk '{print $2}')
      nohup python -m scripts.responsibility.collect_rollouts --policy models/${name}_s$seed \
        --policy_name td3_${name}_s$seed --adversary $adv --out_dir rollouts \
        > logs/rollouts/td3_${name}_s${seed}_$tag.log 2>&1 &
    done
  done
done
wait
ls rollouts/*/                # 每个策略四个目录：none、cat、fair2_0.1、fairinf_0.5
ls rollouts/*/*/ | head       # 每个目录 100 个 <scene>.pkl
```

- 交叉评测矩阵（M1.4）：训练设置 × 测试对手（`none`、`cat`、`fair2_0.1`、`fairinf_0.5`）。每种测试对手下，所有策略面对的对手条件相同。
- 同一个模型的三次运行都会先跑正常回合（对手要针对这一回合的自车轨迹生成），所以 `none/` 会被写两次。两次的内容应该相同（策略和仿真都是确定性的），只是多花一些时间。

### 10b. 计算每个 rollout 的责任（M3.2）

与第 3 步是同一个脚本，只是加了 `--rollouts`。发生碰撞的回合还会做归因，结果写到 `crashes.csv`。

```bash
P=logs/responsibility/policies
for dir in rollouts/*/*/; do
  run=$P/$(basename $(dirname $dir))/$(basename $dir)
  for i in $(seq 0 3); do
    nohup python -m scripts.responsibility.compute_responsibility --rollouts $dir --out-dir $run \
      --num-shards 4 --shard-index $i --device cuda > /dev/null 2>&1 &
  done
done
wait
ls $P/*/*/windows*.csv | wc -l
```

- 规模：88 组 rollout 目录（回放 4 组 + 21 个模型 × 4 种测试对手），每组 100 个场景，共 8800 个场景 × 1 个 agent，约为第 3 步的 9 倍（第 3 步是 500 场景 × 2 个 agent）。
- 进程数：上面是 88 × 4 = 352 个进程，太多。按第 2 步的探测结果调整 `--num-shards`，或者分批跑：先跑 `none` 和 `cat` 两种（方向 3 的 M3.4 只需要这两种）。

### 10c. 对比表（M3.3、M3.4）

```bash
P=logs/responsibility/policies
runs=$(ls -d $P/replay/none $P/replay/cat $P/replay/fair* $P/td3_*/none $P/td3_*/cat $P/td3_*/fair*)
# 给已有的 crashes*.csv 补上（或重算）规则定责：RSS 和路权。只要场景和 rollout，不用 GPU，每个碰撞不到 1 秒
python -m scripts.responsibility.attribute_rules --runs $runs
# 在回放的 logged 驾驶上拟合等级
python -m scripts.responsibility.fit_levels --runs $runs --fit-runs $P/replay/none --out-dir $P/levels
# 阈值取自第一个 run（回放的 logged 驾驶）
python -m scripts.responsibility.compare_policies --runs $runs --hmm $P/levels/hmm.pkl --out-dir $P/compare
cat $P/compare/comparison.md
```

**参照为什么用回放，而不是第 3 步的 `logs/responsibility/sdc`：** MetaDrive 不生成静止车辆。在停车多的场景里，这会让 β_s 平均变化 0.4 m。回放的 logged 驾驶面对的场景与策略完全相同，比较才公平。

表里要看的：
- **碰撞率**和**自车责任碰撞占比**：碰撞有多少是自车的错。
- **激进 / 胆怯窗口占比**，以及相对参照的倍数（`× ref`）。
- **stopped**：停车窗口的占比。停着不动是最极端的胆怯，但这些窗口不参与判定，所以单独列出。
- **各等级占比**。

**M3.3 的验收：**
- `replay/none` 的激进和胆怯占比都应该较小。阈值定义保证：正值窗口里约 10% 超过激进阈值，负值窗口里约 10% 低于胆怯阈值。
- `replay/cat` 里，对手引起的碰撞应该大多判为 `other`。
- `rule agree`：与追尾规则（追尾归后车）的一致率，只统计两者都给出判定的碰撞（M2.1 的验收）。覆盖率见 `comparison.csv` 的 `rule_coverage`。

`comparison.md` 每个种子单独一行。`comparison_seeds.md` 按训练设置和测试对手对 3 个种子取均值 ± 标准差，并给出几张"训练设置 × 测试对手"的矩阵：碰撞率、自车责任碰撞占比、完成率、胆怯和激进窗口占比。这就是 M1.4 的交叉评测表和 M2.3 的 2×2 对比表。

对照计划里的假设：
- **M3.4**：`td3_cat` 对 `td3_replay`、`replay/none`。CAT 训练降低了碰撞率，胆怯窗口占比是人类的几倍？
- **M1.4**：`td3_cat_fair2_0.1` 在 `none` 和 `fair2_0.1` 测试上完成率更高、胆怯更少；在 `cat` 测试上碰撞率不明显变差。
- **β 的消融**：`td3_cat_fair2_0.1` 对 `td3_cat_fairinf_0.5`。如果两者一样，只约束可避免性就够了；如果前者更少胆怯，或者在 `none` 上开得更像人，β 就有它自己的作用。
- **M2.3**：`td3_cat_share` 对 `td3_cat`。完成率提高、胆怯减少，自车责任碰撞不增加。
- **RSS 基线**：`td3_cat_share` 对 `td3_cat_rss`。反事实的定责是否比 RSS 的规则定责更好。同时看 `RSS agree`：两种定责在真实碰撞上的一致率。
- **路权基线**（`responsibility/right_of_way.py`，加州交规 CVC）：RSS 只管同向碰撞，交叉、转弯、汇入的碰撞由路权规则定责。看 `RoW agree`（与反事实定责的一致率），以及 `comparison.csv` 里的 `right_of_way_coverage` 和 `baseline_coverage`（RSS 或路权给出判定的碰撞占比；路权主要补上 RSS 判不了的交叉、转弯和汇入碰撞）。

打包带回：

```bash
tar czf policies_$(date +%m%d).tgz $P/compare $P/levels $P/*/*/crashes*.csv logs/rollouts logs/blame
```

## 10d. 把 WOMD 分片转成 CAT 格式，并做路权规则的留出验证（不用 GPU）

CAT 的 500 个场景是路权规则的开发集，规则是看着这些场景里的错误补出来的。留出验证用 WOMD `validation_interactive` 的其余场景，每个场景取它的两个 object of interest 作为一对。

```bash
W=~/womd_v1_2_1/validation_interactive            # WOMD v1.2.1 的 tfrecord 分片（L4 服务器上现有 14 个，共 150 个）
C=~/womd_v1_2_1/cat_format/validation_interactive # 转成 CAT 格式的场景
# 转换：用 CAT 自己的转换函数，两个 OOI 都是车辆的场景全部保留，index.csv 标注 sdc_in_ooi 和 in_cat；可续跑
~/venvs/cat39/bin/python -m scripts.responsibility.convert_womd_split --tfrecords $W --out-dir $C --workers 8
# 先试点 200 个场景：看 errors 是否为 0
python -m scripts.responsibility.validate_right_of_way --scenes $C/scenes --n 200 \
  --exclude responsibility/unitraj_configs/cat_scenario_ids.txt --workers 8 --out-dir logs/responsibility/right_of_way/pilot
# 正式运行（可续跑）
python -m scripts.responsibility.validate_right_of_way --scenes $C/scenes \
  --exclude responsibility/unitraj_configs/cat_scenario_ids.txt --workers 8 \
  --out-dir logs/responsibility/right_of_way/validation_interactive
cat logs/responsibility/right_of_way/validation_interactive/summary.md
```

- 转换速度：CAT 的 497 个场景单进程 54 s（约 0.1 s/场景）。其余 136 个分片要用在 Waymo 官网接受过许可的账号下载到 `$W`（本机的服务账号无权访问 Waymo 的存储桶，2026-10-09 试过，返回 403），然后再跑同一条命令，已转换的分片会跳过。
- 转出的场景也能直接给开环的责任计算用（`compute_responsibility --scenes $C/scenes --use-ooi`）。

- 速度：500 个场景、8 个进程用 48 s。
- 要看的：每条规则"有路权者先通过"的比例和 95% 置信区间；`CVC: first-in` 和 `CVC: yield-right` 两行，用来检验"无管控路口默认关闭 CVC 规则"这个决定在留出集上是否仍然成立。
- 带回：`pairs.csv` 和 `summary.md`。

## 11. ⚠ UniTraj MTR：第二个预测模型（UNITRAJ_PLAN.md 的 U0、U2、U3、U5）

> **暂停（2026-10-04）：** 第二个预测模型改用 CAT-K 的 SMART（`SMART_PLAN.md`），不需要训练。本节保留备用。

目标：训练一个 Waymo 设定的 MTR（1.1 s 历史、8 s 未来、64 个意图点），用它重新计算责任，检查结论是否依赖预测模型。

本地已完成并测试：
- 数据接口：在真实场景上，历史和未来与日志的对齐误差小于 3e-6 m；
- `--model mtr` 接入；
- 校准、一致性检查和模型对照脚本的逻辑，用替身模型测试过。

**本地无法验证、要在服务器上确认的：**
- MTR 的 CUDA 算子能否编译；
- 真实模型的输出。

每一小步后面都写了检查方式，出错就停下来把日志发给我。

**CAT 的 500 个场景来自 WOMD 的 `validation_interactive`**，所以：
- **训练**只用 `training`，不会泄漏；
- **校准**用 `validation`，脚本会按 `responsibility/unitraj_configs/cat_scenario_ids.txt` 自动排除这 500 个场景（其中有 3 个重复，实际 497 个 id）。

### 11a. 环境（约 30 分钟）

UniTraj 放在 cat 旁边（`../UniTraj`），或者设置环境变量 `UNITRAJ=/path/to/UniTraj`。

```bash
nvcc --version                     # 需要 CUDA 12.x 编译器；没有就先 module load 或 conda 装 cuda-nvcc=12.1
bash scripts/responsibility/setup_unitraj_env.sh 2>&1 | tee logs/setup_unitraj.log
```

脚本建两个环境：
- `~/venvs/unitraj39`：训练 MTR、用 MTR 计算责任；
- `~/venvs/womd39`：把 WOMD 转成 ScenarioNet 格式。它依赖 TensorFlow 和 waymo-open-dataset，版本和训练环境冲突，所以单独建。

日志最后应看到 `MTR ops and scenarionet import` 和 `conversion environment ready`。

**GPU 冒烟测试**：用未训练的 MTR 跑一遍完整流程，在花时间转换数据、训练之前先确认环境没问题。

```bash
source ~/venvs/unitraj39/bin/activate
python -m scripts.responsibility.verify_unitraj --checkpoint random --n 2
```

- `repeat`、`batch`、`nms`、"masked slot vs deleted track" 必须 PASS。
- "近处 vs 远处车辆" 那一项在随机权重下可能 FAIL，属于正常。
- 最后一行打印构造输入和预测的速度。

### 11b. 下载和转换 WOMD

需要先在 Waymo 官网接受许可，用 `gsutil` 下载 `waymo_open_dataset_motion_v_1_2_0/uncompressed/scenario/` 下的 `training/` 和 `validation/`。

```bash
source ~/venvs/womd39/bin/activate
W=/data/womd/scenario          # tfrecord 所在目录
S=/data/womd_sn                # ScenarioNet 输出目录
# 先转换一小部分做试点（11c）：training 前 20 个分片，validation 前 5 个
python -m scenarionet.convert_waymo -d $S/training   --raw_data_path $W/training   --num_workers 32 --start_file_index 0 --num_files 20
python -m scenarionet.convert_waymo -d $S/validation --raw_data_path $W/validation --num_workers 32 --start_file_index 0 --num_files 5
ls $S/training | head; du -sh $S/*
```

- 输出目录的最后一级名字必须是 `training` 和 `validation`，两者不同。UniTraj 用路径的最后两级给缓存命名，同名会冲突。
- 转换报找不到 `waymo_open_dataset`，或版本不匹配：按 ScenarioNet 文档装它指定的 waymo-open-dataset 版本，再重跑。

### 11c. 试点：量磁盘和速度

**先试点，不要直接全量训练。** UniTraj 训练前会把所有样本预处理成不压缩的 h5 缓存。MTR 每个被预测的车约 1.8 MB（主要是 768 条地图折线），完整的 WOMD 训练集可能要若干 TB。

```bash
source ~/venvs/unitraj39/bin/activate
python -m scripts.responsibility.train_unitraj --exp-name pilot \
  --train-data /data/womd_sn/training --val-data /data/womd_sn/validation \
  --cache-path /data/unitraj_cache --out-dir /data/unitraj_ckpt --devices 0 --epochs 1 --workers 16 \
  2>&1 | tee logs/unitraj_pilot.log
du -sh /data/unitraj_cache/*/*
grep -E "Loaded .* samples" logs/unitraj_pilot.log
```

- **磁盘：** 用"缓存大小 ÷ 样本数"得到每个样本占多少，乘以全量样本数，估算全量缓存大小。20 个分片大约是训练集的 2%，所以全量约为试点的 50 倍。
- **速度：** 看进度条的 it/s，乘以 batch size 256，得到每秒样本数，再估算每个 epoch 的时间。
- **磁盘不够时的办法：**
  - 只转换一部分分片，例如 `--num_files 300`，约 30% 的数据；
  - 用 `--max-data-num` 限制训练样本数。注意它在缓存建好之后才生效，不能省磁盘；
  - 在 `unitraj_configs/method/MTR_womd.yaml` 里把 `max_num_roads` 从 768 降到 384，每个样本约减半。这会偏离 MTR 原设定，要在报告里说明。

试点结果（每样本大小、每秒样本数、估算的全量时间）先发给我，再定全量方案。

### 11d. 正式训练（几天）

按 11c 的结论转换剩下的分片（`--start_file_index 20 --num_files N`，写到同一个 `$S/training`），然后：

```bash
source ~/venvs/unitraj39/bin/activate
nohup python -m scripts.responsibility.train_unitraj --exp-name mtr_womd \
  --train-data /data/womd_sn/training --val-data /data/womd_sn/validation \
  --cache-path /data/unitraj_cache --out-dir /data/unitraj_ckpt --devices 0 1 2 3 --workers 16 \
  > logs/unitraj_train.log 2>&1 &
tail -f logs/unitraj_train.log
```

- 多卡时，第一张卡的进程先建缓存，然后才启动训练。
- 中断后续训：加 `--resume /data/unitraj_ckpt/mtr_womd/last.ckpt`。
- 训练曲线：`/data/unitraj_ckpt/mtr_womd/logs/`（CSV）；加 `--wandb` 则记到 Weights & Biases。
- 最好的检查点按验证集的 `val/brier_fde` 保存，文件名形如 `epoch12-brier_fde1.84.ckpt`。下文用 `CKPT=$(ls /data/unitraj_ckpt/mtr_womd/epoch*.ckpt)` 表示。
- **验收：** 验证集的 minADE / minFDE 与 MTR 论文在 WOMD 上的数字（约 0.60 / 1.22 m）相差不超过 10%。UniTraj 的指标名和官方评测略有不同，看 `val/minADE6`、`val/minFDE6`。

### 11e. 校准和一致性检查（U3，约 1 小时）

```bash
source ~/venvs/unitraj39/bin/activate
CKPT=$(ls /data/unitraj_ckpt/mtr_womd/epoch*.ckpt)
python -m scripts.responsibility.calibrate_unitraj --checkpoint $CKPT --scenes /data/womd_sn/validation --n 3000
python -m scripts.responsibility.verify_unitraj --checkpoint $CKPT --n 5 2>&1 | tee logs/verify_unitraj.log
```

- **校准**打印温度缩放前后的四个指标：目标意图的 NLL、目标意图的平均概率、top-1 准确率、ECE。结果写到 `$CKPT.calibration.json`，之后计算责任时自动使用这个温度。
  - T > 1 说明原始分数过于自信，T < 1 说明过于保守。
  - 缩放后 ECE 应明显下降。
- **一致性检查**必须 `ALL CHECKS PASSED`。
  - 如果只有"batch"或"masked vs deleted"失败，看打印的差值大小：小于 1e-3 通常是 GPU 浮点误差，把日志发给我判断。

### 11f. 用 MTR 计算责任（U5，与第 3 步同规模）

```bash
source ~/venvs/unitraj39/bin/activate
# 采样 40 条（与 DenseTNT 相同的做法）
OUT=logs/responsibility_mtr MODEL=mtr CHECKPOINT=$CKPT RECORDS=1 bash scripts/responsibility/run_h200.sh
# 用全部 64 个意图、按概率加权的精确 CVaR
OUT=logs/responsibility_mtr_w MODEL=mtr CHECKPOINT=$CKPT EXTRA="--motion-set weighted" AGENTS=sdc \
  bash scripts/responsibility/run_h200.sh
```

MTR 跑得比 DenseTNT 快：一个窗口里的所有输入一次前向算完。但构造输入仍在 CPU 上，`SHARDS` 仍然按 CPU 核数设。

### 11g. 模型对照：结论是否依赖预测模型

```bash
C=logs/responsibility/model_comparison
python -m scripts.responsibility.compare_models --runs logs/responsibility/sdc logs/responsibility_mtr/sdc --out-dir $C/sdc
python -m scripts.responsibility.compare_models --runs logs/responsibility/adv logs/responsibility_mtr/adv --out-dir $C/adv
python -m scripts.responsibility.compare_models --runs logs/responsibility_mtr/sdc logs/responsibility_mtr_w/sdc \
  --labels mtr-sampled mtr-weighted --out-dir $C/sampled_vs_weighted
python -m scripts.responsibility.fit_levels --runs logs/responsibility_mtr/sdc logs/responsibility_mtr/adv \
  --out-dir logs/responsibility_mtr/levels
cat $C/*/model_comparison.md
```

要看的：
- 逐窗口 β_s、β_c 的 Spearman 相关；
- 激进和胆怯标记的 Cohen's κ；
- 场景判定的一致率；
- 两个模型的 HMM 等级结构是否相似。

`sampled_vs_weighted` 显示采样噪声有多大，用来决定 MTR 默认用哪种做法。

**策略评测也用 MTR 算一遍**（需要第 10 步的 rollout）：

```bash
P=logs/responsibility/policies; PM=logs/responsibility/policies_mtr
for dir in rollouts/*/*/; do
  run=$PM/$(basename $(dirname $dir))/$(basename $dir)
  for i in $(seq 0 3); do
    nohup python -m scripts.responsibility.compute_responsibility --rollouts $dir --out-dir $run \
      --model mtr --checkpoint $CKPT --num-shards 4 --shard-index $i --device cuda > /dev/null 2>&1 &
  done
done; wait
runs=$(ls -d $PM/replay/none $PM/replay/cat $PM/replay/fair* $PM/td3_*/none $PM/td3_*/cat $PM/td3_*/fair*)
python -m scripts.responsibility.compare_policies --runs $runs --out-dir $PM/compare
python -m scripts.responsibility.compare_models --runs $P/replay/none $PM/replay/none \
  --policy-tables $P/compare/comparison.csv $PM/compare/comparison.csv --out-dir $C/policies
```

**论文里最关键的一句话**来自最后那张表：两个模型下，策略的排序和"是否比参照更激进/更胆怯"是否一致。

打包带回：

```bash
tar czf unitraj_$(date +%m%d).tgz $C logs/responsibility_mtr/levels logs/verify_unitraj.log logs/unitraj_pilot.log \
  $CKPT.calibration.json /data/unitraj_ckpt/mtr_womd/logs
```
