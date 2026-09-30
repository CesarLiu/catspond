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
- 前 20 个场景上，只有 τ = 2 m 能保住 30%，而 logged 对手的 q90 约 0.35 m（那里的碰撞率只有 10–15%）。如果 500 个场景也是这样，两个标准不能同时满足，要取舍。备选：取推荐的 τ；或者放宽到 τ = 1（约 25%）；或者两组都训练。
- 下文用 `TAU` 和 `RHO` 表示选定的值：

```bash
TAU=1.0   # 按上面 --summarize 输出的 logged adversary q90 修改
RHO=0.3   # 按 fair 规则的碰撞率 / 不可避免碰撞的权衡修改
```

## 7. ⚠ 安装 CAT 的 MetaDrive 仿真依赖（同一个 venv）

CAT 的 `requirements.txt` 固定了 `torch==1.12.0+cu116`，在 H200 上不能用，所以要排除它和 TF 相关的包：

```bash
grep -v -E "^(torch|torchvision|waymo-open-dataset|pickle5)" requirements.txt > /tmp/cat_requirements.txt
python -m pip install "pip<24.1"        # gym==0.22.0 的元数据在新版 pip 下安装会报错
python -m pip install -r /tmp/cat_requirements.txt
# pickle5 是 Python < 3.8 的 backport（需要编译）；3.9 的 pickle 自带 protocol 5，CAT 只是按名字导入
echo "from pickle import *  # noqa" > "$(python -c 'import site; print(site.getsitepackages()[0])')/pickle5.py"
python -c "import metadrive, gym, panda3d; print('metadrive ok')"   # 仓库自带的 metadrive/ 从根目录直接导入
export SDL_VIDEODRIVER=dummy             # 无显示器时 pygame 的 top-down 渲染需要
```

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

三组设置：
- `replay`：没有对手；
- `cat`：CAT 原版对手；
- `cat_constrained<TAU>`：责任约束对手；
- `cat_fair<TAU>_<RHO>`：公平对手（方向 1 的 M1.4）。

```bash
mkdir -p logs/rl
g=0
for seed in 0 1 2; do
  for setting in "--mode replay" "--mode cat" "--mode cat --adv_selection constrained --resp_threshold $TAU" \
                 "--mode cat --adv_selection fair --resp_threshold $TAU --resp_avoid $RHO"; do
    name=$(echo "$setting" | tr -d ' -' )_s$seed
    CUDA_VISIBLE_DEVICES=$((g % $(nvidia-smi -L | wc -l))) nohup python cat_RLtrain.py $setting --seed $seed --save_model \
      > logs/rl/$name.log 2>&1 &
    g=$((g+1))
  done
done
```

- 进度：`tail -f logs/rl/*.log`。
- 曲线数据：`logs/<名字>_MDWaymo-seed<seed>/logger.csv`。名字为 `replay`、`cat`、`cat_constrained<TAU>`、`cat_fair<TAU>_<RHO>`，各组分开记录，不会互相覆盖。
- 模型：`models/<名字>_s<seed>*`（加了 `--save_model` 才会保存；每个种子单独一份，第 10 步要用）。

### 8c. 画学习曲线

```bash
for key in route_completion_normal route_completion_adv crash_rate_normal crash_rate_adv; do
  PYTHONPATH=. python scripts/plot.py --log_dir logs/ --title "$key" --ykey $key --xlim 1000000 --save
done
```

比较三组在正常场景和对抗场景下的完成率与碰撞率。关键问题：用"合理的"对手训练出来的策略，泛化是否更好。

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
| RL 训练对比 ⚠ | 8b–8c | `logs/*_MDWaymo-seed*/logger.csv` 和曲线 |
| 策略的双向风格评测 ⚠ | 10 | `rollouts/`、`logs/responsibility/policies/compare/comparison.md` |

## 10. ⚠ 策略的双向驾驶风格评测（方向 3，M3.0–M3.4）

依赖第 8b 步保存的模型（`models/<名字>_s<seed>*`）。`replay` 基线不需要模型，可以先跑，顺便完成第 10a 步的核查。

### 10a. 采集 rollout，并核查 CAT 的评测流程（M3.0、M3.1）

```bash
mkdir -p logs/rollouts
# 基线：logged ego 回放，正常场景 + CAT 对手（测试集 400–499）
python -m scripts.responsibility.collect_rollouts --policy replay --adversary --out_dir rollouts \
    2>&1 | tee logs/rollouts/replay.log
tail -4 logs/rollouts/replay.log
```

最后几行是核查结果：
- `replay: max ego error ... m`：应 < 0.1 m。这同时确认了 rollout 的第 i 个状态对应场景的第 i 步。如果误差很大，先停下来告诉我。
- `adversary: X m off its log`：X 应明显大于 0。如果接近 0，说明对手轨迹没有生效。代码层面已经确认 `eval_policy` 用全局 `env` 没有问题（`env.engine` 是全局单例），这里是实测确认。
- `plan error lag 0 / lag 1`：对手实际位置与计划轨迹的偏差。预期 lag 1 接近 0，也就是对手比计划晚一步执行。rollout 记录的是实际位置，所以不影响责任的计算，只作记录。

再采集训练好的策略（12 个模型并行，每个一个进程）：

```bash
for seed in 0 1 2; do
  for name in replay cat cat_constrained$TAU cat_fair${TAU}_$RHO; do
    nohup python -m scripts.responsibility.collect_rollouts --policy models/${name}_s$seed \
      --policy_name td3_${name}_s$seed --adversary --adv_selection cat --out_dir rollouts \
      > logs/rollouts/td3_${name}_s$seed.log 2>&1 &
  done
done
wait
ls rollouts/*/*/ | head      # 每个目录 100 个 <scene>.pkl
```

- 所有策略都在 CAT 原版对手（`--adv_selection cat`）下评测，保证对手条件相同。
- 想换成责任约束的对手，加 `--adv_selection constrained --resp_threshold $TAU`，输出目录名会变成 `constrained<TAU>`。

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

- 规模：26 组 rollout 目录（回放 2 组 + 12 个模型 × 2 种对手模式），每组 100 个场景，共 2600 个场景 × 1 个 agent。时间约为第 3 步的 2.6 倍（第 3 步是 500 场景 × 2 个 agent）。
- 进程数：上面是 26 × 4 = 104 个进程。按第 2 步的探测结果调整 `--num-shards`。

### 10c. 对比表（M3.3、M3.4）

```bash
P=logs/responsibility/policies
runs=$(ls -d $P/replay/none $P/replay/cat $P/td3_*/none $P/td3_*/cat)
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

每个种子单独一行。3 个种子的均值和标准差可以从 `comparison.csv` 直接算。

打包带回：

```bash
tar czf policies_$(date +%m%d).tgz $P/compare $P/levels $P/*/*/crashes*.csv logs/rollouts
```
