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

## 5. （可选）敏感性分析

每组设置单独一个输出目录；`--horizon`、`--d-sat` 等参数通过 `EXTRA` 传入：

```bash
OUT=logs/sens_h30   EXTRA="--horizon 30"  SHARDS=32 bash scripts/responsibility/run_h200.sh   # 3 s 度量时长
OUT=logs/sens_dsat5 EXTRA="--d-sat 5"     SHARDS=32 bash scripts/responsibility/run_h200.sh   # 更近的交互范围
OUT=logs/sens_s80   SAMPLES=80            SHARDS=32 bash scripts/responsibility/run_h200.sh   # 样本数翻倍（看 CVaR 的稳定性）
```

比较各组的 `summary` 和等级表。样本数翻倍后数值变化很小，说明 40 个样本够用。

## 6. 对抗生成：离线对比与阈值标定

对比 CAT 原版和责任约束版的选择规则。这一步不需要 MetaDrive，按逻辑 ego 开环评估，分 10 份并行跑：

```bash
mkdir -p logs/advgen
for i in $(seq 0 9); do
  python -m scripts.responsibility.benchmark_advgen --first $((i*50)) --n 50 \
    --thresholds 0.25 0.5 1 2 --out logs/advgen/part_$i.csv --device cuda \
    > logs/advgen/part_$i.log 2>&1 &
done
wait
grep -h "reproduces" logs/advgen/part_*.log             # 每份都应是 50/50：cat 规则与 CAT 原版一致
python -m scripts.responsibility.benchmark_advgen --summarize logs/advgen/part_*.csv
```

结果怎么读：
- 每条规则两列：预测碰撞率、被选中对手轨迹的 β。
- 最后一行是 logged 对手自身 β 的分布，τ 在这里取，**推荐取 q90**。
- 下文用 `TAU` 表示选定的阈值：

```bash
TAU=1.0   # 按上面 --summarize 输出的 logged adversary q90 修改
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
```

看进度条里的 `avg_attack_success_rate` 和 `avg_compute_time`。责任约束版在最后还会打印选中对手轨迹的平均 β。

### 8b. RL 训练（TD3，1e6 步，每组 3 个种子）

三组设置：
- `replay`：没有对手；
- `cat`：CAT 原版对手；
- `cat_constrained<TAU>`：责任约束对手。

```bash
mkdir -p logs/rl
g=0
for seed in 0 1 2; do
  for setting in "--mode replay" "--mode cat" "--mode cat --adv_selection constrained --resp_threshold $TAU"; do
    name=$(echo "$setting" | tr -d ' -' )_s$seed
    CUDA_VISIBLE_DEVICES=$((g % $(nvidia-smi -L | wc -l))) nohup python cat_RLtrain.py $setting --seed $seed --save_model \
      > logs/rl/$name.log 2>&1 &
    g=$((g+1))
  done
done
```

- 进度：`tail -f logs/rl/*.log`。
- 曲线数据：`logs/<名字>_MDWaymo-seed<seed>/logger.csv`。名字为 `replay`、`cat`、`cat_constrained<TAU>`，三组分开记录，不会互相覆盖。
- 模型：`models/<名字>*`（加了 `--save_model` 才会保存）。

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
| 敏感性（可选） | 5 | `logs/sens_*` |
| 对抗选择离线对比与 τ 标定 | 6 | `--summarize` 表格、TAU |
| 闭环攻击成功率 ⚠ | 8a | `logs/advgen/closed_*.log` |
| RL 训练对比 ⚠ | 8b–8c | `logs/*_MDWaymo-seed*/logger.csv` 和曲线 |

## 尚未实现：训练后策略自身的责任

"训练后的 ego 策略本身有多激进"这一项还缺一个脚本：在 MetaDrive 里用训练好的策略跑 500 个场景，记录 ego 轨迹。
`Scene.with_track(...)` 已经可以把 ego 的轨迹替换成仿真轨迹；再用第 3 步同样的流程计算责任，并用 `summarize_responsibility --reference logs/responsibility/sdc` 和 logged 驾驶对比。
需要的话我可以补上这个脚本（它依赖 MetaDrive，需要在服务器上调试）。
