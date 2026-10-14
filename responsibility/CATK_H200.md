# 在另一台 GPU 服务器（如 H200）上跑 CAT-K / SMART 相关实验

本文件列出本仓库里所有用到 CAT-K 的 SMART 的实验：要搬哪些东西，怎么装环境，每个实验的命令、产出和在 L4 上的耗时。实验的背景和结果在 `SMART_PLAN.md`，以及 README 的 "Does the verdict depend on the motion model? (SMART)" 和 "Can it judge aggression?" 两节。

catk 仓库自己的实验（在 WOMD 上训练奖励网络、用责任奖励微调 SMART、WOSAC 评测）不在这里，见 catk 的 `README_responsibility.md`。它们需要 WOMD 的训练集（TB 级），和下面这些实验无关。

## 1. 要搬什么

**代码：从 git 拉。** 先确认两个仓库的这些分支都已推送（`git status -sb` 不显示 ahead）：

| 仓库 | 分支 / 提交 | 放在 |
|---|---|---|
| `git@github.com:CesarLiu/catspond.git` | `counterfactual-responsibility`（`gpudrive-ppo` 也合并了同样的代码） | 任意目录，下文称 `<repo>`，命令都从它的根目录运行 |
| `git@github.com:CesarLiu/catk4test.git` | `main`（本机是 `0763864`） | `~/catk`（脚本默认；别处的话设 `CATK_ROOT`） |

**数据和检查点：不在 git 里，用 `scripts/responsibility/pack_catk_h200.sh` 打包。** WOMD 的使用条款不允许公开发布场景和用它训练的模型，所以只在自己的机器之间拷贝。

| 部分（`PARTS`） | 内容 | 大小 | 哪些实验要 |
|---|---|---|---|
| `core` | `raw_scenes_500/`（CAT 的 500 个场景）；`catk/ckpts/clsft_E9.ckpt`（SMART 检查点） | 317 MB + 69 MB | 全部 |
| `aggression` | `adv_scenes/cat/`（CAT 的对抗场景，含 `index.json`）；`logs/responsibility/validity/scenes/`（其余条件场景） | 317 + 452 MB | E5、E6 |
| `compare` | DenseTNT 的两次运行 `logs/responsibility/{sdc,adv}` 里的 `windows*.csv` 和 `config.json` | 4 MB | E1 的对照 |
| `rollouts` | `rollouts/replay/`（回放的自车，对 4 种对手） | 3 MB | E3 |
| `v121` | `womd/cat_scenarios_v121.tfrecord`（CAT 的 497 个 WOMD v1.2.1 原始场景） | 411 MB | E2 |
| `densetnt` | `advgen/pretrained/densetnt.bin` | 120 MB | 只有在服务器上重建条件场景或重跑 DenseTNT 时才要 |
| `results` | 已有的 SMART 结果：`logs/catk/`，以及 `validity/runs/smart`、`validity/smart` | 330 MB | 想续跑或直接对照时 |

不需要：MetaDrive、GPUDrive、WOMD 的原始分片（`v121` 已经把 CAT 的场景单独抽出来了）。

**在本机打包。** 数据分散在两个工作目录里，所以要指定位置；默认的几个部分约 1.5 GB：

```bash
cd /home/Cesar/catspond-cr
C=/home/Cesar/catspond
DRY=1 CAT_SCENES=$C/adv_scenes/cat DTNT_RUNS=$C/logs/responsibility ROLLOUTS=$C/rollouts/replay \
  CATK_LOGS=$C/logs/catk DENSETNT=$C/advgen/pretrained/densetnt.bin \
  bash scripts/responsibility/pack_catk_h200.sh                 # 先列出各部分和大小
OUT=/tmp/catk_h200.tar CAT_SCENES=$C/adv_scenes/cat DTNT_RUNS=$C/logs/responsibility ROLLOUTS=$C/rollouts/replay \
  bash scripts/responsibility/pack_catk_h200.sh                 # 默认：core aggression compare rollouts v121
# 加上已有结果：PARTS="core aggression compare rollouts v121 results" CATK_LOGS=$C/logs/catk ...
rsync -avP /tmp/catk_h200.tar <h200>:~/
```

**在服务器上解压。** tar 里有三个顶层目录：

```bash
git clone -b counterfactual-responsibility git@github.com:CesarLiu/catspond.git ~/catspond
git clone git@github.com:CesarLiu/catk4test.git ~/catk
tar xf ~/catk_h200.tar -C ~/catspond --strip-components=1 repo
tar xf ~/catk_h200.tar -C ~/catk --strip-components=1 catk
mkdir -p ~/womd && tar xf ~/catk_h200.tar -C ~/womd --strip-components=1 womd   # 只在打包了 v121 时
```

## 2. 环境

需要两个环境：

1. **catk 的环境**（Python 3.11、torch 2.4.1 + cu121、PyG 2.6.1）。跑 SMART，也负责导出、导入 catk 的数据。torch 2.4.1 的 cu121 版本带 Hopper（sm_90）的 kernel，H200 可用。

   ```bash
   cd ~/catk && bash install/setup_server.sh                    # 装到 ~/venvs/catk；CUDA_TAG=cu124 可换 wheel
   source ~/venvs/catk/bin/activate && bash install/check_env.sh  # import、GPU、catk 的单元测试
   ```

2. **本仓库的环境**（Python 3.9）。跑评估、对照和画图，以及需要 DenseTNT 的步骤。

   ```bash
   cd ~/catspond && bash scripts/responsibility/setup_env.sh   # 装到 ~/venvs/cat39
   ~/venvs/cat39/bin/python -m pytest tests/responsibility -q   # 不需要模型和数据
   ```

下文的 `python` 指 `~/venvs/cat39/bin/python`。`run_smart.sh` 和 `run_aggression_smart.sh` 会自己调用 catk 的 Python（`PYTHON`，默认 `~/venvs/catk/bin/python`）。

**冒烟测试（约 4 分钟）：** 在 2 个场景上走完 导出 → SMART → 导入 的全流程，输出放到临时目录：

```bash
mkdir -p /tmp/smoke && ln -sfn $PWD/logs/responsibility/validity/scenes /tmp/smoke/scenes
VAL=/tmp/smoke SETS="near30_0.5" N=2 SHARDS=1 bash scripts/responsibility/run_aggression_smart.sh
ls /tmp/smoke/runs/smart/adv_maxgap/obs    # 0.pkl 1.pkl
```

**H200 上的两个参数：**

- `BATCH`：每次模型调用的场景副本数，默认 32。在 L4（22 GB）上，32 的显存峰值是 6.3 GB；在一个有 61 个 agent 的场景里，128 爆了显存。H200 有 141 GB，可以先试 128 或 256。
- `SHARDS`：并行进程数。每个进程占一份显存，也占 CPU（token 化和地图处理在 CPU 上）。按显存和 CPU 核数一起定。

两者都不改变结果：分片和批量大小不影响数值（catk 的检查"批量计算等于逐个计算"已通过）。中断后重跑同样的命令即可续跑，完成的场景会跳过。

## 3. 实验

L4 上的耗时都是实测，H200 上还没测过。

### E1　SMART 测 CAT 的 500 个 logged 场景，与 DenseTNT 对照（SMART_PLAN S3、S4）

```bash
SHARDS=4 bash scripts/responsibility/run_smart.sh            # 导出到 logs/catk/scenes/cache，结果在 .../run/obs
~/venvs/catk/bin/python -m scripts.responsibility.import_catk --run logs/catk/scenes/run \
    --out-dir logs/catk/scenes/windows --data "CAT's scenes (WOMD v1.1)"
for a in sdc adv; do
  python -m scripts.responsibility.compare_models --runs logs/responsibility/$a logs/catk/scenes/windows/$a \
      --labels DenseTNT SMART --out-dir logs/catk/compare/densetnt_vs_smart_$a
done
```

- 产出：每个 agent 6,500 个窗口；`model_comparison.md` 和散点图。
- 耗时：L4 上单进程约 10 小时，共 144 万条滚动轨迹。
- 已有结果：逐窗口 β_s 的 Spearman 0.60–0.63；激进标记 κ 0.17–0.24；场景层面 Spearman 0.77–0.83。

### E2　WOMD v1.2.1 原生缓存：数据版本的影响

```bash
~/venvs/catk/bin/python -m scripts.responsibility.cache_womd_for_cat --catk-root ~/catk \
    --tfrecords ~/womd/cat_scenarios_v121.tfrecord --out-dir logs/catk/v121
EXPORT=0 OUT=logs/catk/v121 SHARDS=4 bash scripts/responsibility/run_smart.sh
~/venvs/catk/bin/python -m scripts.responsibility.import_catk --run logs/catk/v121/run \
    --out-dir logs/catk/v121/windows --data "CAT's scenes from WOMD v1.2.1"
for a in sdc adv; do
  python -m scripts.responsibility.compare_models --runs logs/catk/scenes/windows/$a logs/catk/v121/windows/$a \
      --labels SMART_v1.1 SMART_v1.2.1 --out-dir logs/catk/compare/smart_v11_vs_v121_$a
done
```

- 耗时：约 10 小时（L4）。
- 已有结果：497 个场景全部找到；v1.1 对 v1.2.1 的 κ 0.86–0.90，可见数据版本几乎没有影响。

### E3　回放参照的 rollout（测试场景 400–499，4 种对手）

```bash
for adv in none cat fair2_0.1 fairinf_0.5; do
  ROLLOUTS=rollouts/replay/$adv OUT=logs/catk/replay_$adv SHARDS=4 bash scripts/responsibility/run_smart.sh
  ~/venvs/catk/bin/python -m scripts.responsibility.import_catk --run logs/catk/replay_$adv/run \
      --rollouts rollouts/replay/$adv --out-dir logs/catk/replay_$adv/windows --data "replay vs $adv"
done
```

- 耗时：4 组合计约 3 小时 15 分（L4）。
- 已有结果：对手对自车 β_s 的场景最大值，中位数依次是 logged +0.03、fair +0.77、只约束可避免性 +1.27、CAT +1.98 m。
- SMART_PLAN S4 里的两张表（对手 β 分布、DenseTNT 与 SMART 的碰撞判定一致率）当时是临时计算的，还没有对应的脚本。

### E4　反应式反事实（前 30 个场景）

```bash
N=30 OUT=logs/catk/trial30_hidden bash scripts/responsibility/run_smart.sh
N=30 OUT=logs/catk/trial30_logged EXTRA="--neighbor-future logged" bash scripts/responsibility/run_smart.sh
for m in hidden logged; do
  ~/venvs/catk/bin/python -m scripts.responsibility.import_catk --run logs/catk/trial30_$m/run \
      --out-dir logs/catk/trial30_$m/windows --data "first 30, $m"
done
for a in sdc adv; do
  python -m scripts.responsibility.compare_models --runs logs/catk/trial30_hidden/windows/$a \
      logs/catk/trial30_logged/windows/$a --labels hidden logged --out-dir logs/catk/trial30_compare_$a
done
```

- 原理：`EXTRA` 里的 `--neighbor-future logged` 写在 `run_smart.sh` 固定的 `hidden` 之后，所以覆盖它。
- 已有结果：β_s 的 Spearman 0.93，激进标记 κ 0.58–0.65。

### E5　激进性检验的 SMART 版（还没跑完的主要实验）

条件场景来自 `aggression` 部分。`run_aggression_smart.sh` 对每个场景集合跑前 N 个场景，再导入到 `logs/responsibility/validity/runs/smart/<条件>`：

```bash
SHARDS=4 bash scripts/responsibility/run_aggression_smart.sh                         # 6 个集合，各 100 个场景
SETS="fairinf_0.5 near30_0.5" SHARDS=4 bash scripts/responsibility/run_aggression_smart.sh   # 只补还缺的两个
python -m scripts.responsibility.evaluate_aggression --config smart --frame-s 0.5 \
    --cat-scenes adv_scenes/cat --json logs/responsibility/validity/eval_smart.json
python -m scripts.responsibility.plot_aggression --out-dir logs/responsibility/validity/figures \
    --cat-scenes adv_scenes/cat --smart-json logs/responsibility/validity/eval_smart.json \
    --dtnt-json logs/responsibility/validity/eval_dtnt_s5.json
```

- 6 个集合：logged、cat、near0_1000（mode）、near1_0.5、fairinf_0.5、near30_0.5（max-gap）。
- 耗时：L4 上双进程，每个集合 100 个场景约 40–60 分钟。
- 已有结果（CAT、mode 两个条件）：CAT 攻击者在 10% 误报下的检出率 86%，β_c AUROC 0.81，攻击者 vs 受害者 0.69。near 1 m 已跑完，但还没进评估。
- 还缺：fair-inf 和 max-gap。补上后才能回答两个问题：激进程度是否单调，timid 能否测出。
- DenseTNT 的对照 JSON 用本仓库的环境生成（只读已有的 DenseTNT 运行，不需要 GPU）：`python -m scripts.responsibility.evaluate_aggression --config paper --subsample 5 --frame-s 0.5 --cat-scenes adv_scenes/cat --json logs/responsibility/validity/eval_dtnt_s5.json`。它需要 `validity/runs/paper`。这些运行不在打包里，在本机生成 JSON 再拷过去更省事。

### E6　SMART 的机制探针（惰性、对攻击者的反应）

用 E5 里 cat 和 logged 两个集合的 catk 缓存：

```bash
V=$PWD/logs/responsibility/validity
cd ~/catk && TF_CPP_MIN_LOG_LEVEL=3 ~/venvs/catk/bin/python $OLDPWD/scripts/responsibility/probe_smart.py \
    --cat-cache $V/smart/cat/cache --logged-cache $V/smart/logged/cache \
    --index $OLDPWD/adv_scenes/cat/index.json --out $V/smart_probe.csv --limit 100
```

- 已有结果：碰撞前 0.5 s，SMART 预测 ego 少走 0.72 m（DenseTNT 是 +0.07 m）；攻击者的备选轨迹有 61% 也进入 ego 3 m 以内。
- 所有场景共用一个带种子的随机流：只跑子集（`--limit`）时，逐帧值会和全量运行不同。

### E7（可选）　导出与 catk 原生缓存的逐字段核对

需要 catk 自己的验证集缓存 `~/catk/data/womd_cache/validation`（58 MB，不在打包里，从本机 `~/catk/data/` 拷）：

```bash
~/venvs/catk/bin/python -m scripts.responsibility.verify_catk_export --export logs/catk/scenes \
    --native ~/catk/data/womd_cache/validation
```

## 4. 带回本机的结果

- `logs/catk/<运行>/`：`windows/`、`run/config.json`、`compare/`。`run/obs` 较大，需要重新导入时才带。
- E5：`logs/responsibility/validity/runs/smart/`、`smart/*/run/`、`eval_smart.json` 和 `figures/`。
- E6：`smart_probe.csv`。

`logs/` 不进 git。按 CLAUDE.md 的约定，把数字写进 README 和 `SMART_PLAN.md` 的状态块后，和代码改动一起提交。

## 5. 常见问题

- **显存不够（`CUDA out of memory`）：** 降低 `BATCH`，或减少 `SHARDS`。
- **某个分片失败：** `run_smart.sh` 会提示 `a shard failed`。看 `OUT/logs/shard_*.log`，然后原样重跑，会自动续跑。
- **导入时提示 agent 不是 object of interest：** WOMD v1.2.1 给一个场景的两辆车重新编了号，`import_catk` 会计数并跳过这两辆车，属于正常现象。
- **脚本里的 `grep`：** 在本机的 shell 里，`grep` 是个会遵守 `.gitignore` 的函数。在新机器上不会有这个问题。
