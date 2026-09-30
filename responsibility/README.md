# Counterfactual responsibility with CAT's DenseTNT

Measures whether a vehicle — by default the self-driving car — drives
aggressively, using the counterfactual responsibility metrics of Hsu et al.,
["Interpretable Trajectory Prediction for Autonomous Vehicles via
Counterfactual Responsibility"](https://saferobotics.princeton.edu/research/responsibility)
(IROS 2023), with CAT's pretrained DenseTNT as the model of what drivers do.
It is a port of the SMART/WOMD implementation in the `catk` repository; the
geometry, CVaR and HMM code are shared verbatim.

At every context step k of a scene (every 0.5–1 s):

| quantity | question | definition |
|---|---|---|
| **safety responsibility** β_s (m) | Did the agent give up safety margin compared with what it could otherwise have done? | CVaR_α over N DenseTNT samples of the agent's motion of D_g(sample, b) − D_g(logged, b), max over interacting neighbours b. D_g = closest approach over the next 2 s, saturated at 10 m |
| **courtesy responsibility** β_c (nats) | Did its presence force others to change plans? | KL(π_b(goal \| scene) ‖ π_b(goal \| scene without the agent)), exact over b's goal grid, max over vehicle neighbours b |

β_s > 0: most of the agent's own alternatives would have kept more distance to
someone than it did. β_c large: some vehicle intends something quite different
because of the agent. Both are open-loop, like the paper: DenseTNT only sees
the 1.1 s of history up to k, never anyone's future.

## Setup

The code runs in CAT's environment (Python 3.9; `advgen/utils_cython*.so` is
built for it) plus `pytest`. Without conda, a CPU environment with
[uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.9
uv venv --python 3.9 --seed ~/venvs/cat39 && source ~/venvs/cat39/bin/activate
uv pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
uv pip install "tensorflow-cpu==2.12.0" "numpy<1.24" pyyaml matplotlib tqdm pytest scipy     "opencv-python-headless==4.7.0.72"   # MP4 videos; without it the visualisation writes GIF only
```

MetaDrive is not needed: scenes are read straight from `raw_scenes_500/*.pkl`.
`advgen/pretrained/densetnt.bin` must be in place (see the main readme).
Run everything from the repository root.

```bash
python -m pytest tests/responsibility -q                       # no model or data needed
python -m scripts.responsibility.verify_densetnt --n 5         # against CAT on real scenes
```

`verify_densetnt` checks that the wrapper rebuilds CAT's inputs field by field
and reproduces its 32 adversary trajectories (to ~1e-12 m), that goal sampling
matches the goal distribution, that repeated passes agree exactly, that
removing an agent drops exactly its polyline and moves a nearby vehicle's
distribution more than a far one, that DenseTNT's partner slot is
immaterial (~1e-7 nats), and that a replayed rollout (below) gives the logged
values once the objects MetaDrive does not spawn are left out.

## Is the ego aggressive?

```bash
# 1. responsibility of the self-driving car in every window of 50 scenes
python -m scripts.responsibility.compute_responsibility --out-dir logs/responsibility/sdc --n 50
# 2. flag windows and scenes
python -m scripts.responsibility.summarize_responsibility --run logs/responsibility/sdc
```

Step 1 writes `windows.csv` (one row per scene and context step: β_s, β_c,
speed, and the neighbour each maximum came from) and per-neighbour details in
`obs/<scene>.pkl`; it is resumable. Useful flags: `--agent adv` (CAT's
adversary) or a track id, `--stride`, `--n-samples`, `--d-sat`,
`--no-courtesy` (3–10× faster).

Step 2 flags a window when β_s or β_c exceeds a threshold and a scene when it
has a flagged window, and writes `summary/scenes.csv`,
`summary/windows_flagged.csv` and `summary/responsibility.png` (β_s vs β_c).
Thresholds are either given (`--safety-threshold`, `--courtesy-threshold`) or
the `--quantile` (0.9) of the positive values of a **reference** run
(`--reference`, default the run itself). To judge a driving policy, calibrate
on logged driving and judge the policy's run against it:

```bash
python -m scripts.responsibility.summarize_responsibility --run logs/responsibility/policy \
    --reference logs/responsibility/sdc
```

Windows below `--min-speed` (1 m/s) are not judged.

To evaluate a trajectory produced in simulation instead of the log (e.g. the
ego of a CAT-trained policy), replace the agent's track before computing:
`Scene.with_track(agent, positions, headings, velocities)` over the 91 steps.

## Visualisation

```bash
python -m scripts.responsibility.visualize_responsibility --scene 17 \
    --run logs/responsibility/sdc --levels logs/responsibility/levels/levels.csv \
    --out-dir logs/responsibility/video_17
```

One frame per context step t_k (`frames/t_XXX.png`), stitched into
`responsibility.gif` and `responsibility.mp4` (the MP4 needs OpenCV,
`opencv-python-headless`, which `setup_env.sh` installs). Each frame shows:

- **scene:** the map, every agent at t_k with 1 s of history, the agent's
  motion set (DenseTNT goal samples; the first 2 s solid, coloured by goal
  probability), its logged future, and the neighbours it was compared with,
  with their logged futures. The neighbour behind β_s is outlined red, the one
  behind β_c blue. `--ego-heatmap` adds the agent's own goal distribution.
- **courtesy:** that neighbour's goal distribution with the agent in the scene
  and without it, i.e. the two sides of the KL.
- **timeline:** β_s and β_c over the clip with the current step marked; with
  `--levels`, every window's level as background, aggressive levels hatched.

It runs the same code as `compute_responsibility`, and `--run` takes that
run's settings, so the numbers equal the run's `windows.csv`.

### Records: inspect and visualise a run offline

`compute_responsibility --save-records` also writes
`OUT/records/<scene>.pkl` (0.5–1.5 MB per scene), and a live visualisation
saves its `record.pkl`. A record holds the scene itself, plus, for every
context step t_k:

- the agent's motion set (samples [N, 80, 2] and their goal log-probabilities);
- its goal distribution;
- each vehicle neighbour's goal distributions with and without the agent;
- all values, per neighbour included.

Goal distributions keep the most probable goals covering `--top-mass` (0.99)
of the probability, in scene coordinates. Values are identical with or
without records. Records render without DenseTNT, TensorFlow or the scene
files:

```bash
python -m scripts.responsibility.visualize_responsibility \
    --record logs/responsibility/sdc/records/17.pkl --levels logs/responsibility/levels/levels.csv \
    --out-dir logs/responsibility/video_17
```

In Python: `responsibility.records.load_record(path)` returns a dict with
`scene` (`scene_of(record)` rebuilds the `Scene`), `agent`, `agent_id`,
`config` and `frames`. Each frame has `step`, `observation`, `samples`,
`sample_log_prob`, `goals` and `courtesy` {neighbour id: {with, without,
kl}}. In windows without neighbours the metric draws no samples, so the
record keeps a display-only motion set (`metric_samples` False), drawn off
the metric's random stream.

## Responsibility levels

Thresholds say "more than X"; levels say what kinds of behaviour logged
driving contains. `fit_levels` fits a Gaussian HMM (paper Sec. IV-A; catk's
implementation) over the (β_s, β_c) sequences of one or more runs, chooses
the number of levels by BIC, orders them from calmest (0) to most aggressive,
and labels every window with the causal Bayes filter (Eq. 6):

```bash
python -m scripts.responsibility.fit_levels --runs logs/responsibility/sdc logs/responsibility/adv \
    --out-dir logs/responsibility/levels
# judge a policy against levels learned from logged driving
python -m scripts.responsibility.fit_levels --runs logs/responsibility/sdc logs/responsibility/policy \
    --fit-runs logs/responsibility/sdc --out-dir logs/responsibility/levels_policy
```

On 20 logged scenes (self-driving car, 1 s windows) it finds 4 levels:

| level | β_s (m) | β_c (nats) | share | elevated in |
|---|---|---|---|---|
| 0 | 0.00 | 0.02 | 51% | – (calm) |
| 1 | 0.01 | 0.16 | 19% | – (mild courtesy, below half a spread) |
| 2 | 0.44 | 0.12 | 23% | safety: margin given up |
| 3 | 0.05 | 0.80 | 7% | courtesy: others' plans changed |

The levels separate *kinds* of aggressiveness rather than forming one
ranking, so every level that stands out from the calmest one by more than
half a spread (the feature's standard deviation over the fitted windows) in
safety or in courtesy counts as aggressive (here levels 2 and 3;
`--aggressive-levels K` takes the top K instead), and a scene counts when at
least `--scene-share` (50%) of its windows are. Because levels are relative to the fitted population, the
informative output is the comparison between runs: each run's share of
windows per level (printed, and in `scenes_levels.csv`). `--log-courtesy`
fits on log(1 + β_c), which tames courtesy's heavy tail. Outputs: `hmm.pkl`,
`bic.json`, `levels.csv` (every window with its level and posterior),
`scenes_levels.csv`, `levels.png` (scatter by level, BIC curve).

## Responsibility-constrained adversarial generation

CAT picks, among DenseTNT's 32 trajectories for the adversary, the one
maximising P(OV)·P(AV)·P(collision): whichever likely trajectory hits the ego.
Often that is the adversary driving into the ego, a crash the adversary is to
blame for and the ego could not have prevented.
`responsibility/adversarial.py` also scores each candidate j by the
adversary's **safety responsibility toward the ego**: the CVaR, over the
adversary's own DenseTNT motion set, of how much more distance to the ego its
alternatives would have kept than candidate j, over the full 8 s CAT checks,
averaged over the ego trajectories CAT keeps. Low β means the adversary would
plausibly drive this way anyway, so a collision is one the ego has to handle.

| `--adv_selection` | rule |
|---|---|
| `cat` (default) | CAT's original generator, unchanged |
| `constrained` | the highest collision score among candidates with β ≤ `--resp_threshold` (m); if none of them collides, the closest approach among them; if none qualifies, the least responsible candidate |
| `penalized` | argmax collision score · exp(−max(β, 0) / `--resp_penalty`) |
| `fair` | like `constrained`, but a candidate must also be avoidable for the ego: avoidability ≥ `--resp_avoid` (ρ, default 0.3); if none qualifies, the most avoidable candidate within the threshold |

**Ego avoidability** of a candidate is the share of the ego's own DenseTNT
motion set at the generation step that never touches the adversary driving
it (footprints covered by three circles, over the full 8 s). That motion set
is what drivers with the ego's history would do. An avoidability near 0
means a crash whatever a human in the ego's place does: nothing a policy
could learn to avoid. The ego's alternatives do not react to the adversary,
so this underestimates what a reacting driver could avoid.

`cat_advgen.py` and `cat_RLtrain.py` take these options (e.g.
`python cat_RLtrain.py --mode cat --adv_selection fair --resp_threshold 1.0 --resp_avoid 0.3`,
a run named `cat_fair1_0.3`);
the generator is a drop-in subclass of CAT's `AdvGenerator`, and
`cat_advgen.py` prints the collision rate and mean adversary responsibility of
the chosen trajectories at the end. Courtesy responsibility is not used here:
DenseTNT conditions on history only, so the adversary's influence on the ego's
goals is the same for every candidate.

Offline comparison (no MetaDrive), open-loop against the logged ego:

```bash
python -m scripts.responsibility.benchmark_advgen --n 50 --thresholds 0.5 1 2 --avoid 0.1 0.3 0.5 \
    --out logs/responsibility/advgen_benchmark.csv --plot logs/responsibility/advgen_tradeoff.png
```

It reports, per rule:
- how often the chosen trajectory is predicted to collide;
- the chosen adversaries' responsibility and ego avoidability;
- the share of scenes with an *unavoidable* collision (avoidability < 0.1).

It also reports the logged adversaries' own responsibility and avoidability,
to calibrate `--resp_threshold` (τ) and `--resp_avoid` (ρ) on. On the first
20 scenes:

| rule | predicted collision | adversary β mean / median | ego avoidability | unavoidable collisions |
|---|---|---|---|---|
| `cat` | 95% | +3.96 / +3.40 m | 0.29 | 30% |
| `penalized` (1 m) | 95% | +2.81 / +1.59 m | 0.39 | 20% |
| `constrained` 2 m | 50% | +0.98 / +1.00 m | 0.62 | 10% |
| `constrained` 1 m | 30% | +0.51 / +0.56 m | 0.58 | 5% |
| `constrained` 0.5 m | 15% | +0.10 / +0.06 m | 0.79 | 0% |
| `fair` 2 m, ρ 0.3 | 40% | +0.74 / +0.83 m | 0.81 | 0% |
| `fair` 1 m, ρ 0.3 | 25% | +0.34 / +0.24 m | 0.81 | 0% |
| `fair` 1 m, ρ 0.5 | 15% | +0.23 / +0.06 m | 0.89 | 0% |
| logged adversaries | – | median 0.00, q90 +0.35 m | median 0.95, q10 0.61, min 0.28 | – |

CAT's adversaries give up metres of margin their own alternatives would have
kept, far beyond anything the logged adversaries do. In almost a third of the
scenes they produce a crash that no plausible ego motion escapes, while every
logged adversary leaves the ego a way out (avoidability at least 0.28).
Bounding the adversary's responsibility removes most of those crashes, and
`fair` removes all of them. Each constraint costs attack success; the
collisions that remain are the ones the ego can and has to handle.

`--plot tradeoff.png` draws this trade-off: predicted collision rate
against the chosen adversaries' mean β, and against the share of
unavoidable collisions, one curve over τ for `constrained` and for `fair` at
each ρ. The summary ends with the `fair` settings recommended for RL. A
setting must keep a predicted collision rate of `--min-collision` (30%);
among those, the ones with the fewest unavoidable collisions come first,
then the ones whose adversary is least responsible. On the first 20 scenes
(τ ∈ {0.25, 0.5, 1, 2} m, ρ ∈ {0.1, 0.3, 0.5}), `fair` has no unavoidable
collision anywhere. Only τ = 2 m keeps 30% (ρ 0.5: 30%, ρ 0.3 and 0.1:
40%). At the logged adversaries' q90 (τ ≈ 0.35 m), every rule stays at
10–15%. A fair adversary that is as responsible as logged drivers rarely
produces a collision against the logged ego, so the attack rate and a
realistic τ have to be traded off. The 500-scene run decides.
It also checks that the `cat` rule reproduces `AdvGenerator.generate` exactly
(CAT's own code run on the same candidates), and it runs the drop-in
generator the way CAT's scripts call it. Candidates are the adversary
predicted on its own; CAT batches it with the ego, which moves the prediction
slightly (see Design decisions), so every rule chooses among the same
candidates.

## Driving policies (rollouts)

The same measurements apply to a policy driving in MetaDrive. A **rollout**
(`responsibility/rollouts.py`; recorded by `collect_rollouts.py` on the
server) holds what happened in one scene: the ego's simulated states per
0.1 s step, the adversary's if CAT placed one, which logged objects the
simulator spawned, and how the episode ended. `--rollouts DIR` plays each
rollout back into its scene and queries the simulated ego:

```bash
python -m scripts.responsibility.compute_responsibility --rollouts rollouts/td3_cat/none \
    --out-dir logs/responsibility/policies/td3_cat/none
```

In the rebuilt scene the ego (and the adversary) follow the simulation and
end with the episode; objects the simulator never spawned are removed
(CAT runs MetaDrive with `no_static_vehicles`, which drops every vehicle
whose logged positions spread less than 3 m); everything else keeps its
logged states, since CAT's traffic is not reactive. Windows run up to a
collision (the window must see it), or stop a full metric horizon before
any other end of the episode (what the ego would have done afterwards is
unknown). Outputs are those of a logged run, so summaries, levels and
records work unchanged.

Removing parked vehicles changes the values (on scene 1, with 70 of them,
safety moves by 0.4 m on average), so compare a policy against **the
replayed log** (`--policy replay` rollouts), not against the plain logged
run: that is the logged driving in the scene the policy actually saw.

### Recording rollouts (on the server, needs MetaDrive)

```bash
# the reference: the logged ego replayed, without and with CAT's adversary
python -m scripts.responsibility.collect_rollouts --policy replay --adversary --out_dir rollouts
# a TD3 policy saved by cat_RLtrain.py --save_model (models/<mode>_s<seed>)
python -m scripts.responsibility.collect_rollouts --policy models/cat_s0 --policy_name td3_cat_s0 \
    --adversary --adv_selection cat --out_dir rollouts
```

Each scene (default: CAT's test split, 400–499) is played the way CAT's
`eval_policy` does it: a normal episode, then, with `--adversary`, an
adversary generated against the ego's own trajectory from that episode
(`--adv_selection` as for `cat_RLtrain.py`) and a second episode with it.
Rollouts go to `rollouts/<policy_name>/none/` and `.../<mode>/`. Two checks
are printed: with `--policy replay`, the recorded ego against the logged one
(should stay below 0.1 m, which also confirms that state i is the log's step
i), and in adversarial episodes how far the adversary left its logged track
and whether it follows its plan with a lag of 0 or 1 step. The recording
logic is tested against a stand-in environment (`tests/responsibility/test_collect.py`).

### Comparing policies: aggressive and timid, and whose fault the collisions were

Responsibility has two directions. β_s > 0 is aggressive: the agent kept
less margin than its own alternatives. β_s < 0 is the opposite: it kept more
margin to everyone than its alternatives would have. A little of that is
ordinary caution; far more than logged drivers ever keep is **timid**, the
failure mode of a policy trained to avoid collisions at any cost.
`summarize_responsibility` now flags both: aggressive above the
`--quantile` (0.9) of the reference's positive values, timid below the
`--timid-quantile` (0.1) of its negative safety values.

Every vehicle collision of a rollout is also **attributed**
(`responsibility/blame.py`, written to `crashes.csv` by
`compute_responsibility --rollouts`). In the 2 s window that ends with the
collision it measures both sides: β_ego, the ego's safety responsibility
toward the other, and β_other, the other's toward the ego. Each comes from
that agent's own DenseTNT motion set. A car that was rear-ended did what its
alternatives do (β ≈ 0), while the follower closed in far more than its
alternatives (β ≫ 0). The collision counts as the ego's fault when its
positive part exceeds the other's by more than 0.1 m, the other's in the
reverse case, and shared otherwise; the ego's share is
w = β_ego⁺ / (β_ego⁺ + β_other⁺).

As a baseline, every attribution also carries the verdict of the **rear-end
rule**, the traffic-law reading of the most common collision: the follower
is at fault. It applies when both travel the same way (headings within 30°)
and the other lies off the ego's front or back rather than its side, and
says nothing otherwise (`crashes.csv`, column `rule`).

```bash
python -m scripts.responsibility.fit_levels --runs P/replay/none P/td3_cat_s0/none P/td3_cat_s0/cat \
    --fit-runs P/replay/none --out-dir P/levels            # P=logs/responsibility/policies
python -m scripts.responsibility.compare_policies --runs P/replay/none P/td3_cat_s0/none P/td3_cat_s0/cat \
    --hmm P/levels/hmm.pkl --out-dir P/compare
```

`compare_policies` writes one row per run (policy × adversary mode) to
`comparison.csv`/`.md`, with thresholds from the first run (or
`--reference`), the replayed log. Columns:

- from the rollouts: crash rate, route completion, arrival and out-of-road rates;
- the ego-fault and other-fault shares of the attributed collisions, and
  their agreement with the rear-end rule where both decide (with the rule's
  coverage in `comparison.csv`);
- the share of windows the ego was stopped (below `--min-speed`, not judged);
- the aggressive and timid shares of the judged windows, and each relative
  to the reference ("× ref");
- the median β_s and the p90 of β_c;
- with `--hmm`, the share of windows in each responsibility level.

`comparison.png` shows β_s per run and the level (or aggressive/timid) shares.

Runs of one training setting with different seeds (policy names ending in
`_s<seed>`, as `cat_RLtrain.py` names its models) are averaged in
`comparison_seeds.csv`/`.md`: mean ± standard deviation of every column per
training setting and test adversary. The markdown adds matrices of training
setting × test adversary for the crash rate, the ego-fault share, route
completion and the timid and aggressive shares: the cross evaluation of
policies trained against different adversaries (and penalties), each tested
without an adversary, with CAT's and with the fair one.

### Training with a responsibility-weighted collision penalty

In CAT's training environment a collision does not end the episode: at every
step the ego touches another vehicle, MetaDrive replaces that step's driving
reward with −`crash_vehicle_penalty` (1). An adversary that drives into the
ego costs it the same as a collision the ego caused, which teaches it to
avoid driving on rather than to drive well. With `--blame_weighting share`,
`cat_RLtrain.py` keeps only the ego's share of each penalty
(`responsibility/blame_reward.py`):

- a **collision** is a run of consecutive steps in contact with the same
  vehicle, attributed once at its first step, in the scene rebuilt from the
  training episode up to then (the same attribution as above);
- the **weight** is the ego's share w when the verdict is "ego" or "other".
  Otherwise the full penalty is kept (w = 1): when the sides differ by less
  than `--blame_margin` (0.1 m), after the end of the log, when the partner
  is not a predicted vehicle, or when the attribution fails. Doubt keeps the
  penalty, so noise in the attribution does not become noise in the reward;
- at every penalised step of the collision the reward becomes
  w · (−penalty) + (1 − w) · (the driving reward the penalty replaced). w = 1
  is MetaDrive's reward, w = 0 the step as if nothing had been hit.

An episode's transitions go into the replay buffer when it ends, already
weighted. Every attributed collision is logged to
`logs/blame/<run>_s<seed>.csv` (β of both sides, verdict, rule, weight, time),
and the running mean weight is printed at every evaluation. It works with any
adversary (`--mode replay` too). Runs are named with a `_share` suffix
(`cat_share`, `cat_fair1_0.3_share`). An attribution costs two DenseTNT
passes, once per collision; the constrained and fair generators share
their DenseTNT with it.

```bash
python cat_RLtrain.py --mode cat --blame_weighting share --seed 0 --save_model          # cat_share_s0
python cat_RLtrain.py --mode cat --adv_selection fair --resp_threshold 1 --resp_avoid 0.3     --blame_weighting share --seed 0 --save_model                                         # cat_fair1_0.3_share_s0
```

```bash
python -m scripts.responsibility.summarize_blame --logs logs/blame/*.csv --out logs/blame/summary.md
```

summarises those logs per run. It reports the number of collisions, the
share of each verdict and the mean weight. It also shows how often the
full penalty was kept and how often the other was mostly at fault
(w < 0.5), plus the agreement with the rear-end rule, the time per
attribution, and the mean weight over spans of training.

The weighting is tested against a stand-in for the training env
(`tests/responsibility/test_blame_reward.py`: a rear-ended ego keeps none of
the penalty; doubt and failed attributions keep all of it).

## GPU server (e.g. Ubuntu 24.04 + H200)

The full sequence of experiments on the server, step by step, is in
[RUNBOOK_H200.md](RUNBOOK_H200.md) (in Chinese).

| component | version | why |
|---|---|---|
| Python | **3.9** | `advgen/utils_cython.cpython-39-*.so` is prebuilt for it (rebuilding needs Cython and a compiler); TF 2.12 supports it |
| torch | **2.4.1 + CUDA 12.1** (not CAT's 1.12.0+cu116) | 1.12/cu116 has no Hopper (sm_90) kernels; 2.4.1 still has Python 3.9 wheels and loads `densetnt.bin` unchanged |
| torchvision | 0.19.1 (matches torch 2.4.1) | DenseTNT's raster CNN |
| tensorflow-cpu | 2.12.0 | only builds DenseTNT's input tensors; the CPU build stays off the GPU |
| numpy | < 1.24 | required by TF 2.12 |
| opencv-python-headless | 4.7.0.72 (CAT's opencv version, without GUI) | MP4 output of the visualisation; without it only the GIF is written |

```bash
bash scripts/responsibility/setup_env.sh          # uv venv at ~/venvs/cat39 (BACKEND=conda also works)
source ~/venvs/cat39/bin/activate
python -m pytest tests/responsibility -q
python -m scripts.responsibility.verify_densetnt --n 3 --device cuda   # must end with ALL CHECKS PASSED
bash scripts/responsibility/run_h200.sh           # step 2: sdc + adv over 500 scenes, summaries, levels
```

To run on a newer Python instead, rebuild advgen's Cython extension for it
(needs a C compiler; it overwrites `advgen/utils_cython*.so` for that Python
only) and re-run `verify_densetnt`. A rebuild from `utils_cython.pyx` with
Cython 0.29 passes every check:

```bash
pip install "cython>=0.29.34,<3" && python scripts/responsibility/build_cython.py
```

`run_h200.sh` starts `SHARDS` (16) processes per agent, spread over the
visible GPUs, each taking every SHARDS-th scene (`--num-shards/--shard-index`)
and writing its own `windows.shard-<i>-of-<n>.csv`; logs are in
`OUT/<agent>/logs/`. Building DenseTNT's inputs is CPU work, so many
processes share one GPU and CPU threads are divided between them. Re-running
the same command resumes. On a laptop CPU a scene takes ~1 min per agent at
1 s windows; check the first lines of a shard log for the rate on the server
and raise `SHARDS` if the GPU and CPUs are not busy. Variables: `OUT`,
`SCENES`, `N`, `AGENTS`, `SHARDS`, `STRIDE` (0.5 s), `SAMPLES`.

## Design decisions

- **Motion set = goal samples, not NMS modes.** DenseTNT scores a 224×224
  grid of goals with a softmax; CAT keeps the 32 goals left by non-maximum
  suppression, a mode-seeking selection. Responsibility needs draws from the
  model's distribution, so goals are sampled i.i.d. from the softmax and
  completed with DenseTNT's own completion head.
- **Courtesy is an exact KL over goals.** The goal grid depends only on the
  predicted agent's pose, so b's distributions with and without the agent share
  their support; the KL is computed over the grid, no sampling. It measures
  intent (where b means to go), not how b's trajectory to a given goal shifts.
- **One instance per forward pass.** In a batch, advgen pads all instances to
  the longest and its attention mask (`attention_mask[i][:n][:n]`) limits only
  the rows, so shorter instances attend to padding: CAT's own ego+adversary
  batch moves the adversary's prediction by 1–2 cm in most scenes and by 32 m
  in one (a different NMS pick). Encoding alone removes this.
- **Removal holds everything else fixed.** DenseTNT's pair model needs a second
  object of interest; it is chosen the same in both passes and never the
  removed agent. Agents are listed nearest first, since the input holds 128
  (in a 156-agent scene the scene order had dropped the ego from its
  neighbour's input).
- **Neighbours by interaction evidence** (footprint gap ≤ 10 m, post-encroachment
  time ≤ 2 s, constant-velocity TTC ≤ 4 s over the metric horizon, within
  50 m), as in catk, not a plain radius.
- **Courtesy toward vehicles only.** CAT's DenseTNT predicts vehicles
  (`agent_type='vehicle'`); safety is measured toward every neighbour.
- **D_g saturates at 10 m** and CVaR uses the upper-tail convention with
  α = 0.1 (close to the mean), as in catk (`responsibility/risk.py`).

## Limitations

- At 1–2 s horizons the closest approach is often at the first step, where
  every sample starts where the log does; slow or stationary agents therefore
  get β_s ≈ 0, which says "no alternative did better", not "safe".
- Courtesy is intent-level and DenseTNT only sees 1.1 s of history, so an
  agent's influence enters only through its recent past and current state.
- Thresholds are relative to a reference population; the verdict is "unusual
  compared with that driving", not an absolute safety judgement.
