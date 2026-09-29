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
uv pip install "tensorflow-cpu==2.12.0" "numpy<1.24" pyyaml matplotlib tqdm pytest scipy
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
distribution more than a far one, and that DenseTNT's partner slot is
immaterial (~1e-7 nats).

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

## GPU server (e.g. Ubuntu 24.04 + H200)

| component | version | why |
|---|---|---|
| Python | **3.9** | `advgen/utils_cython.cpython-39-*.so` is prebuilt for it (rebuilding needs Cython and a compiler); TF 2.12 supports it |
| torch | **2.4.1 + CUDA 12.1** (not CAT's 1.12.0+cu116) | 1.12/cu116 has no Hopper (sm_90) kernels; 2.4.1 still has Python 3.9 wheels and loads `densetnt.bin` unchanged |
| torchvision | 0.19.1 (matches torch 2.4.1) | DenseTNT's raster CNN |
| tensorflow-cpu | 2.12.0 | only builds DenseTNT's input tensors; the CPU build stays off the GPU |
| numpy | < 1.24 | required by TF 2.12 |

```bash
bash scripts/responsibility/setup_env.sh          # uv venv at ~/venvs/cat39 (BACKEND=conda also works)
source ~/venvs/cat39/bin/activate
python -m pytest tests/responsibility -q
python -m scripts.responsibility.verify_densetnt --n 3 --device cuda   # must end with ALL CHECKS PASSED
bash scripts/responsibility/run_h200.sh           # step 2: sdc + adv over 500 scenes, summaries, levels
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
