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

## Command reference

Run from the repository root with the environment active. The sections below
explain each step; this is the whole pipeline in one place, for the
simplified evaluation of one pair per scene (`--use-ooi`: the self-driving
car and CAT's adversary) with the NMS motion set, the most accurate 40-trajectory set
measured (see "Valid counterfactuals" under Design decisions).

**Compute.** `run_h200.sh` runs every agent in shards, then the summaries and
the levels, into `OUT/<agent>/`:

```bash
OUT=logs/responsibility/ooi_nms AGENTS="sdc adv" SHARDS=8 RECORDS=1 \
EXTRA="--use-ooi --motion-set nms --valid-counterfactuals" \
bash scripts/responsibility/run_h200.sh
```

The same for one agent by hand (`--agent adv` swaps the roles of the pair):

```bash
python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 \
    --out-dir logs/responsibility/ooi_nms/sdc --agent sdc \
    --use-ooi --motion-set nms --valid-counterfactuals --save-records --device cuda
```

- Shards: `--num-shards N --shard-index i` for i = 0 … N−1, all into the
  same `--out-dir`. Each shard writes `windows.shard-<i>-of-<N>.csv`, and
  every reader takes both forms.
- A subset: `--n 30 --first 0`. Leaving out β_c is 3–10× faster:
  `--no-courtesy`.
- `--valid-counterfactuals` is `--lane-route`, `--drivable-half-width 3`,
  `--kinematics`, `--collision-filter` and `--courtesy-valid-goals`.
- On a map built by perception (no lane topology), use
  `--intent --drivable-edges --kinematics --collision-filter` instead.
- To take β_c only over the neighbour's own logged drive mode, add
  `--courtesy-same-mode lanes` (HD map) or `--courtesy-same-mode path`
  (no map needed; 6 m of its logged path). This overrides
  `--courtesy-valid-goals`.
- Motion sets: `sampled` (default), `weighted` (the reference; a median of
  944 trajectories per window), `topk`, `nms`.
- MTR instead of DenseTNT: `--model mtr --checkpoint <ckpt>`.
- A driving policy instead of the log:
  `--rollouts rollouts/<policy>/<adversary>`.
- An output directory resumes. Its `config.json` pins the settings, and a
  run with other settings is refused.

**Summarise and fit levels** (`run_h200.sh` does both):

```bash
python -m scripts.responsibility.summarize_responsibility --run logs/responsibility/ooi_nms/sdc
python -m scripts.responsibility.fit_levels \
    --runs logs/responsibility/ooi_nms/sdc logs/responsibility/ooi_nms/adv \
    --out-dir logs/responsibility/ooi_nms/levels
```

**Visualise.**

- Offline, from records, with no model needed:

  ```bash
  python -m scripts.responsibility.visualize_responsibility \
      --record logs/responsibility/ooi_nms/sdc/records/{17,23,42}.pkl \
      --levels logs/responsibility/ooi_nms/levels/levels.csv \
      --out-dir logs/responsibility/ooi_nms/videos_sdc
  ```

- Live, with the model loaded once for every scene and both roles:

  ```bash
  python -m scripts.responsibility.visualize_responsibility --scene 17 23 42 \
      --agent sdc adv --use-ooi --motion-set nms --device cuda \
      --out-dir logs/responsibility/videos_ooi
  ```

- Live, with a run's settings, so the values equal its `windows.csv`:

  ```bash
  python -m scripts.responsibility.visualize_responsibility --scene 17 23 \
      --run logs/responsibility/ooi_nms/sdc --agent sdc --use-ooi \
      --levels logs/responsibility/ooi_nms/levels/levels.csv \
      --out-dir logs/responsibility/video_17_23
  ```

One video goes to `--out-dir`. Several go to `--out-dir/<scene>`, or to
`--out-dir/<scene>_<agent>` with several agents. With `--run`, the motion
set, the filters and `use_ooi` come from the run's `config.json`, and
`--use-ooi` is refused for a run made without it. A DenseTNT process takes a
few hundred MB of GPU memory.

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
`--no-courtesy` (3–10× faster), `--use-ooi` (below).

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

`--scene` and `--record` take several values (the model is loaded once),
and `--agent sdc adv` renders each scene for both objects of interest; see
the command reference above. One frame per context step t_k (`frames/t_XXX.png`), stitched into
`responsibility.gif` and `responsibility.mp4` (the MP4 needs OpenCV,
`opencv-python-headless`, which `setup_env.sh` installs). Each frame shows:

- **scene:** the map, every agent at t_k with 1 s of history, the agent's
  motion set exactly as β_s scored it (DenseTNT goal samples, or with
  `--motion-set weighted|topk|nms` the weighted set; the first 2 s solid,
  coloured by goal probability or weight), its logged future, and the neighbours it was compared with,
  with their logged futures. The neighbour behind β_s is outlined red, the one
  behind β_c blue. `--ego-heatmap` adds the agent's own goal distribution.
- **courtesy:** that neighbour's goal distribution with the agent in the scene
  and without it, i.e. the two sides of the KL.
- **timeline:** β_s and β_c over the clip with the current step marked; with
  `--levels`, every window's level as background, aggressive levels hatched.

It runs the same code as `compute_responsibility`, and `--run` takes that
run's settings, so the numbers equal the run's `windows.csv`. Without `--run`,
live mode takes `--motion-set`, `--n-samples` and `--use-ooi` itself; with
`--run`, `--use-ooi` is refused unless the run used it.

### Records: inspect and visualise a run offline

`compute_responsibility --save-records` also writes
`OUT/records/<scene>.pkl` (0.5–1.5 MB per scene), and a live visualisation
saves its `record.pkl`. A record holds the scene itself, plus, for every
context step t_k:

- the agent's motion set (samples [N, 80, 2] and their goal log-probabilities,
  or for a weighted set the log of each trajectory's weight);
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

On all 500 scenes (the self-driving car and CAT's adversary as logged, 0.5 s
windows, fitted together) BIC chooses 7 levels:

| level | β_s (m) | β_c (nats) | share sdc / adv | elevated in |
|---|---|---|---|---|
| 0 | 0.00 | 0.01 | 36% / 38% | – (calm) |
| 1 | 0.00 | 0.02 | 1% / 2% | – |
| 2 | 0.01 | 0.09 | 18% / 18% | – |
| 3 | 0.01 | 0.27 | 16% / 14% | courtesy |
| 4 | 0.44 | 0.08 | 15% / 19% | safety: margin given up |
| 5 | 0.01 | 0.82 | 8% / 5% | courtesy: others' plans changed |
| 6 | 0.39 | 0.59 | 7% / 5% | both |

The levels separate *kinds* of aggressiveness rather than forming one
ranking, so every level that stands out from the calmest one by more than
half a spread (the feature's standard deviation over the fitted windows) in
safety or in courtesy counts as aggressive (here levels 3 to 6;
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
to calibrate `--resp_threshold` (τ) and `--resp_avoid` (ρ) on. On all 500
scenes (τ = ∞ keeps every candidate, so `fair` at τ = ∞ constrains
avoidability alone: the ablation of β):

| rule | predicted collision | adversary β mean / median | ego avoidability | unavoidable collisions |
|---|---|---|---|---|
| `cat` | 95% | +5.00 / +5.14 m | 0.30 | 40% |
| `penalized` (1 m) | 95% | +3.98 / +3.96 m | 0.39 | 26% |
| `constrained` 2 m | 33% | +1.01 / +1.18 m | 0.65 | 14% |
| `constrained` 1 m | 24% | +0.32 / +0.29 m | 0.71 | 10% |
| `constrained` 0.5 m | 18% | +0.01 / +0.00 m | 0.77 | 6% |
| `fair` 2 m, ρ 0.1 | 25% | +0.87 / +1.05 m | 0.73 | 0.4% |
| `fair` 2 m, ρ 0.3 | 21% | +0.78 / +0.96 m | 0.79 | 0.4% |
| `fair` 1 m, ρ 0.3 | 15% | +0.17 / +0.12 m | 0.82 | 0.4% |
| `fair` ∞, ρ 0.3 (avoidability only) | 63% | +3.86 / +3.55 m | 0.62 | 0.4% |
| `fair` ∞, ρ 0.5 (avoidability only) | 49% | +3.39 / +3.03 m | 0.75 | 0.4% |
| logged adversaries | – | median −0.05, q90 +0.23 m | median 0.97, q10 0.52 | 2% (avoidability < 0.1) |

CAT's adversaries give up metres of margin their own alternatives would have
kept: a mean of 5 m, against a q90 of 0.23 m for the logged adversaries.
In 40% of the scenes they produce a crash that no plausible ego motion
escapes, against 2% of the logged adversaries. Bounding the adversary's
responsibility removes most of those crashes, and `fair` all but 2 scenes
(0.4%: no candidate there is avoidable enough). Each constraint costs attack
success; the collisions that remain are the ones the ego can and has to
handle.

`--plot tradeoff.png` draws this trade-off: predicted collision rate
against the chosen adversaries' mean β, and against the share of
unavoidable collisions, one curve over τ for `constrained` and for `fair` at
each ρ. The summary ends with the `fair` settings recommended for RL. A
setting must keep a predicted collision rate of `--min-collision` (30%);
among those, the ones with the fewest unavoidable collisions come first,
then the ones whose adversary is least responsible.

On 500 scenes no finite τ up to 2 m reaches 30%. `fair` peaks at 25% (τ 2 m,
ρ 0.1), and at the logged adversaries' q90 (τ ≈ 0.25 m) every rule stays at
8–16%. A fair adversary that is as responsible as logged drivers rarely
produces a collision against the logged ego. Avoidability alone (τ = ∞)
keeps 49–78% collisions with almost no unavoidable ones, so the
recommendation falls on it, but those adversaries are about as
irresponsible as CAT's (β +3.4 to +4.4 m). Avoidability makes a collision
solvable; β makes the adversary drive like people do. Which of the two
training needs is what the RL experiments have to show.
It also checks that the `cat` rule reproduces `AdvGenerator.generate` exactly
(CAT's own code run on the same candidates), and it runs the drop-in
generator the way CAT's scripts call it. Candidates are the adversary
predicted on its own; CAT batches it with the ego, which moves the prediction
slightly (see Design decisions), so every rule chooses among the same
candidates.

### Adversarial scenes as files

`scripts/responsibility/export_adv_scenes.py` writes the adversarial
counterpart of every scene. It runs CAT's generator against the logged ego,
as `cat_advgen.py`'s second round does, but without MetaDrive and keeping
the result as data.

- **What changes.** Each file is a copy of the original scene in which only
  the adversary's track from step 11 on is replaced by the generated plan:
  position, heading, velocity and validity. Steps 0–10 are the history CAT
  plans from.
- **Where it goes.** The files are `OUT/<rule>/<scene>.pkl`, named like
  `raw_scenes_500`, so `Scene.load`, `compute_responsibility.py`,
  `visualize_responsibility.py --scene` and MetaDrive read them as they are.
- **What is recorded.** `metadata.adversary` and `OUT/<rule>/index.json`
  hold the rule, the chosen candidate, its predicted collision score, the
  adversary's β and the first step at which the plan overlaps the logged
  ego.

With `--rule cat` (500 scenes, `adv_scenes/cat`, about 1 s a scene on the
GPU), the plan overlaps the logged ego in 476 scenes (95%). The first overlap
comes at a median step of 53, and the adversary's median β is +5.14 m, as
in the benchmark above.

```bash
python -m scripts.responsibility.export_adv_scenes --out-dir adv_scenes                     # CAT's rule
python -m scripts.responsibility.export_adv_scenes --rule fair --resp_threshold 2 --resp_avoid 0.1 --out-dir adv_scenes
python -m scripts.responsibility.visualize_responsibility --scene adv_scenes/cat/0.pkl --agent adv \
    --out-dir adv_scenes/videos/cat/0_adv
```

**Offline and in MetaDrive.** The export imports no MetaDrive module. It
does differ from CAT's generation in MetaDrive in one input, the vehicle
sizes CAT uses to decide which candidates hit the ego:

- **Offline:** the sizes logged at step 10.
- **In MetaDrive:** the ego is always the default vehicle (4.51×1.85 m). An
  adversary 4–5.5 m long gets one of three vehicle models in turn, through a
  counter over the whole session (`even_sample_vehicle_class`), so its size
  depends on what was spawned before it.

`scripts/responsibility/verify_adv_export.py` runs `cat_advgen.py`'s two
rounds in MetaDrive and compares its plans with the exported ones. The check
covered scenes 0–2 and every third test scene from 400 to 487, 33 in all:

| sizes used offline | same plan as MetaDrive |
|---|---|
| the logged sizes (what the export uses) | 27 of 33 |
| MetaDrive's ego, the logged adversary | 29 of 33 |
| the sizes MetaDrive used, for both | 32 of 33 |

The sizes are therefore the only real difference. The remaining scene (451)
matched with the logged sizes, which suggests two candidates scored almost
equally. MetaDrive's adversary size cannot be known from the scene alone, so
the export keeps the logged sizes.

In these scenes the ego is the logged one and does not react to an
adversary that was not there. Its own β_s toward the adversary is therefore
often positive as well: in scene 0 it reaches 1.2 m at 3.5 s, because some
of its alternatives would have kept more distance.

### Swapping the ego and the adversary

CAT takes the ego from `metadata.sdc_id` and the adversary from the other
object of interest. MetaDrive spawns its ego vehicle, its route and the
replay policy from the same track, and steers only replayed traffic as an
adversary. So the roles are swapped in the scene data, not in advgen.
`swap_roles` writes a copy of the scenes in which the other object of
interest is the self-driving car (`sdc_id`, `sdc_track_index`) and the
logged self-driving car is the adversary (`responsibility/swap.py`):

```bash
python -m scripts.responsibility.swap_roles --scenes raw_scenes_500 --out-dir raw_scenes_500_swapped
python cat_advgen.py --scenes_dir raw_scenes_500_swapped                        # CAT's benchmark, roles swapped
python cat_RLtrain.py --mode cat --scenes_dir raw_scenes_500_swapped --seed 0  # run name cat_swapped_...
python -m scripts.responsibility.export_adv_scenes --scenes raw_scenes_500_swapped --out-dir adv_scenes_swapped
python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500_swapped --use-ooi \
    --out-dir logs/responsibility/swapped/sdc
```

- **Eligibility.** A scene is swapped only if the other object of interest
  is a vehicle valid at every step, because MetaDrive spawns the ego at
  step 0 and drives its whole logged route. 459 of the 500 scenes qualify.
  In 24 the other object is missing at step 0, and in 17 its track has
  gaps.
- **Layout.** The swapped scenes keep their file names. The folder holds
  only scenes, because MetaDrive asserts that every file in it is one; the
  index of swapped and skipped scenes is
  `raw_scenes_500_swapped.index.json`, next to it.
- **Train/test split.** `cat_RLtrain.py --scenes_dir` counts CAT's split
  over the files present: scenes 0–399 train, 400 on test. That is 369 / 90
  for the swapped folder. The evaluation runs once over the test scenes.
- **Run names.** A swapped folder adds `_swapped` to the run name.
- **Other scripts.** `cat_advgen.py --scenes_dir` runs over every scene in
  the folder. The other scripts read any scene folder with `--scenes`.

**Checked in MetaDrive** on this branch. On swapped scenes 0–2, the ego
follows the new self-driving car's log exactly (0.00 m at step 20), and the
original one is replayed traffic. In CAT's two-round generation on scenes
0–5, every adversary is the original self-driving car. The attack hits
4 of 6 swapped scenes and 6 of 6 original ones, too few scenes to compare
rates. A 300-step `cat_RLtrain.py` run on the swapped folder ran as
`cat_swapped_MDWaymo-seed99`.

### Near misses instead of collisions

`--adv_selection near` keeps CAT's generation, the same 32 DenseTNT
candidates for the adversary, but chooses a near miss instead of a
collision. For each candidate, gap_j is the smallest gap between the
adversary's and the ego's footprints over the 8 s CAT plans. Each footprint
is three circles, checked every 0.1 s, and gap_j is averaged over the ego
trajectories by P(AV_i). The rule then works as follows:

1. Drop every candidate that touches the ego, meaning CAT's own test
   predicts a collision, or the gap to any ego trajectory is ≤ 0.
2. Of the rest, take the most probable candidate whose gap is within
   `--near_gap ± --near_tol` (m). Defaults: 1.0 ± 0.5.
3. If no candidate is in that band, take the one closest to `--near_gap`.
4. If every candidate collides, take the one with the largest gap.

The circles are slightly larger than the car (0.28 m at the side of a
4.8 × 2 m car), so gap_j understates the true gap a little.

```bash
python cat_advgen.py --adv_selection near --near_gap 1.0 --near_tol 0.5
python cat_RLtrain.py --mode cat --adv_selection near --near_gap 1.0 --seed 0     # run name cat_near1_0.5_...
python -m scripts.responsibility.export_adv_scenes --rule near --near_gap 1.0 --out-dir adv_scenes
```

**Measured on the first 30 scenes.** The adversary is planned against the
logged ego, as in `export_adv_scenes`. Its closest approach is then
measured against the logged ego, and the scenes are replayed in MetaDrive
(logged ego, generated adversary):

| rule | overlap | closest approach P10 / median / P90 | within target ± 0.5 m | ego collisions in MetaDrive | adversary β_s (median) |
|---|---|---|---|---|---|
| cat | 100% | −2.64 / −1.42 / −0.73 m | – | 23 / 30 | 3.40 m |
| near 0.5 m | 0% | 0.18 / 0.61 / 1.45 m | 70% | 0 / 30 | 0.86 m |
| near 1.0 m | 0% | 0.57 / 1.20 / 1.62 m | 77% | 0 / 30 | 0.34 m |
| near 2.0 m | 0% | 0.83 / 1.84 / 2.27 m | 73% | 0 / 30 | −0.01 m |

- **Outside the target band.** In 23–30% of the scenes no candidate falls
  within the band. The closest one is taken, which accounts for the tails
  of the gap distribution.
- **Responsibility.** The farther the miss, the less responsible the
  adversary: its safety responsibility goes from 3.40 m for CAT's
  collisions to about 0 at 2 m.
- **In training.** The ego is the policy, not the log, and it reacts. The
  realised gap is that of the policy's own trajectory, against which the
  candidates were not scored: CAT scores them against the ego's past
  rollouts.
- **Exported folders and MetaDrive.** `export_adv_scenes` keeps
  `index.json` inside the rule folder. MetaDrive asserts that every file in
  a scene folder is a `.pkl` file, so copy the `.pkl` files out before
  loading the folder there. That is how this table was measured.

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

**Collisions of a replayed ego.** MetaDrive's agent manager moves a
replay policy's vehicle to its next logged pose after the collision check,
and the vehicle's `before_step` then clears that step's flags. So
`crash_vehicle` never reaches `info`, and `crash_vehicle_done` never ends
the episode. `collect_rollouts` therefore reads collisions from
`ego_crash_flag`, which is set by the same check and never cleared; CAT's
`cat_advgen.py` counts attacks by it for the same reason. The episode then
ends at the collision, as it would for a driven ego. Before this fix the
replay reference had no collision at all, although ego and adversary boxes
overlapped in 86 of CAT's 100 test episodes.

**A fix to CAT's adversary replay.** Once CAT's traffic manager has spawned
the adversary, it applies the plan one row per step from its first row.
That only lines up for an adversary present from step 0, which gets row
k − 1 at step k. In 24 of the 500 scenes (8 of the test split) the
adversary appears later in the log, at steps 1–10. The plan's first rows
are zero padding for the steps before it existed, so the adversary sat at
the origin for as many steps and then drove its plan as many steps late. A
gap in its logged history did the same for one step. Both generators now
hand over a `StepAlignedPlan` (`advgen/adv_generator.py`), which applies
the row of the current step and holds the adversary at its nearest logged
state where the log has none. `AdvGenerator.before_episode` also clears
the manager's adversary: a plan left over from an earlier episode used to
drive any later vehicle of the same name, for example in
`eval_policy`'s normal episodes. For an adversary present from step 0
nothing changes. With the fix, every adversary follows its plan exactly one
step late (lag-1 error 0.00 m).

`before_episode` runs only after `env.reset()`, though, and the reset already
steps the traffic manager once with the previous episode's adversary. A
`StepAlignedPlan` is never emptied, unlike CAT's popped list, so in that
first step it moved the next scenario's object of the same name. A same-named
cyclist, which is not in the manager's `v_map`, raised a KeyError that
stopped `cat_share_s1` at 483k steps on 2026-10-07. The plan now has length 0
once the env has moved to another scenario, so the manager no longer applies
it there.

Measured over training scenes 0–399, the adversary's name exists in another
scene 2.9% of the time, which is about 240 one-frame moves per training run.
Only 0.001% of these put the old adversary within 5 m of the new ego's start
(about 0.05 per run). One run crashed on a cyclist. The other 16 runs are
kept.

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

The formal rule-based baseline is **RSS** (Responsibility-Sensitive Safety,
Shalev-Shwartz et al. 2017; `responsibility/rss.py`, columns `rss` and
`rss_case`). It finds the danger threshold: the step from which the two
cars stayed closer than RSS's safe distance, both longitudinally and
laterally. From there, both owe the proper response of the axis that became
unsafe last:

- longitudinal: the rear car brakes at least 4 m/s² after a 1 s response
  time, and the front car brakes at most 8 m/s²;
- lateral: both stop closing in laterally.

A car whose speed left that envelope (beyond 0.5 m/s longitudinally,
0.2 m/s laterally) is responsible. A brake-check is therefore the front
car's fault, where the rear-end rule would blame the follower. The other
parameters are ad-rss-lib's defaults.

Lane directions come from the map's nearest lane centreline. RSS covers
same-direction collisions (rear-end, cut-in, side-swipe). Oncoming and
crossing ones need right-of-way rules from the lane graph, so they get
`n/a`, and `compare_policies` reports RSS's coverage.

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
  their agreement with the rear-end rule and with RSS where both decide
  (coverage, and RSS's own ego-fault share, in `comparison.csv`);
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
setting × test adversary for the crash rate, the ego-fault share (and
RSS's), route completion and the timid and aggressive shares: the cross evaluation of
policies trained against different adversaries (and penalties), each tested
without an adversary, with CAT's and with the fair one.

#### The reference on the test split

These are the logged egos replayed in MetaDrive on CAT's 100 test scenes,
against each test adversary. The thresholds are calibrated on the run
without an adversary.

| test adversary | collisions (closed loop / offline prediction) | counterfactual ego / other / shared | rear-end rule decides, agrees | RSS decides, agrees | route completion |
|---|---|---|---|---|---|
| none | 0 | – | – | – | 95.8% |
| CAT | 83 / 95% | 22 / 55 / 6 | 25 of 83, 16 of 23 | 42 of 83, 21 of 29 | 67.0% |
| fair (2 m, ρ 0.1) | 16 / 25% | 9 / 6 / 1 | 0 of 16 | 6 of 16, 0 of 1 | 90.2% |
| avoidability only (∞, ρ 0.5) | 38 / 49% | 12 / 21 / 5 | 11 of 38, 6 of 11 | 21 of 38, 6 of 12 | 85.3% |

The "agrees" counts include only collisions where both the
counterfactual verdict and the rule or RSS name a side (ego or other).

The replayed ego cannot react, so a collision with CAT's adversary is the
adversary's doing. The counterfactual verdict says so in two thirds of
them: the adversary's median β is +1.02 m, the ego's +0.21 m. Where the
rear-end rule or RSS decide, they agree with it about 70% of the time, but
they decide only 30% and 51% of these collisions.

The fair adversary's collisions come out balanced: median β +0.22 m for the
adversary and +0.26 m for the ego. That is what it was built for: it
drives no less acceptably than the ego.

An unchanged ego trajectory scores 1.9× as many aggressive windows next to
CAT's adversary as without one (19.0% against 10.1%), because β_s is
measured against what the others actually do. Policies must therefore be
compared under the same test adversary, which is what `comparison_seeds.md`'s
matrices do.

#### Trained policies (all 17 models, 2026-10-08)

These are the results of `run_eval.sh` with `MODELS=models_eval`, on the 100
test scenes with every model under four test adversaries
(`logs/responsibility/eval/compare`). Each value is a mean ± std over seeds:
three seeds each, except one for the two ablations.

| training | crash, CAT adversary | of which ego / other fault | route completion, no adversary | route completion, CAT adversary | arrival, no adversary |
|---|---|---|---|---|---|
| TD3 replay | 42.0 ± 7.5% | 17.3 / 19.0% | 70.3 ± 4.7% | 63.5 ± 7.6% | 48.0% |
| cat | 42.3 ± 5.0% | 15.0 / 22.0% | 66.5 ± 1.6% | 57.2 ± 3.3% | 46.3% |
| cat_share | 39.3 ± 1.5% | 19.7 / 13.7% | 72.3 ± 7.0% | 64.7 ± 6.6% | 50.3% |
| cat_rss | 35.7 ± 2.5% | 13.3 / 21.0% | 68.2 ± 3.5% | 61.6 ± 4.8% | 48.3% |
| cat_fair2_0.1 | 34.0 ± 4.4% | 16.8 / 12.8% | **76.5 ± 1.6%** | **71.9 ± 1.4%** | **56.3%** |
| cat_fairinf_0.5 (1 seed) | 35.0% | 15.0 / 16.0% | 77.2% | 69.6% | 55.0% |
| cat_fair2_0.1_share (1 seed) | 59.0% | 18.3 / 36.6% | 71.3% | 57.8% | 49.0% |

The ego- and other-fault columns are the crash rate times the share of each
verdict; shared verdicts make up the rest.

- **The fair adversary is the one clear effect.** Against `cat` (Welch
  t-test, 3 vs 3 seeds), route completion rises by 10.0 pp without an
  adversary (p = 0.002) and by 14.7 pp against CAT's adversary (p = 0.008).
  Collisions with CAT's adversary fall by 8.3 pp (p = 0.10). The fall comes
  from collisions that are the other's fault (12.8 vs 22.0%); the ego-fault
  collisions stay at 15–17%.
- **CAT's training does not beat replay training.** Against CAT's adversary
  the crash rates are 42.3 vs 42.0% (p = 0.95), and route completion is
  lower (57.2 vs 63.5%).
- **The weighted collision penalties show no significant effect.**
  Against `cat`, `cat_share` is −3.0 pp on crashes (p = 0.41) and +7.5 pp
  on route completion against CAT's adversary (p = 0.18), and `cat_rss` is
  −6.7 pp on crashes (p = 0.13). The collision penalty they weight is small: −1 a step, about 4
  for a collision, against −10 and the end of the episode for leaving the
  road.
- **Behaviour profiles do not tell the settings apart.** Every TD3 policy has
  25–34% aggressive windows (2.5–3.4× the log) and 4–8% timid ones (4–7×).
- **The policies are weak overall.** Without an adversary they still collide
  in 15–21% of episodes (with non-reacting logged traffic) and leave the road
  in 28–37%.
- **Single-seed ablations.** `cat_fairinf_0.5` matches the fair adversary,
  which suggests avoidability matters more than β. `cat_fair2_0.1_share`
  looks like an outlier (59% crashes). Both need more seeds.

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

`--blame_weighting rss` is the rule-based baseline (`cat_rss`, …). It
removes the whole penalty when RSS puts the collision on the other car
alone, and keeps all of it otherwise. It needs no DenseTNT. Every
collision's RSS verdict is logged in both modes, and `summarize_blame`
reports RSS's verdicts and its agreement with the counterfactual one.

The weighting is tested against a stand-in for the training env
(`tests/responsibility/test_blame_reward.py`: a rear-ended ego keeps none of
the penalty; doubt and failed attributions keep all of it).

## Does the verdict depend on the motion model? (SMART)

`scripts/responsibility/run_smart.sh` measures CAT's scenes with CAT-K's SMART
(the `clsft_E9` checkpoint) through catk's own pipeline (`responsibility/SMART_PLAN.md`):
`export_catk.py` writes catk's cache, catk's `compute_responsibility` runs with
`--query interest --neighbor-future hidden --courtesy-estimator exact`, and
`import_catk.py` turns the observations into `sdc` and `adv` runs of this
repository's format, which `compare_models.py` compares with DenseTNT's.
`cache_womd_for_cat.py` builds the same scenes from WOMD v1.2.1 instead
(from `extract_womd_scenarios.py`'s extract of the 497 scenario ids).

```bash
bash scripts/responsibility/run_smart.sh                          # about 10 h on an L4 for the 500 scenes
~/venvs/catk/bin/python -m scripts.responsibility.import_catk --run logs/catk/scenes/run --out-dir logs/catk/scenes/windows
python -m scripts.responsibility.compare_models --runs logs/responsibility/sdc logs/catk/scenes/windows/sdc \
    --labels DenseTNT SMART --out-dir logs/catk/compare/densetnt_vs_smart_sdc
```

On the 500 logged scenes (6,500 windows per agent, flags at each model's own q90):

| comparison | agent | Spearman β_s | Spearman β_c | aggressive flag: agree / κ | scene verdict: agree / κ |
|---|---|---|---|---|---|
| SMART on WOMD v1.1 vs v1.2.1 | SDC | 0.92 | 0.99 | 0.97 / 0.86 | 0.95 / 0.90 |
| | adversary | 0.94 | 0.99 | 0.98 / 0.89 | 0.94 / 0.88 |
| DenseTNT vs SMART (v1.1) | SDC | 0.60 | 0.47 | 0.82 / 0.17 | 0.62 / 0.24 |
| | adversary | 0.63 | 0.52 | 0.84 / 0.24 | 0.69 / 0.36 |

- The data version does not matter (κ 0.86–0.90): the driveways and
  re-processed maps of v1.2.1 leave SMART's verdicts as they were.
- The motion model does. The two models rank windows moderately alike, but
  their top-decile aggressive flags agree only slightly (κ 0.17–0.24). β_s has
  the same scale in both (positive median 0.05–0.09 m, q90 0.5–0.8 m). Scene
  averages agree far better: Spearman 0.77 (SDC) and 0.82 (adversary) for the
  mean β_s.
- SMART's β_c of the SDC is unexplained so far: median 0.54 nats, against 0.06
  for the adversary; DenseTNT gives 0.05–0.07 for both. Compare SMART's β_c
  by rank only.

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

For the RL experiments, `scripts/responsibility/run_rl.sh` trains the
seven settings × three seeds, and `scripts/responsibility/run_eval.sh`
then evaluates every model and the replayed log against the four test
adversaries and writes the comparison tables. The runbook's steps 8b and
10 describe both.

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
  50 m), as in catk, not a plain radius. For a simplified evaluation,
  `--use-ooi` (in `compute_responsibility` and in live visualisation)
  measures the agent against the scenario's other objects of interest only,
  in every window and whatever the evidence. Every one of CAT's 500 scenes
  has two objects of interest, the self-driving car and the adversary, so
  each scene is measured for that one pair. `--agent` must then be `sdc` or
  `adv`. The interaction scores are still written per neighbour.
- **Courtesy toward vehicles only.** CAT's DenseTNT predicts vehicles
  (`agent_type='vehicle'`); safety is measured toward every neighbour.
- **D_g saturates at 10 m** and CVaR uses the upper-tail convention with
  α = 0.1 (close to the mean), as in catk (`responsibility/risk.py`).
- **Valid counterfactuals (optional, `--valid-counterfactuals`).** β_s can be
  taken over the valid part of the motion set only
  (`responsibility/motion_filter.py`, `responsibility/lanes.py`). The
  trajectories stay the motion model's; the filters only remove some of them.
  - **Same intent, judged by lanes (`--lane-route`).** The route is a set of
    lanes, built in four steps:
    1. Each logged position from step k on is matched to one lane: the closest
       centreline within 2.5 m that runs within 45° of the agent's heading.
    2. A lane counts once the agent has driven 5 m along it. Where a turn lane
       branches off, it overlaps the straight lane for a metre or two, and a
       car going straight passes it there. The lane the agent is on at k
       always counts.
    3. The left and right neighbours of these lanes are added, since a lane
       change keeps the intent.
    4. The chain of exit lanes after the last one is added for 100 m, taking
       every branch, because where the agent goes after the log ends is
       unknown.

    DenseTNT's goal distribution is restricted to the goals within 2 m of
    these lanes **before sampling**, and renormalised, so all 40 samples are
    valid. 99% of its goal mass lies within 2 m of some lane centreline. A
    model without goals has its trajectories' end points checked instead. An
    agent the map does not cover is not restricted: on no lane at k or at its
    last logged position, or on a lane for less than half of its logged
    positions. Parking lots and driveways are the usual cases, and CAT's WOMD
    v1.1 maps have no driveways.
  - **Drivable area:** every point (every 0.5 s) is within 3 m of a vehicle
    lane centreline.
  - **Kinematics:** longitudinal acceleration stays within −9 to +5 m/s² and
    lateral acceleration below 7 m/s², from 0.5 s-averaged speeds.
  - **Collision:** the trajectory does not drive through a third agent's
    logged future. b itself does not count, because its distance is what β_s
    measures.
  - **Courtesy (`--courtesy-valid-goals`):** the KL is taken over b's goals on
    the lanes b can reach (its lanes, then their exit chain for 200 m, each
    with neighbours), with both distributions renormalised there. There is no
    restriction where less than half of b's goal mass lies on those lanes,
    because then the map misses where b is going.

    **Without lane topology, drop `--courtesy-valid-goals`** rather than
    approximate it (for b's own drive mode, see `--courtesy-same-mode`
    below). The restriction needs
    the exit chain of the lane graph. Two map-free stand-ins were measured
    against it with DenseTNT on the first 30 scenes: every SDC window and
    vehicle neighbour, 937 pairs, 730 of them with the HD support defined.
    The stand-ins were (A) b's goals within D m of b's logged path, and (B)
    b's goals within a forward cone of its heading. Both moved β_c further
    from the HD values than no restriction does:

    | β_c support | MAE | MAE on the top 10% | Spearman |
    |---|---|---|---|
    | none | 0.0039 nats | 0.018 nats | 0.996 |
    | A, logged path ±8 m | 0.017 nats | 0.122 nats | 0.983 |
    | A, logged path ±20 m | 0.0075 nats | 0.047 nats | 0.992 |
    | B, forward cone 90° | 0.0067 nats | 0.030 nats | 0.982 |
    | B, forward cone 120° | 0.0058 nats | 0.026 nats | 0.987 |

    The HD support leaves out little: a median of 1.7% of b's goal mass, and
    16.5% at the 90th percentile. A support around b's logged path does more
    harm. It cuts away the goals of another intent, and those are exactly
    where a's presence moves b's mass, so it hides the change that β_c
    measures.
  - **β_c within b's own drive mode (`--courtesy-same-mode`).**
    `--courtesy-valid-goals` keeps every goal b can reach, so a change of
    mode (straight → turn) still counts. `--courtesy-same-mode` keeps only
    the goals of the mode b drove in the log, so β_c measures how a changes
    b's plan within that mode. It has two versions:
    - `lanes`: goals within 2 m of b's lane route, as `--lane-route` builds
      it for the queried agent.
    - `path`: goals within `--courtesy-path-lateral` (6 m) of b's logged
      path, extended 100 m along its last heading. No map is needed.

    There is no restriction where b's mode holds less than 0.1% of its goal
    mass, since there is nothing to renormalise. The mass inside the
    mode is written per neighbour as `courtesy_goal_mass`.

    **Calibration of `path` against `lanes`**, on the 761 pairs above with b
    on an HD lane route. DenseTNT puts a median 97.6% of b's mass on b's
    own route (10th percentile 72.6%). The two are close in the median
    (0.015 vs 0.016 nats unrestricted), but on the 10% of pairs with the
    largest β_c the restriction changes β_c by 0.126 nats: these are the
    pairs where a moves b's mass between modes.

    | β_c support, against `lanes` | MAE | MAE on the top 10% | Spearman |
    |---|---|---|---|
    | none | 0.0165 nats | 0.126 nats | 0.986 |
    | reachable (`--courtesy-valid-goals`) | 0.0121 nats | 0.101 nats | 0.992 |
    | `path`, 3 m | 0.0170 nats | 0.099 nats | 0.950 |
    | `path`, 6 m (default) | 0.0102 nats | 0.055 nats | 0.965 |
    | `path`, 8 m | 0.0099 nats | 0.056 nats | 0.969 |

    Inside 6 m, `path` keeps 86% of the mass on b's route and 47% of the
    mass off it, which is goals on parallel lanes and just past the lanes'
    reach. Beyond 6 m it stops improving.
  - **A path-based alternative to the lane route.** `--route-tolerance` keeps
    the trajectories that stay within that many metres of the logged path,
    which is extended 100 m along the last heading.
  - **Same intent without lane topology (`--intent`, `responsibility/intent.py`).**
    A map built by perception has lane lines and road edges but almost no
    topology in intersections, so the lane route has nothing to work with
    there. `--intent` judges the manoeuvre from the trajectories:
    1. *Heading:* at the last logged step T* (at most 8 s ahead), the
       alternative heads within 45° (`--intent-heading`) of the logged path's
       direction **where the alternative is**, not of the log at the same
       time. A braking alternative still in the middle of the logged turn
       is kept; with the same-time comparison it was not. At 45° the
       same-time comparison drops 4.7% of the mass that ends on a route
       lane, and this comparison drops 2.6%. An agent whose logged path is shorter
       than 5 m shows no intent and is not restricted.
    2. *Lateral:* it stays within 8 m (`--intent-lateral`) of the logged path.
       This drops a turn that has only begun by T*, which still heads within
       45° of the path.
    3. *Road edges (`--intent-edges`, off by default):* every second, the
       segment from the alternative to the nearest point of the logged path
       crosses no `ROAD_EDGE_BOUNDARY` or `ROAD_EDGE_MEDIAN`. A missing edge
       removes nothing, so an incomplete map only loosens the test.

    **Calibration against the lane route on the first 30 scenes.** The
    reference is β_s over the weighted motion sets of
    `logs/filter_trial/weighted_filtered`, restricted by the HD lane route
    (end points within 2 m of a route lane), on 930 SDC–neighbour pairs. The
    sets are taken offline from the records, with no other filter.

    | restriction | MAE β_s | MAE on the 105 pairs > 0.05 m | largest error |
    |---|---|---|---|
    | none | 0.0041 m | 0.0094 m | 0.62 m |
    | heading 45° | 0.0026 m | 0.0102 m | 0.30 m |
    | heading 45° + lateral 8 m (default) | 0.0027 m | 0.0126 m | 0.26 m |
    | + road edges (HD) | 0.0030 m | 0.0158 m | 0.26 m |
    | + road edges, 30% dropped, 0.3 m noise | 0.0030 m | 0.0154 m | 0.26 m |
    | + road edges, 60% dropped, 0.5 m noise | 0.0028 m | 0.0137 m | 0.26 m |

    The heading test drops all the mass that ends on a non-route lane more
    than 15 m from the logged path. What it keeps of the non-route mass lies
    within 15 m and heads the same way: two lanes over, beyond the lane
    route's one-hop neighbours, or a turn that has only begun (scene 15,
    steps 20–35, which the lateral test drops). Road edges move β_s
    **away** from the lane route, and degrading them moves it back, because
    edges at intersection corners and islands cut off alternatives the lane
    route keeps. That is why they are off by default. At a 2 s horizon the
    choice matters little in any case: even no restriction is within 0.01 m
    of the lane route on the pairs that matter.
  - **Drivable area without centrelines (`--drivable-edges`,
    `responsibility/edges.py`).** The drivable test above needs lane
    centrelines, and a perceived map has none inside intersections. Here the
    path is checked every 0.5 s from the agent's position, and the
    trajectory must cross no `ROAD_EDGE_BOUNDARY` or `ROAD_EDGE_MEDIAN`. An
    edge that the agent's own logged path crosses does not count (map noise,
    or a driveway, which WOMD v1.1 lacks), and a gap in the edges removes
    nothing.

    **Measured on the first 30 scenes.** The reference is the centreline
    test (3 m) on the HD map, over the weighted sets of
    `logs/filter_trial/weighted_filtered` (345 windows, 930 SDC–neighbour
    pairs, no other filter). A perception-like map is made from the HD one:
    lane topology removed, and the centrelines of the 32% of lanes that cross
    another lane at more than 30° (intersection connectors) removed. The
    road edges are then cut into 10 m pieces, a share of them dropped, and
    the rest jittered.

    | drivable test | wrongly dropped | wrongly kept | MAE β_s | MAE, pairs > 0.05 m | largest error |
    |---|---|---|---|---|---|
    | centrelines, perceived map | 15.7% | 16.7% | 0.033 m | 0.087 m | 4.71 m |
    | road edges (as in the HD map) | 1.3% | 36% | 0.0042 m | 0.026 m | 0.72 m |
    | road edges, 30% dropped, 0.3 m noise | 1.2% | 51% | 0.0059 m | 0.035 m | 0.79 m |
    | road edges, 60% dropped, 0.5 m noise | 1.3% | 66% | 0.0080 m | 0.037 m | 1.19 m |

    "Wrongly dropped" is the share of the mass the reference keeps that the
    test removes. "Wrongly kept" is the share of the mass the reference
    removes that the test keeps. On a perceived map the centreline test
    removes the alternatives that cross an intersection. The edge test
    almost never removes too much, even with gaps and noise. It keeps road
    surface that is more than 3 m from a lane centre (parking lanes,
    shoulders), and whatever leaves through a gap. Combining the two did not
    help. Keeping points within 3 m of a centreline, or farther than C from
    every centreline, and also checking the edges, wrongly dropped 19–22%
    of the mass for C = 4.5, 6 and 8 m: points just past the end of a
    removed intersection lane fall between 3 m and C.

  `--motion-set weighted` uses DenseTNT's whole goal grid (0.999 of the mass),
  probability-weighted, instead of 40 samples. That set holds a median of 944
  trajectories per window (P10 402, P90 2742, max 15568 over the 345 windows
  with neighbours of `logs/filter_trial/weighted_filtered`), each completed
  and filtered. `--motion-set topk` keeps the `--n-samples` most probable
  goals instead, probability-weighted and renormalised over them. The top 40
  hold a median 0.69 of the mass (P10 0.39). Offline, from the weighted run's
  records, with the same filters re-applied to the top k, against the full
  weighted β_s on 961 neighbour pairs:

  | motion set | MAE β_s, all pairs | MAE on the 110 pairs with β_s > 0.05 m | largest error |
  |---|---|---|---|
  | top 40 | 0.014 m | 0.060 m | 1.00 m |
  | top 100 | 0.008 m | 0.036 m | 0.91 m |
  | top 200 | 0.004 m | 0.021 m | 0.65 m |
  | 40 samples (`logs/filter_trial/filtered`) | 0.010 m | 0.035 m | 0.89 m |
  | 40 by NMS (`--motion-set nms`, run) | 0.005 m | 0.015 m | 0.53 m |

  The top 40 are deterministic but less accurate than 40 samples where β_s
  matters: truncation drops the low-probability, slower executions that
  `weighted` was added for. `--motion-set nms` (`modes.py`) instead spreads
  its `--n-samples` goals over the distribution with CAT's goal NMS (7.2 m
  times CAT's speed scale factor, 0.5–1.0), fills up with the next most
  probable goals when fewer survive, and weights each goal by the probability
  of the grid goals nearest to it, so the weights sum to 1. It was run with
  DenseTNT on the same 30 scenes and filters (on CPU, 10–19 s per scene). It
  is the most accurate of the 40-trajectory sets, more accurate than the top
  200: correlation 0.996 with the weighted β_s (40 samples: 0.981), the
  β_s > 0.05 m flag agrees on 99.9% of the pairs, and the mean difference is
  −0.001 m. MTR's adapter applies the same rule to its 64 intention
  endpoints.

  **Results on the first 30 scenes.** The runs cover 390 SDC windows with
  DenseTNT (`logs/filter_trial`) and are compared with the unfiltered run of
  the same seed.

  | | β_s | β_c |
  |---|---|---|
  | Spearman with the unfiltered values | 0.96 | 0.99 |
  | top-decile flags: agreement / κ | 0.995 / 0.94 | 0.985 / 0.91 |
  | mean change | −0.002 m | −0.007 nats |

  - **Lane route.** The goal mass on the route lanes has a median of 0.98 and
    is below 0.9 in 21% of the windows; that is where the lane route filters
    something out. The windows that changed most are scene 29 at step 10
    (β_s +0.57 → +0.01 m) and scene 16 at step 35 (+1.27 → +0.93 m).
  - **Each filter alone, as a share of probability mass.** These were measured
    with the path-based route:
    - route 23%
    - drivable area 12%
    - collision 1%
    - kinematics 0%
  - **Why little changes.** Within the 2 s metric horizon (the paper's T_f, 20
    steps of 0.1 s on nuScenes), alternatives with another intent have not
    yet left the route: their median distance from the logged path over 2 s
    is 0.36 m. The filters make the set valid, but they cannot add reactions
    the motion model does not predict. The paper names this limit itself
    (Sec. VI-A): the motion set is open-loop, and "controls do not change even
    after observing other agents' future states". The next item tests
    reactions directly.
  - **Reactions (SMART, first 30 scenes, `logs/catk/trial30`).** catk's
    `--neighbor-future logged` lets SMART's alternatives see the neighbours'
    logged motion step by step, never ahead of time. It needs no retraining,
    since SMART is autoregressive with 0.5 s tokens. Compared with the
    open-loop `hidden` mode:

    | | SDC | adversary |
    |---|---|---|
    | Spearman β_s | 0.93 | 0.93 |
    | aggressive-flag κ | 0.65 | 0.58 |
    | mean β_s | +0.074 m in both modes | +0.123 vs +0.120 m |

    Where the logged SDC slowed by more than 2 m/s within the 2 s horizon
    (20 windows), the alternatives that slowed by at least half as much were:

    | | DenseTNT | SMART `hidden` | SMART `logged` |
    |---|---|---|---|
    | all 20 windows | 92% | 93% | 93% |
    | the one sudden onset (scene 1, step 50) | 5% | 35% | 38% |

    In the other windows the deceleration had already begun and shows in the
    history, so the alternatives simply continue it. Within 2 s, the
    neighbours' motion hardly changes what the alternatives do.

## Limitations

- At 1–2 s horizons the closest approach is often at the first step, where
  every sample starts where the log does; slow or stationary agents therefore
  get β_s ≈ 0, which says "no alternative did better", not "safe".
- Courtesy is intent-level and DenseTNT only sees 1.1 s of history, so an
  agent's influence enters only through its recent past and current state.
- Thresholds are relative to a reference population; the verdict is "unusual
  compared with that driving", not an absolute safety judgement.
- Over the 2 s metric horizon, the alternatives stay close to the logged
  future: DenseTNT's median ADE is 0.26 m. They continue a deceleration that
  has begun, but rarely anticipate one that has not (see "Reactions" above).
  An earlier version of this note said that only 4% brake where the driver
  braked. That number counted alternatives braking harder than the driver,
  not alternatives braking at all.
- What the RL settings use holds up under SMART (`SMART_PLAN.md`, S4 on the
  replay reference rollouts):
  - **Adversary β.** The ordering of the adversaries' β toward the ego is the
    same under both models: logged +0.03, fair +0.77, avoidability only
    +1.27, CAT +1.98 m (SMART, per-scene median). The scale is not: SMART puts
    CAT's adversaries at about a third of DenseTNT's value, so τ = 2 m is a
    DenseTNT-calibrated threshold.
  - **Collision verdicts.** The two models agree on 62–81% of the
    collisions, about as often as the counterfactual verdict agrees with RSS
    and the rear-end rule.
- Window-level flags depend on the motion model: DenseTNT and SMART agree with
  κ 0.17–0.24 on the aggressive windows of the same scenes. Scene-level and
  aggregate results are more stable (Spearman 0.77–0.83 of the scene mean and
  maximum β_s).
