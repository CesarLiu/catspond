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
  50 m), as in catk, not a plain radius.
- **Courtesy toward vehicles only.** CAT's DenseTNT predicts vehicles
  (`agent_type='vehicle'`); safety is measured toward every neighbour.
- **D_g saturates at 10 m** and CVaR uses the upper-tail convention with
  α = 0.1 (close to the mean), as in catk (`responsibility/risk.py`).
- **Valid counterfactuals (optional, `--valid-counterfactuals`).** β_s can be
  taken over the valid part of the motion set only (`responsibility/motion_filter.py`).
  The trajectories stay the motion model's; filters only remove some of them:
  - **route:** the alternative stays within 2 m of the agent's logged route,
    so its intent is unchanged and speed and timing stay free;
  - **drivable:** it stays within 3 m of a vehicle lane centreline;
  - **kinematics:** longitudinal acceleration stays within −9 to +5 m/s² and
    lateral acceleration below 7 m/s²;
  - **collision:** it does not drive through a third agent's logged future.

  `--motion-set weighted` uses DenseTNT's whole goal grid (0.999 of the mass),
  probability-weighted, instead of 40 samples.

  On the first 30 scenes (390 SDC windows, DenseTNT, `logs/filter_trial`), each
  filter alone removes, in probability mass: route 23%, drivable 12%,
  collision 1%, kinematics 0%. β_s hardly changes: Spearman 0.93 with the
  unfiltered values, mean change −0.003 m (40 samples) and −0.009 m (weighted
  grid); only 1% of windows drop by more than 0.5 m. Within the 2 s metric
  horizon the alternatives that leave the route have not yet left it (their
  median distance from the logged path over 2 s is 0.36 m). The filters make
  the set valid, but they cannot add what the motion model does not predict:
  alternatives that brake when a neighbour cuts in.

## Limitations

- At 1–2 s horizons the closest approach is often at the first step, where
  every sample starts where the log does; slow or stationary agents therefore
  get β_s ≈ 0, which says "no alternative did better", not "safe".
- Courtesy is intent-level and DenseTNT only sees 1.1 s of history, so an
  agent's influence enters only through its recent past and current state.
- Thresholds are relative to a reference population; the verdict is "unusual
  compared with that driving", not an absolute safety judgement.
- The motion set lacks short-term reactions. Over the 2 s metric horizon,
  DenseTNT's alternatives stay within 0.26 m (median ADE) of the logged future.
  In the 403 windows where the logged driver braked by more than 2 m/s, only
  4% of them braked as well. β_s therefore cannot show that braking would
  have kept more distance. Filtering (above) does not change this.
- Window-level flags depend on the motion model: DenseTNT and SMART agree with
  κ 0.17–0.24 on the aggressive windows of the same scenes. Scene-level and
  aggregate results are more stable (Spearman 0.77–0.83 of the scene mean and
  maximum β_s).
