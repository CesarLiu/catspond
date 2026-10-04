#!/usr/bin/env bash
# Runbook step 10 on a server, after run_rl.sh: evaluate every trained model and the replayed
# log on CAT's test split against no adversary, CAT's, the fair one (tau 2 m, rho 0.1) and the
# avoidability-only one (tau inf, rho 0.5); compute their responsibility and collision
# attributions; fit responsibility levels on the replayed log; write the comparison tables.
#
# usage (from the repository root, environment active):
#   bash scripts/responsibility/run_eval.sh
#   PARALLEL=12 bash scripts/responsibility/run_eval.sh
#
# variables (defaults):
#   MODELS    directory of cat_RLtrain.py models (<name>_s<seed>_actor ...)   (models)
#   ROLLOUTS  where rollouts go: <ROLLOUTS>/<policy>/<adversary mode>/      (rollouts)
#   OUT       where runs go: <OUT>/<policy>/<adversary mode>/              (logs/responsibility/policies)
#   FIRST, N  scenes FIRST .. FIRST+N-1                                     (400, 100: CAT's test split)
#   PARALLEL  processes at a time                                          (by free memory at 3 GB each, at most the cores)
#   LEVELS    1: fit levels on the replayed log and report shares per level (1)
#
# Every stage resumes: collect_rollouts skips scenes it has recorded, compute_responsibility
# scenes it has measured. Policies are named td3_<model> (td3_cat_s0, ...), so
# compare_policies averages the seeds of a setting (comparison_seeds.md: matrices of
# training setting x test adversary).
set -uo pipefail

MODELS=${MODELS:-models}
ROLLOUTS=${ROLLOUTS:-rollouts}
OUT=${OUT:-logs/responsibility/policies}
FIRST=${FIRST:-400}
N=${N:-100}
LEVELS=${LEVELS:-1}
mem_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
by_mem=$(( (mem_gb - 2) / 3 ))
PARALLEL=${PARALLEL:-$(( by_mem < $(nproc) ? by_mem : $(nproc) ))}
[ "$PARALLEL" -ge 1 ] || PARALLEL=1

export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export ROLLOUTS OUT FIRST N
mkdir -p "$OUT/logs" logs/rollouts

# 1. rollouts: one job per policy, its three adversary modes in turn (each also writes none/)
policies=(replay)
for actor in "$MODELS"/*_s[0-9]*_actor; do
  [ -e "$actor" ] && policies+=("${actor%_actor}")
done
echo "$(date '+%m-%d %H:%M') rollouts of ${#policies[@]} policies (the replayed log and $((${#policies[@]} - 1)) models), $PARALLEL at a time"
printf '%s\n' "${policies[@]}" | xargs -P "$PARALLEL" -I{} bash -c '
  policy="{}"
  if [ "$policy" = replay ]; then name=replay; else name=td3_$(basename "$policy"); fi
  for adv in "--adv_selection cat" \
             "--adv_selection fair --resp_threshold 2 --resp_avoid 0.1" \
             "--adv_selection fair --resp_threshold inf --resp_avoid 0.5"; do
    tag=$(echo "$adv" | awk "{print \$2 \$4}")
    nice -n 5 python -m scripts.responsibility.collect_rollouts --policy "$policy" --policy_name "$name" --adversary $adv \
      --first "$FIRST" --n "$N" --out_dir "$ROLLOUTS" > "logs/rollouts/${name}_$tag.log" 2>&1 \
      || echo "FAILED: rollouts of $name against $tag (logs/rollouts/${name}_$tag.log)"
  done
  echo "$(date "+%m-%d %H:%M") rollouts of $name done"'

# 2. responsibility and collision attributions of every rollout directory
dirs=()
for d in "$ROLLOUTS"/*/*/; do [ -d "$d" ] && dirs+=("${d%/}"); done
echo "$(date '+%m-%d %H:%M') responsibility of ${#dirs[@]} rollout directories, $PARALLEL at a time"
printf '%s\n' "${dirs[@]}" | xargs -P "$PARALLEL" -I{} bash -c '
  dir="{}"; mode=$(basename "$dir"); policy=$(basename "$(dirname "$dir")")
  nice -n 5 python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 --rollouts "$dir" \
    --out-dir "$OUT/$policy/$mode" --device cuda > "$OUT/logs/${policy}_$mode.log" 2>&1 \
    || echo "FAILED: responsibility of $policy/$mode ($OUT/logs/${policy}_$mode.log)"'

# 3. levels on the replayed log, then the comparison (thresholds from the first run, replay/none)
runs=("$OUT/replay/none")
for d in "$OUT"/*/*/; do
  d=${d%/}
  [ "$d" = "$OUT/replay/none" ] || [ "$(basename "$(dirname "$d")")" = logs ] || [ ! -e "$d/config.json" ] || runs+=("$d")
done
hmm=()
if [ "$LEVELS" = 1 ]; then
  python -m scripts.responsibility.fit_levels --runs "${runs[@]}" --fit-runs "$OUT/replay/none" --out-dir "$OUT/levels" \
    > "$OUT/logs/levels.log" 2>&1 && hmm=(--hmm "$OUT/levels/hmm.pkl") || echo "FAILED: levels ($OUT/logs/levels.log)"
fi
python -m scripts.responsibility.compare_policies --runs "${runs[@]}" "${hmm[@]}" --out-dir "$OUT/compare" \
  > "$OUT/logs/compare.log" 2>&1 || echo "FAILED: comparison ($OUT/logs/compare.log)"
echo "$(date '+%m-%d %H:%M') done: $OUT/compare/comparison_seeds.md (seed averages and matrices), comparison.md (every run)"
