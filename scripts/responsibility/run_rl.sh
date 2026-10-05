#!/usr/bin/env bash
# Runbook step 8b on a server: train the seven RL settings x seeds with cat_RLtrain.py.
#
#   replay                no adversary
#   cat                   CAT's adversary
#   cat_share             CAT's adversary, collision penalty weighted by the ego's share
#   cat_rss               CAT's adversary, collision penalty weighted by RSS (rule-based baseline)
#   cat_fair2_0.1         the fair adversary (beta <= 2 m, ego avoidability >= 0.1)
#   cat_fairinf_0.5       avoidability only (tau = inf, rho = 0.5): the ablation of beta
#   cat_fair2_0.1_share   the fair adversary and the share-weighted penalty
#
# usage (from the repository root, environment active; MetaDrive installed as in runbook step 7):
#   bash scripts/responsibility/run_rl.sh
#   PARALLEL=12 SEEDS="0 1 2" bash scripts/responsibility/run_rl.sh
#   ABLATION_SEEDS="0 1 2" bash scripts/responsibility/run_rl.sh   # all 21 runs
#   DRY=1 bash scripts/responsibility/run_rl.sh          # print the runs, start nothing
#
# variables (defaults):
#   SEEDS         seeds                                            ("0 1 2")
#   SETTINGS      names from the list above                        (all seven)
#   ABLATION_SEEDS  the seeds (of SEEDS) for the two ablations,
#                 cat_fairinf_0.5 and cat_fair2_0.1_share: by default
#                 seed 0 only, so 5 x 3 + 2 x 1 = 17 runs. More seeds
#                 only if seed 0 differs clearly from cat              ("0")
#   STEPS         training steps                                   (1000000)
#   PARALLEL      runs at a time                                   (by free memory at 4.5 GB a run, at most the cores)
#   NO_STORE_MAP  1: rebuild maps each episode (--no_store_map)    (1)
#                 Caching every map (0) is faster but grew a run past 5 GB within 8 minutes
#                 on the pilot machine; with 1 a run stayed at 3.3-4.2 GB.
#   DRY           1: only print what would run                     (0)
#
# Each run logs to logs/rl/<name>_s<seed>.log and, once it exits cleanly, leaves
# logs/rl/<name>_s<seed>.done; re-running the script skips those (a run that failed
# starts over: cat_RLtrain.py does not resume). Models go to models/<name>_s<seed>_*,
# saved at every evaluation (--eval_freq, 25000 steps); curves to logs/<name>_MDWaymo-seed<seed>-0/,
# attributions to logs/blame/. logs/rl/resources.log gets a line a minute; below 2 GB of
# available memory the youngest run is stopped (and has to be started again), below
# 5 GB of disk all of them.
set -uo pipefail

SEEDS=${SEEDS:-"0 1 2"}
SETTINGS=${SETTINGS:-"replay cat cat_share cat_rss cat_fair2_0.1 cat_fairinf_0.5 cat_fair2_0.1_share"}
ABLATION_SEEDS=${ABLATION_SEEDS:-0}
STEPS=${STEPS:-1000000}
NO_STORE_MAP=${NO_STORE_MAP:-1}
DRY=${DRY:-0}
mem_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
by_mem=$(( (mem_gb - 4) * 10 / 45 ))
PARALLEL=${PARALLEL:-$(( by_mem < $(nproc) ? by_mem : $(nproc) ))}
[ "$PARALLEL" -ge 1 ] || PARALLEL=1

FAIR="--adv_selection fair --resp_threshold 2 --resp_avoid 0.1"
ABL="--adv_selection fair --resp_threshold inf --resp_avoid 0.5"
planned() {  # setting, seed
  case "$1" in
    cat_fairinf_0.5|cat_fair2_0.1_share) [[ " $ABLATION_SEEDS " == *" $2 "* ]] ;;
    *) true ;;
  esac
}
flags_of() {
  case "$1" in
    replay) echo "--mode replay" ;;
    cat) echo "--mode cat" ;;
    cat_share) echo "--mode cat --blame_weighting share" ;;
    cat_rss) echo "--mode cat --blame_weighting rss" ;;
    cat_fair2_0.1) echo "--mode cat $FAIR" ;;
    cat_fairinf_0.5) echo "--mode cat $ABL" ;;
    cat_fair2_0.1_share) echo "--mode cat $FAIR --blame_weighting share" ;;
    *) echo "unknown setting $1" >&2; exit 1 ;;
  esac
}

if command -v nvidia-smi >/dev/null; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
  GPUS=("")
fi
extra="--max_timesteps $STEPS --save_model"
[ "$NO_STORE_MAP" = 1 ] && extra="$extra --no_store_map"

mkdir -p logs/rl
jobs=()
k=0
for seed in $SEEDS; do
  for name in $SETTINGS; do
    run=${name}_s$seed
    planned "$name" "$seed" || continue
    if [ -f "logs/rl/$run.done" ]; then
      echo "skip $run (done)"
      continue
    fi
    gpu=${GPUS[$((k % ${#GPUS[@]}))]}
    k=$((k + 1))
    jobs+=("$run|$gpu|$(flags_of "$name") --seed $seed $extra")
  done
done
echo "${#jobs[@]} runs, $PARALLEL at a time, on GPU(s) '${GPUS[*]}' (${mem_gb} GB available, $(nproc) cores)"
if [ "$DRY" = 1 ]; then
  for j in "${jobs[@]}"; do echo "  ${j%%|*}: python -u cat_RLtrain.py ${j##*|}"; done
  exit 0
fi
[ "${#jobs[@]}" -gt 0 ] || exit 0

export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

(
  while true; do
    mem=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
    disk=$(df -BM --output=avail . | tail -1 | tr -dc 0-9)
    runs=$(pgrep -fc "^python -u cat_RLtrain.py")
    echo "$(date '+%m-%d %H:%M') mem_avail ${mem}M disk_avail ${disk}M runs $runs" >> logs/rl/resources.log
    if [ "$disk" -lt 5000 ]; then
      echo "$(date '+%m-%d %H:%M') disk below 5 GB: stopping all runs" >> logs/rl/resources.log
      pkill -f "^python -u cat_RLtrain.py"
    elif [ "$mem" -lt 2000 ] && [ "$runs" -gt 0 ]; then
      echo "$(date '+%m-%d %H:%M') memory below 2 GB: stopping the youngest run" >> logs/rl/resources.log
      pkill -n -f "^python -u cat_RLtrain.py"
    fi
    sleep 60
  done
) &
watchdog=$!
trap 'kill $watchdog 2>/dev/null' EXIT

printf '%s\n' "${jobs[@]}" | xargs -P "$PARALLEL" -I{} bash -c '
  job="{}"; run=${job%%|*}; rest=${job#*|}; gpu=${rest%%|*}; flags=${rest#*|}
  echo "$(date "+%m-%d %H:%M") start $run"
  CUDA_VISIBLE_DEVICES=$gpu nice -n 5 python -u cat_RLtrain.py $flags > logs/rl/$run.log 2>&1
  status=$?
  [ $status -eq 0 ] && touch logs/rl/$run.done
  echo "$(date "+%m-%d %H:%M") $run exit $status"'

echo
echo "finished: $(ls logs/rl/*.done 2>/dev/null | wc -l) runs done; failed or stopped ones have no .done:"
for seed in $SEEDS; do for name in $SETTINGS; do
  planned "$name" "$seed" && [ ! -f "logs/rl/${name}_s$seed.done" ] && echo "  ${name}_s$seed"
done; done
