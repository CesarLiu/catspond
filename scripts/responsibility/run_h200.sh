#!/usr/bin/env bash
# Step 2 on a GPU server: safety/courtesy responsibility of the self-driving
# car and of CAT's adversary over all scenes, then aggressiveness summaries and
# responsibility levels.
#
# usage (from the repository root, environment active; see setup_env.sh):
#   bash scripts/responsibility/run_h200.sh
#   SHARDS=24 STRIDE=5 bash scripts/responsibility/run_h200.sh
#
# variables (defaults):
#   OUT      output root                               (logs/responsibility)
#   SCENES   scene directory                           (raw_scenes_500)
#   N        scenes                                    (500)
#   AGENTS   agents to evaluate                        ("sdc adv")
#   SHARDS   processes per agent                       (16)
#   STRIDE   steps between context windows (0.1 s)     (5)
#   SAMPLES  DenseTNT samples per window (safety)      (40)
#   RECORDS  1: also write per-scene records for offline
#            inspection/visualisation (--save-records)  (0)
#   EXTRA    further compute_responsibility.py flags    ("")
#            e.g. EXTRA="--horizon 30 --d-sat 15"
#
# The work per window is mostly CPU (building DenseTNT's inputs) plus small GPU
# passes, so many processes share a GPU; SHARDS x len(AGENTS) processes are
# spread round-robin over the visible GPUs and share the CPU cores. Logs go to
# OUT/<agent>/logs/shard_<i>.log. Re-running the same command resumes:
# finished scenes are skipped.
set -euo pipefail

OUT=${OUT:-logs/responsibility}
SCENES=${SCENES:-raw_scenes_500}
N=${N:-500}
AGENTS=${AGENTS:-"sdc adv"}
SHARDS=${SHARDS:-16}
STRIDE=${STRIDE:-5}
SAMPLES=${SAMPLES:-40}
RECORDS=${RECORDS:-0}
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"
[ "$RECORDS" = 1 ] && EXTRA_ARGS+=(--save-records)

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
  GPUS=("")
fi
read -r -a AGENT_LIST <<< "$AGENTS"
n_proc=$(( SHARDS * ${#AGENT_LIST[@]} ))
threads=$(( $(nproc) / n_proc ))
[ "$threads" -ge 1 ] || threads=1
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
export TF_NUM_INTRAOP_THREADS=$threads TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONDONTWRITEBYTECODE=1
echo "$n_proc processes on GPU(s) '${GPUS[*]}', $threads CPU thread(s) each"

pids=()
names=()
k=0
for agent in "${AGENT_LIST[@]}"; do
  mkdir -p "$OUT/$agent/logs"
  for ((i = 0; i < SHARDS; i++)); do
    gpu=${GPUS[$((k % ${#GPUS[@]}))]}
    k=$((k + 1))
    device=cuda
    [ -n "$gpu" ] || device=cpu
    CUDA_VISIBLE_DEVICES=$gpu python -m scripts.responsibility.compute_responsibility \
      --scenes "$SCENES" --n "$N" --agent "$agent" --out-dir "$OUT/$agent" \
      --stride "$STRIDE" --n-samples "$SAMPLES" --device "$device" \
      --num-shards "$SHARDS" --shard-index "$i" "${EXTRA_ARGS[@]}" > "$OUT/$agent/logs/shard_$i.log" 2>&1 &
    pids+=($!)
    names+=("$agent shard $i")
  done
done

status=0
for j in "${!pids[@]}"; do
  if ! wait "${pids[$j]}"; then
    echo "FAILED: ${names[$j]} (see its log)" >&2
    status=1
  fi
done
[ $status -eq 0 ] || { echo "some shards failed; fix and re-run to resume" >&2; exit 1; }

echo; echo "=== aggressive windows by threshold (calibrated on the self-driving car)"
python -m scripts.responsibility.summarize_responsibility --run "$OUT/sdc"
if [[ " ${AGENT_LIST[*]} " == *" adv "* ]]; then
  python -m scripts.responsibility.summarize_responsibility --run "$OUT/adv" --reference "$OUT/sdc"
fi

echo; echo "=== responsibility levels (HMM over all runs)"
runs=()
for agent in "${AGENT_LIST[@]}"; do runs+=("$OUT/$agent"); done
python -m scripts.responsibility.fit_levels --runs "${runs[@]}" --out-dir "$OUT/levels"
python -m scripts.responsibility.fit_levels --runs "${runs[@]}" --out-dir "$OUT/levels_log" --log-courtesy
echo; echo "done: $OUT/<agent>/summary, $OUT/levels, $OUT/levels_log"
