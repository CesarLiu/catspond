#!/usr/bin/env bash
# SMART_PLAN.md S3: responsibility of the self-driving car and CAT's adversary over CAT's scenes
# with CAT-K's SMART, by catk's own pipeline, with settings matched to the DenseTNT runs here.
#
#   1. export the scenes into catk's cached format (export_catk.py; catk's environment)
#   2. catk's compute_responsibility on them:
#        --query interest            the two objects of interest: the SDC and CAT's adversary
#        --neighbor-future hidden    the open-loop reading DenseTNT has (history up to k only)
#        --courtesy-estimator exact  the exact KL, as the goal KL here
#      sample count, CVaR alpha, saturation, horizon and window stride are catk's defaults,
#      which equal the DenseTNT runs' (40, 0.1, 10 m, 2 s, 0.5 s)
#
# usage (from this repository's root; catk installed by its install/setup_server.sh):
#   bash scripts/responsibility/run_smart.sh
#   ROLLOUTS=rollouts/replay/cat OUT=logs/catk/replay_cat bash scripts/responsibility/run_smart.sh
#   EXPORT=0 OUT=logs/catk/v121 bash scripts/responsibility/run_smart.sh   # cache from cache_womd_for_cat.py
#
# variables (defaults):
#   CATK_ROOT   the catk checkout                                     ($HOME/catk)
#   PYTHON      catk's Python                                         ($HOME/venvs/catk/bin/python)
#   CKPT        SMART checkpoint                                      ($CATK_ROOT/ckpts/clsft_E9.ckpt)
#   SCENES      CAT's scenes                                          (raw_scenes_500)
#   ROLLOUTS    a rollout directory: measure the scenes rebuilt from it instead ("")
#   OUT         output root: OUT/cache (exported scenes), OUT/run (catk results)  (logs/catk/scenes)
#   SHARDS      processes (each with its own GPU batches)              (1)
#   BATCH       scene copies per model call; 128 ran out of the L4's 22 GB with
#               the exact courtesy in a 61-agent scene, 32 peaked at 6.3 GB   (32)
#   EXPORT      1: export SCENES/ROLLOUTS into OUT/cache first; 0: use OUT/cache as it is  (1)
#   EXTRA       further catk compute_responsibility flags              ("")
# Re-running resumes: exported scenes and finished scenarios are skipped.
set -euo pipefail

CATK_ROOT=${CATK_ROOT:-$HOME/catk}
PYTHON=${PYTHON:-$HOME/venvs/catk/bin/python}
CKPT=${CKPT:-$CATK_ROOT/ckpts/clsft_E9.ckpt}
SCENES=${SCENES:-raw_scenes_500}
ROLLOUTS=${ROLLOUTS:-}
OUT=${OUT:-logs/catk/scenes}
SHARDS=${SHARDS:-1}
BATCH=${BATCH:-32}
EXPORT=${EXPORT:-1}
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"
export TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

repo=$(pwd)
out=$(realpath -m "$OUT")
mkdir -p "$out/logs"
if [ "$EXPORT" = 1 ]; then
  echo "$(date '+%m-%d %H:%M') export to $out/cache"
  src=(--scenes "$SCENES")
  [ -n "$ROLLOUTS" ] && src+=(--rollouts "$ROLLOUTS")
  "$PYTHON" -m scripts.responsibility.export_catk --catk-root "$CATK_ROOT" "${src[@]}" --out-dir "$out" \
    > "$out/logs/export.log" 2>&1
  tail -n 1 "$out/logs/export.log"
fi

echo "$(date '+%m-%d %H:%M') responsibility with $(basename "$CKPT"), $SHARDS process(es)"
cd "$CATK_ROOT"
pids=()
for ((i = 0; i < SHARDS; i++)); do
  "$PYTHON" -m scripts.responsibility.compute_responsibility --ckpt "$CKPT" --data-root "$out/cache" \
    --out-dir "$out/run" --num-shards "$SHARDS" --shard-index "$i" --rollout-batch-size "$BATCH" \
    --query interest --neighbor-future hidden --courtesy-estimator exact "${EXTRA_ARGS[@]}" \
    > "$out/logs/shard_$i.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
cd "$repo"
[ $status -eq 0 ] || { echo "a shard failed: see $out/logs/shard_*.log; re-run to resume" >&2; exit 1; }
echo "$(date '+%m-%d %H:%M') done: $(ls "$out/run/obs" | wc -l) scenario results in $out/run/obs"
