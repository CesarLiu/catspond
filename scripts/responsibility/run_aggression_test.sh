#!/usr/bin/env bash
# The aggression test (responsibility/aggression_test.py): can counterfactual responsibility tell
# aggressive driving apart? Builds the labelled conditions from CAT's scenes, measures them, and
# writes the offline beta_s tables the analyses read. Each step resumes and skips what exists.
#
#   1. scene sets in VAL/scenes: DenseTNT's adversary candidates chosen by other rules
#      (export_adv_scenes.py) and the ego re-timed (retime_agent.py); CAT's own adversaries
#      go to CAT_SCENES unless they are there already
#   2. compute_responsibility for every CONFIGS x CONDITIONS into VAL/runs/<config>/<condition>
#   3. offline_safety_grid.py for every run with records -> VAL/offline/<config>__<condition>.csv.gz
#
# Then (no model needed):
#   python -m scripts.responsibility.evaluate_aggression --config paper --cat-scenes $CAT_SCENES
#   python -m scripts.responsibility.analyze_aggression_offline --config nms_ooi
#   python -m scripts.responsibility.aggression_hmm_schemes
#   python -m scripts.responsibility.plot_aggression --out-dir $VAL/figures --cat-scenes $CAT_SCENES
# SMART's runs come from run_smart.sh per scene set and import_catk_obs.py (README, "Can it judge
# aggression?").
#
# usage (from the repository root, environment active):
#   bash scripts/responsibility/run_aggression_test.sh
#   CONFIGS="paper" CONDITIONS="adv_logged sdc_logged adv_cat sdc_vs_cat" bash scripts/responsibility/run_aggression_test.sh
#
# variables (defaults):
#   VAL         output root                                   (logs/responsibility/validity)
#   SCENES      the logged scenes                             (raw_scenes_500)
#   CAT_SCENES  CAT's adversarial scenes (export_adv_scenes)  (adv_scenes/cat)
#   N           scenes 0 .. N-1                               (100)
#   CONFIGS     run settings, keys of aggression_test.CONFIGS (paper sampled_ooi nms_ooi nmsvalid_ooi nms_ooi_court)
#   CONDITIONS  keys of aggression_test.CONDITIONS            (all of them)
#   PARALLEL    runs at a time (a DenseTNT run takes a few hundred MB of GPU memory)  (4)
#   DEVICE      cuda or cpu                                   (cuda)
set -euo pipefail
VAL=${VAL:-logs/responsibility/validity}
SCENES=${SCENES:-raw_scenes_500}
CAT_SCENES=${CAT_SCENES:-adv_scenes/cat}
N=${N:-100}
CONFIGS=${CONFIGS:-"paper sampled_ooi nms_ooi nmsvalid_ooi nms_ooi_court"}
PARALLEL=${PARALLEL:-4}
DEVICE=${DEVICE:-cuda}
PY=${PYTHON:-python}
CONDITIONS=${CONDITIONS:-$($PY -c "from responsibility.aggression_test import CONDITIONS; print(' '.join(CONDITIONS))")}
mkdir -p "$VAL/scenes" "$VAL/runs" "$VAL/offline" "$VAL/joblogs"

echo "== 1. scene sets"
if [ ! -f "$CAT_SCENES/$((N - 1)).pkl" ]; then
  $PY -m scripts.responsibility.export_adv_scenes --scenes "$SCENES" --out-dir "$(dirname "$CAT_SCENES")" --n "$N" --device "$DEVICE"
fi
for spec in "fair inf 0.5 - -" "near - - 1 0.5" "near - - 0 1000" "near - - 30 0.5"; do
  read -r rule tau rho gap tol <<< "$spec"
  args=(--rule "$rule")
  [ "$tau" != - ] && args+=(--resp_threshold "$tau" --resp_avoid "$rho")
  [ "$gap" != - ] && args+=(--near_gap "$gap" --near_tol "$tol")
  $PY -m scripts.responsibility.export_adv_scenes --scenes "$SCENES" --out-dir "$VAL/scenes" --n "$N" \
      --device "$DEVICE" "${args[@]}"
done
for f in 0.5 0.7 1.3; do
  [ -f "$VAL/scenes/sdc_speed$f/0.pkl" ] || \
    $PY -m scripts.responsibility.retime_agent --scenes "$SCENES" --out "$VAL/scenes/sdc_speed$f" --factor $f --n "$N"
done

echo "== 2. runs"
jobs=$($PY - "$VAL" "$SCENES" "$CAT_SCENES" $CONFIGS -- $CONDITIONS <<'EOF'
import sys
from responsibility.aggression_test import CONDITIONS, CONFIGS, scene_dir
val, raw, cat, rest = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
configs, conditions = rest[:rest.index("--")], rest[rest.index("--") + 1:]
for cfg in configs:
    for c in conditions:
        cond = CONDITIONS[c]
        print(f"{cfg}|{c}|{scene_dir(cond.scene_set, val, raw, cat)}|{cond.agent}|{CONFIGS[cfg]}")
EOF
)
run_one() {
  IFS='|' read -r cfg cond sc ag flags <<< "$1"
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m scripts.responsibility.compute_responsibility --scenes "$sc" \
      --agent "$ag" --n "$N" $flags --out-dir "$VAL/runs/$cfg/$cond" --device "$DEVICE" \
      > "$VAL/joblogs/${cfg}__${cond}.log" 2>&1 && echo "done $cfg $cond" || echo "FAILED $cfg $cond (see joblogs)"
}
export -f run_one
export PY N VAL DEVICE
echo "$jobs" | xargs -r -d '\n' -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}

echo "== 3. offline beta_s tables"
for d in "$VAL"/runs/*/*; do
  [ -d "$d/records" ] || continue
  o="$VAL/offline/$(basename "$(dirname "$d")")__$(basename "$d").csv.gz"
  [ -f "$o" ] || echo "$d $o"
done | xargs -r -P "$PARALLEL" -L 1 $PY -m scripts.responsibility.offline_safety_grid
echo "== done: $VAL"
