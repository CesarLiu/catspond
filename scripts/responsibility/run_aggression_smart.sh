#!/usr/bin/env bash
# The aggression test (responsibility/aggression_test.py, README "Can it judge aggression?") with
# CAT-K's SMART instead of DenseTNT: for each scene set, run_smart.sh exports the first N scenes
# into catk's format and measures both objects of interest with catk's own pipeline (hidden
# neighbour futures, exact courtesy, 40 samples, 2 s, 0.5 s windows), then import_catk_obs.py
# splits the observations into VAL/runs/smart/<condition>/obs, which evaluate_aggression.py reads.
#
#   scene set     folder                        conditions (adversary, ego)
#   logged        SCENES                        adv_logged, sdc_logged
#   cat           CAT_SCENES                    adv_cat, sdc_vs_cat
#   near0_1000    VAL/scenes/near0_1000         adv_mode, sdc_vs_mode
#   near1_0.5     VAL/scenes/near1_0.5          adv_near1, sdc_vs_near1
#   fairinf_0.5   VAL/scenes/fairinf_0.5        adv_fairinf, sdc_vs_fairinf
#   near30_0.5    VAL/scenes/near30_0.5         adv_maxgap, sdc_vs_maxgap
#
# The VAL/scenes folders come from run_aggression_test.sh (or a copy of them). Each set resumes:
# finished scenarios are skipped, and a set whose N results exist is only re-imported.
# About 1 min per scene and set on an L4 with two shards (10 h for 500 scenes with one).
#
# usage (from this repository's root):
#   bash scripts/responsibility/run_aggression_smart.sh                       # every set
#   SETS="fairinf_0.5 near30_0.5" SHARDS=4 bash scripts/responsibility/run_aggression_smart.sh
# then, in this repository's environment:
#   python -m scripts.responsibility.evaluate_aggression --config smart --frame-s 0.5 \
#       --cat-scenes $CAT_SCENES --json $VAL/eval_smart.json
#
# variables (defaults): VAL (logs/responsibility/validity), SCENES (raw_scenes_500),
# CAT_SCENES (adv_scenes/cat), N (100), SETS (all six), SHARDS (2), BATCH (32),
# CATK_ROOT ($HOME/catk), PYTHON (catk's: $HOME/venvs/catk/bin/python), CKPT (run_smart.sh's)
set -euo pipefail
VAL=${VAL:-logs/responsibility/validity}
SCENES=${SCENES:-raw_scenes_500}
CAT_SCENES=${CAT_SCENES:-adv_scenes/cat}
N=${N:-100}
SETS=${SETS:-"logged cat near0_1000 near1_0.5 fairinf_0.5 near30_0.5"}
SHARDS=${SHARDS:-2}
CATK_ROOT=${CATK_ROOT:-$HOME/catk}
PYTHON=${PYTHON:-$HOME/venvs/catk/bin/python}
export CATK_ROOT PYTHON
repo=$(pwd)
val=$(realpath -m "$VAL")

declare -A ADV=([logged]=adv_logged [cat]=adv_cat [near0_1000]=adv_mode [near1_0.5]=adv_near1
                [fairinf_0.5]=adv_fairinf [near30_0.5]=adv_maxgap)
declare -A SDC=([logged]=sdc_logged [cat]=sdc_vs_cat [near0_1000]=sdc_vs_mode [near1_0.5]=sdc_vs_near1
                [fairinf_0.5]=sdc_vs_fairinf [near30_0.5]=sdc_vs_maxgap)
for s in $SETS; do
  case $s in
    logged) folder=$SCENES ;;
    cat) folder=$CAT_SCENES ;;
    *) folder=$VAL/scenes/$s ;;
  esac
  folder=$(realpath -m "$folder")
  [ -d "$folder" ] || { echo "missing scene folder $folder (run_aggression_test.sh step 1)" >&2; exit 1; }
  out=$val/smart/$s
  done_n=0
  [ -d "$out/run/obs" ] && done_n=$(find "$out/run/obs" -name '*.pkl' | wc -l)
  if [ "$done_n" -lt "$N" ]; then
    echo "== $s: SMART on $N scenes of $folder"
    SCENES=$folder N=$N OUT=$out SHARDS=$SHARDS BATCH=${BATCH:-32} bash scripts/responsibility/run_smart.sh
  fi
  echo "== $s: import -> runs/smart/${ADV[$s]}, ${SDC[$s]}"
  (cd "$CATK_ROOT" && TF_CPP_MIN_LOG_LEVEL=3 "$PYTHON" "$repo/scripts/responsibility/import_catk_obs.py" \
      "$out/run" "$folder" "$val/runs/smart/${ADV[$s]}" "$val/runs/smart/${SDC[$s]}" --n "$N")
done
echo "== done; next: python -m scripts.responsibility.evaluate_aggression --config smart --frame-s 0.5 --cat-scenes $CAT_SCENES"
