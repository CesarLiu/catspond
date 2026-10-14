#!/usr/bin/env bash
# Packs what the CAT-K / SMART experiments need on another server and git does not carry
# (responsibility/CATK_H200.md): the scenes, SMART's checkpoint, and the inputs of each
# experiment. Code comes from git (this repository's counterfactual-responsibility branch and
# CesarLiu/catk4test); nothing here is published (WOMD's terms cover the scenes and the model).
#
# The tar has three top folders, extracted separately on the server:
#   repo/   into this repository's root     tar xf PACK -C <repo> --strip-components=1 repo
#   catk/   into the catk checkout           tar xf PACK -C <catk> --strip-components=1 catk
#   womd/   anywhere (CAT's 497 WOMD v1.2.1 scenarios as one tfrecord)
#
# Parts (PARTS, default "core aggression compare rollouts v121"):
#   core         raw_scenes_500/ (CAT's 500 scenes, 317 MB); catk/ckpts/clsft_E9.ckpt (68 MB)
#   aggression   adv_scenes/cat/ (CAT's adversarial scenes, 317 MB) and
#                logs/responsibility/validity/scenes/ (the other scene sets, 452 MB)
#   compare      windows*.csv and config.json of the DenseTNT runs logs/responsibility/{sdc,adv}, and
#                evaluate_aggression's DenseTNT results VAL/eval_dtnt_s5.json (and eval_paper.json)
#                where they exist, for plot_aggression.py after the SMART aggression test
#   rollouts     rollouts/replay/ (the replayed logged ego against four adversaries, 3 MB)
#   v121         womd/cat_scenarios_v121.tfrecord (411 MB; extract_womd_scenarios.py)
#   densetnt     advgen/pretrained/densetnt.bin (120 MB): to rebuild scene sets or DenseTNT runs there
#   results      the SMART results so far: logs/catk/ and logs/responsibility/validity/{runs/smart,smart}
#
# usage (from this repository's root):
#   DRY=1 bash scripts/responsibility/pack_catk_h200.sh                 # list the parts and sizes
#   OUT=/tmp/catk_h200.tar bash scripts/responsibility/pack_catk_h200.sh
#   PARTS="core aggression" bash scripts/responsibility/pack_catk_h200.sh
#
# Source locations (defaults relative to this repository's root; override where data lives elsewhere):
#   SCENES (raw_scenes_500)  CAT_SCENES (adv_scenes/cat)  VAL (logs/responsibility/validity)
#   DTNT_RUNS (logs/responsibility)  ROLLOUTS (rollouts/replay)  CATK_LOGS (logs/catk)
#   TFRECORD ($HOME/womd_v1_2_1/cat_extract/cat_scenarios_v121.tfrecord)
#   DENSETNT (advgen/pretrained/densetnt.bin)  CATK_ROOT ($HOME/catk)
set -euo pipefail
PARTS=${PARTS:-"core aggression compare rollouts v121"}
OUT=${OUT:-catk_h200.tar}
SCENES=${SCENES:-raw_scenes_500}
CAT_SCENES=${CAT_SCENES:-adv_scenes/cat}
VAL=${VAL:-logs/responsibility/validity}
DTNT_RUNS=${DTNT_RUNS:-logs/responsibility}
ROLLOUTS=${ROLLOUTS:-rollouts/replay}
CATK_LOGS=${CATK_LOGS:-logs/catk}
TFRECORD=${TFRECORD:-$HOME/womd_v1_2_1/cat_extract/cat_scenarios_v121.tfrecord}
DENSETNT=${DENSETNT:-advgen/pretrained/densetnt.bin}
CATK_ROOT=${CATK_ROOT:-$HOME/catk}

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
link() {  # link SRC as DEST inside the staging tree
  [ -e "$1" ] || { echo "missing: $1" >&2; exit 1; }
  mkdir -p "$stage/$(dirname "$2")"
  ln -s "$(realpath "$1")" "$stage/$2"
}
for part in $PARTS; do
  case $part in
    core)
      link "$SCENES" repo/raw_scenes_500
      link "$CATK_ROOT/ckpts/clsft_E9.ckpt" catk/ckpts/clsft_E9.ckpt ;;
    aggression)
      link "$CAT_SCENES" repo/adv_scenes/cat
      link "$VAL/scenes" repo/logs/responsibility/validity/scenes ;;
    compare)
      for a in sdc adv; do
        for f in "$DTNT_RUNS/$a"/windows*.csv "$DTNT_RUNS/$a/config.json"; do
          link "$f" "repo/logs/responsibility/$a/$(basename "$f")"
        done
      done
      for j in eval_dtnt_s5.json eval_paper.json; do
        [ -e "$VAL/$j" ] && link "$VAL/$j" "repo/logs/responsibility/validity/$j"
      done ;;
    rollouts) link "$ROLLOUTS" repo/rollouts/replay ;;
    v121) link "$TFRECORD" womd/cat_scenarios_v121.tfrecord ;;
    densetnt) link "$DENSETNT" repo/advgen/pretrained/densetnt.bin ;;
    results)
      link "$CATK_LOGS" repo/logs/catk
      link "$VAL/runs/smart" repo/logs/responsibility/validity/runs/smart
      link "$VAL/smart" repo/logs/responsibility/validity/smart ;;
    *) echo "unknown part $part" >&2; exit 1 ;;
  esac
done
(cd "$stage" && find . -mindepth 2 -maxdepth 6 -type l | sort | while read -r l; do
   printf '%8s  %s\n' "$(du -shL "$l" | cut -f1)" "${l#./}"; done)
echo "total: $(du -shL "$stage" | cut -f1)"
[ "${DRY:-0}" = 1 ] && exit 0
# shellcheck disable=SC2046
tar -C "$stage" -chf "$OUT" $(ls "$stage")
echo "wrote $OUT ($(du -sh "$OUT" | cut -f1))"
