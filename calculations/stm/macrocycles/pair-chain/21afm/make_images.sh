#!/bin/bash -l
#SBATCH --job-name=bias_img
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_%j.out
#SBATCH --partition=test
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128G                # total for all 8 processes

# Makes the constant-height / constant-current STM and/or BRSTM image for
# every bias pair lo < hi (default -2.0 ... 2.0 V, step 0.1 -> 820 images),
# then puts them in a contact sheet (images/contact_*.jpg).
#
# Image options are the same as for brstm_img.py and go straight on the
# sbatch line; anything you give overrides the defaults below:
#
#   sbatch make_images.sh                                         # defaults
#   sbatch make_images.sh --modes brstm --plot-type height --zs 3.2
#   sbatch make_images.sh --modes stm --plot-type current --iso-vals 1e-10 --weights 0.5
#   sbatch make_images.sh --modes stm brstm --plot-type height --zs 3.0
#
# Options for this script only:
#   --bias-range LO HI STEP   bias grid (default -2.0 2.0 0.1)
#   --min-range W             only pairs with hi - lo >= W (e.g. 0.2 skips the
#                             0.1 V-wide ranges)
#   --no-sheet                skip the contact sheet
#
# --bias-range and --min-range decide both which images are made and which
# go on the sheet. contact_sheet.py takes the same two options, if you want
# another sheet later from images that already exist.
#
# Use one value each for --zs / --iso-vals / --weights. Do not pass --biases
# (the pairs are generated here). Existing images are skipped, so if the job
# hits the time limit just submit it again with the same options.
#
# BRSTM: the slow interpolation step is done once per bias (spread over all
# processes) and cached in brstm_interp/. Delete that directory if the AFM
# or STM .npz data change.

DEFAULTS=(--modes stm --plot-type current --iso-vals 1e-9 --weights 1)

# ---- split script-only options from brstm_img.py options ----
BMIN=-2.0; BMAX=2.0; BSTEP=0.1
MINRANGE=
SHEET=1
ARGS=()
while (($#)); do
  case $1 in
    --bias-range) BMIN=$2; BMAX=$3; BSTEP=$4; shift 4 ;;
    --min-range)  MINRANGE=$2; shift 2 ;;
    --no-sheet)   SHEET=0; shift ;;
    *)            ARGS+=("$1"); shift ;;
  esac
done
ARGS=("${DEFAULTS[@]}" "${ARGS[@]}")   # later options win in argparse

cd "$SLURM_SUBMIT_DIR"
mkdir -p out
export OMP_NUM_THREADS=1   # one thread per python process
unset DISPLAY              # non-GUI matplotlib backend
P=${SLURM_NTASKS:-8}
CACHE=brstm_interp

# ---- bias grid: single values and all pairs lo < hi ----
# Values are written with as many decimals as STEP (at least 1), matching the
# existing file names (e.g. '-2.0-1.5').
mapfile -t singles < <(python - "$BMIN" "$BMAX" "$BSTEP" <<'PY'
import sys
lo, hi, st = sys.argv[1:]
d = max(1, len(st.split('.')[1]) if '.' in st else 0)
lo, hi, st = float(lo), float(hi), float(st)
for k in range(round((hi - lo) / st) + 1):
    print(f'{round(lo + k * st, d) + 0.0:.{d}f}')   # + 0.0 turns -0.0 into 0.0
PY
)
# Smallest index gap b - a such that hi - lo >= MINRANGE (1 = all pairs)
GAP=1
if [[ -n $MINRANGE ]]; then
  GAP=$(python -c "import math; print(max(1, math.ceil($MINRANGE / $BSTEP - 1e-6)))")
fi
pairs=()
for ((a=0; a<${#singles[@]}; a++)); do
  for ((b=a+GAP; b<${#singles[@]}; b++)); do
    pairs+=("${singles[a]}-${singles[b]}")
  done
done
echo "Options: ${ARGS[*]}"
echo "Bias grid: ${#singles[@]} values, ${#pairs[@]} pairs${MINRANGE:+ (hi - lo >= $MINRANGE)}"

# Run brstm_img.py on P processes, each with a round-robin share of the
# biases given in array $2. Each process loads the data once.
run_parallel() {
  local tag=$1; local -n items=$2; shift 2
  local p k chunk
  for ((p=0; p<P; p++)); do
    chunk=()
    for ((k=p; k<${#items[@]}; k+=P)); do chunk+=("${items[k]}"); done
    ((${#chunk[@]})) || continue
    python brstm_img.py "$@" --biases "${chunk[@]}" \
      > "out/${tag}_${SLURM_JOB_ID}_part${p}.log" 2>&1 &
  done
  wait
}

# 1) BRSTM interpolation, one bias per task (does nothing unless brstm is
#    in --modes, or if everything is already cached).
run_parallel interp singles "${ARGS[@]}" --no-show --interp-only --interp-cache "$CACHE"

# 2) Images
run_parallel img pairs "${ARGS[@]}" --no-show --savepng --separate --skip-existing \
  --interp-cache "$CACHE"

# 3) Contact sheet(s)
if ((SHEET)); then
  python contact_sheet.py "${ARGS[@]}" --bias-range "$BMIN" "$BMAX" "$BSTEP" \
    ${MINRANGE:+--min-range "$MINRANGE"}
fi
