#!/bin/bash -l
#SBATCH --job-name=stm_ch
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_%j.out
#SBATCH --partition=test
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128G                # total for all 8 processes

cd "$SLURM_SUBMIT_DIR"
mkdir -p out
export OMP_NUM_THREADS=1   # one thread per python process
unset DISPLAY              # non-GUI matplotlib backend

IMG="$SLURM_SUBMIT_DIR/brstm_img.py"   # adjust if the script lives elsewhere
Z=2.7
BIAS=0.3
WEIGHT=0.5
NUMS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"

for n in $NUMS; do
  printf "geom%02d\n" "$n"
done | xargs -P "$SLURM_NTASKS" -I{} \
  bash -c 'cd "$1/afm" || { echo "$1: no afm dir"; exit 1; }
           mkdir -p out
           python "$0" --modes stm --no-show --savepng --separate \
             --plot-type height --zs "$2" \
             --weights "$4" --biases "$3" --repeat 3 3 \
             > "out/stm_ch_z$2_b$3.log" 2>&1
           echo "$1 done (exit $?)"' "$IMG" {} "$Z" "$BIAS" "$WEIGHT"

echo "All geometries done."
