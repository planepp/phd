#!/bin/bash -l
#SBATCH --job-name=stm
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_%j.out
#SBATCH --partition=test
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

cd "$SLURM_SUBMIT_DIR"

export VASP_PP_PATH=/projappl/project_2001912
SCRIPT="$VASP_PP_PATH/pyFDBM"
export OMP_NUM_THREADS=1   # one thread per python process

BIAS=0.3
NUMS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"

# one geometry per python process, 8 at a time
for n in $NUMS; do
  printf "geom%02d\n" "$n"
done | xargs -P "$SLURM_NTASKS" -I{} \
  bash -c 'cd "$1/afm" || exit 1
           echo "=== $1 (bias $2) ==="
           python "$0/pystm.py" --dir sample -b "$2" > "out/stm_b$2.log" 2>&1
           echo "--- $1 done (exit $?) ---"' "$SCRIPT" {} "$BIAS"

echo "All geometries done."
