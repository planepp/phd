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

# 41 biases: -2.0 ... 2.0, step 0.1, run 8 at a time
for n in {0..40}; do
  awk -v n="$n" 'BEGIN{printf "%.1f\n", (n-20)/10}'
done | xargs -P "$SLURM_NTASKS" -I{} \
  bash -c 'echo "=== Bias $1 ==="; python "$0/pystm.py" --dir sample -b "$1"' "$SCRIPT" {}

echo "All biases done."
