#!/bin/bash -l
#SBATCH --job-name=ph1
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_id_%j.out
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/15.2.0 openmpi/5.0.10 python-data/3.12-31.03
export OMP_NUM_THREADS=1
ulimit -s unlimited

VASP_BIN="/projappl/project_2001912/vasp-env/view/bin/vasp_std"
srun "$VASP_BIN"


