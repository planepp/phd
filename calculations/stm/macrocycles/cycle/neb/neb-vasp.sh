#!/bin/bash -l
#SBATCH --job-name=neb-vasp
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_id_%j.out
#SBATCH --partition=medium
#SBATCH --time=12:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=384

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/15.2.0 openmpi/5.0.10
export OMP_NUM_THREADS=1
ulimit -s unlimited

VASP_BIN="/projappl/project_2001912/vasp-vtst-env/view/bin/vasp_gam"
srun "$VASP_BIN"
