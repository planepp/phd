#!/bin/bash -l
#SBATCH --job-name=emol_cycle
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_id_%j.out
#SBATCH --partition=small
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=160
#SBATCH --mem-per-cpu=2G

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/15.2.0 openmpi/5.0.10 python-data/3.12-31.03
export OMP_NUM_THREADS=1
ulimit -s unlimited

VASP_BIN="/projappl/project_2001912/vasp-env/view/bin/vasp_gam"
srun "$VASP_BIN"


