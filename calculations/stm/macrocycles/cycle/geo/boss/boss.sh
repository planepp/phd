#!/bin/bash -l
#SBATCH --job-name=BossCycle
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_id_%j.out
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=384

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/15.2.0 openmpi/5.0.10 python-data/3.12-31.03
export OMP_NUM_THREADS=1
ulimit -s unlimited

source /projappl/project_2001912/boss-venv/bin/activate

boss op boss.in
