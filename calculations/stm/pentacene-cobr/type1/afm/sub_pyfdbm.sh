#!/bin/bash -l
#SBATCH --job-name=type1
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_id_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=3G

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/15.2.0 openmpi/5.0.10 python-data/3.12-31.03

set -eo pipefail

export VASP_PP_PATH=/projappl/project_2001912
export PYTHONNOUSERSITE=1
SCRIPT="$VASP_PP_PATH/pyFDBM"
VASPCMD="srun -n 64 $VASP_PP_PATH/vasp-env/view/bin/vasp_gam"

source "$VASP_PP_PATH/env_d3/bin/activate"
python "$SCRIPT/prepare_calculation.py" --vaspcmd "$VASPCMD" -s -t -g -i input.in > prep.out
python "$SCRIPT/density_interactions.py" -i input.in > dens.out
srun -n 32 python "$SCRIPT/dftd3-py.py" -i input.in > d3.out
deactivate

#source "$VASP_PP_PATH/env_pyfdbm2/bin/activate"
#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export MKL_NUM_THREADS=1
#srun -n 64 python "$SCRIPT/relax_xy.py" -i input.in -m powell -k 0.4 > relax.out
#deactivate
