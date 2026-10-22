#!/bin/bash -l
#SBATCH --job-name=pair-chain
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_id_%j.out
#SBATCH --time=36:00:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem-per-cpu=6G      # 64 × 6 GB = 384 GB

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/15.2.0 openmpi/5.0.10 python-data/3.12-31.03

set -e   # stop at the first step that fails

export VASP_PP_PATH=/projappl/project_2001912
SCRIPT="$VASP_PP_PATH/pyFDBM"
VASPCMD="srun -K -n 64 $VASP_PP_PATH/vasp-env/view/bin/vasp_gam"

python "$SCRIPT/prepare_calculation.py" --vaspcmd "$VASPCMD" -s -t -g -i input.in > prep.out
python "$SCRIPT/density_interactions.py" -i input.in > dens.out

source "$VASP_PP_PATH/env_pyfdbm/bin/activate"
srun -K -n 32 python "$SCRIPT/dftd3-py.py" -i input.in > d3.out
deactivate

source "$VASP_PP_PATH/env_pyfdbm2/bin/activate"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun -K -n 64 python "$SCRIPT/relax_xy.py" -i input.in -m powell -k 0.4 > relax.out
deactivate
