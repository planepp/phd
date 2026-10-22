#!/bin/bash -l
#SBATCH --job-name=vasp_test
#SBATCH --account=project_2001912
#SBATCH --output=%x_id_%j.out
#SBATCH --partition=test
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/11.2.0  openmpi/4.1.2  openblas/0.3.18-omp  StdEnv  netlib-scalapack/2.1.0  hdf5/1.10.7-mpi  fftw/3.3.10-mpi
module load python-data/3.10-24.04
export VASP_PP_PATH=/projappl/project_2001912
SCRIPT="$HOME/pyFDBM"
VASPCMD="srun -n 16 /projappl/project_2001912/plane_vasp.6.2.1/bin/vasp_std"

python "$SCRIPT/prepare_calculation.py" --vaspcmd "$VASPCMD" -s -t -g -i input.in > prep.out
python "$SCRIPT/density_interactions.py" -i input.in > dens.out

source "$HOME/env_pyfdbm/bin/activate"
srun -n 16 python "$SCRIPT/dftd3-py.py" -i input.in > d3.out
deactivate

source "$HOME/env_pyfdbm2/bin/activate"
srun -n 16 python "$SCRIPT/relax_xy.py" -i input.in -m powell -k 0.4 > relax.out
deactivate

