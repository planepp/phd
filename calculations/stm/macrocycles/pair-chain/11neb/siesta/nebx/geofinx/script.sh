#!/bin/sh
#SBATCH --job-name=pc-finx
#SBATCH --account=project_2001912
#SBATCH --time=24:00:00
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G

module purge --force
module load csc-tools/default
module load StdEnv
module load gcc/15.2.0
module load ucx/1.20.0
module load openmpi/5.0.10
module load openblas/0.3.30
module load netlib-scalapack/2.2.2
module load hdf5/1.14.6
module load fftw/3.3.10
module load cmake/3.31.11

SIESTA=/projappl/project_2001912/siesta-5.4.2/install/bin/siesta

srun $SIESTA < input.fdf > output.out  
