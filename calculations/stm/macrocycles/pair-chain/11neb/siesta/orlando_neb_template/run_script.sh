#!/bin/sh
#SBATCH --job-name=shi
#SBATCH --account=project_2001912
#SBATCH --time=0:15:00
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=8
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

#-----------------------------------
#  run_script for Running Siesta
#  A.Akhtar 23/3/2021
#-----------------------------------
echo 'Flos Lib Loaded'
FLOS_PLACE=/projappl/project_2001912/flos
#export LUA_PATH="$HOME/Projects/flos/?.lua;$HOME/Projects/flos/?/init.lua;$LUA_PATH;;"
export LUA_PATH="${FLOS_PLACE}/?.lua;${FLOS_PLACE}/?/init.lua;$LUA_PATH;;"

SIESTA=/projappl/project_2001912/siesta-5.4.2/install/bin/siesta
folder_name=results
#np=8

#-------------------------------------
if [ -d $folder_name ] ; then
  echo "Directory $foldername exists..."
  exit
fi

mkdir $folder_name
cp *.psf *.psml *.DM* *.XV *.fdf *.lua *.xyz $folder_name
cd $folder_name
#mpirun -n $np --use-hwthread-cpus $SIESTA < input.fdf > output.out  &
srun $SIESTA < input.fdf > output.out  
#---------
#cd ..
