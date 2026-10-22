#!/bin/bash
#SBATCH --account=project_2001912
#SBATCH -p medium # test - for testing 1h ; medium - up to 20 nodes/36 hours ; large - 20-200 nodes/36 hours #
#SBATCH --time=36:00:00      # dd-hh:mm:ss
#SBATCH -J BossRotor        # name 
#SBATCH -o sbatch-%j.out # where the outputs & errors are written
#SBATCH --nodes=6                  # N nodes to run (N x 64 = n); max 192 ; max debug 48
#SBATCH --ntasks-per-node=128                # n processes to run (N x 64 = n); max 192 ; max debug 48
##SBATCH --cpus-per-task=4

module load python-data/3.10-24.04 vasp/6.4.3

source /scratch/project_2001912/boss/bin/activate
boss op boss.in

