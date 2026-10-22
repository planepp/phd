#!/bin/bash
#SBATCH --job-name=phase1-FM          # Job name
#SBATCH --account=project_2001912   # Billing project, has to be defined!
#SBATCH --partition=medium          # Job partition (queue)
#SBATCH --time=24:00:00             # Max. duration of the job
#SBATCH --nodes=2                   # Number of nodes used for the job
##SBATCH --ntasks=92                  # Number of tasks allocated
#SBATCH --ntasks-per-node=384       # Full node (ntasks)
#SBATCH --cpus-per-task=1           # Number of CPU cores allocated per task 
##SBATCH --mem-per-cpu=4000M         # Memory to reserve per CPU core
##SBATCH --output=slurm-%j.out       # Standard output of the job script
##SBATCH --mail-type=BEGIN          # Uncomment to enable mail

srun /projappl/project_2001912/vasp-env/view/bin/vasp_ncl

