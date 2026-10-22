#!/bin/bash -l
#SBATCH --job-name=geom02
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_%j.out
#SBATCH --partition=test
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32      # parallel relaxations
#SBATCH --cpus-per-task=4         # torch threads per relaxation (32 x 4 = 128 cores)
#SBATCH --mem=128G

# Runs mlip.py in every subfolder that contains a POSCAR.
# Folders that already have a result.json are skipped, so if the job
# hits the time limit, just submit it again.

cd "$SLURM_SUBMIT_DIR"
mkdir -p out
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Folders still to do
todo=()
for d in */; do
  d=${d%/}
  [[ -f "$d/POSCAR" ]] || continue
  [[ -f "$d/result.json" ]] && ! grep -q '"error"' "$d/result.json" && continue
  todo+=( "$d" )
done
echo "${#todo[@]} folders to relax"
[[ ${#todo[@]} -eq 0 ]] && exit 0

# Split the folders over the processes; each one loads the MACE model once
P=$SLURM_NTASKS
(( P > ${#todo[@]} )) && P=${#todo[@]}
for ((p=0; p<P; p++)); do
  chunk=()
  for ((k=p; k<${#todo[@]}; k+=P)); do chunk+=( "${todo[k]}" ); done
  python mlip.py "${chunk[@]}" --skip-done --threads "$SLURM_CPUS_PER_TASK" \
    > "out/mace_${SLURM_JOB_ID}_part${p}.log" 2>&1 &
done
wait

n_ok=$(grep -L '"error"' */result.json 2>/dev/null | wc -l)
echo "Done: ${n_ok} folders relaxed"
