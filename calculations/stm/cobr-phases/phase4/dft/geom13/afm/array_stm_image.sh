#!/bin/bash -l
#SBATCH --job-name=stm_img
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_%j.out
#SBATCH --partition=test
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128G                # total for all 8 processes; adjust (see notes)

# Makes all 820 constant-current STM images for bias pairs lo < hi on
# -2.0 ... 2.0 V (step 0.1). Images that already exist are skipped, so
# if the job hits the time limit, just submit it again.

cd "$SLURM_SUBMIT_DIR"
mkdir -p out
export OMP_NUM_THREADS=1   # one thread per python process
unset DISPLAY              # non-GUI matplotlib backend

ZS=2.8
WEIGHT=0.5

# All pairs lo < hi
pairs=()
for ((a=-20; a<=20; a++)); do
  for ((b=a+1; b<=20; b++)); do
    lo=$(awk -v n=$a 'BEGIN{printf "%.1f", n/10}')
    hi=$(awk -v n=$b 'BEGIN{printf "%.1f", n/10}')
    pairs+=( "${lo}-${hi}" )
  done
done

# Split pairs round-robin into one chunk per process; each process loads
# the data once and makes its whole chunk of images.
P=$SLURM_NTASKS
for ((p=0; p<P; p++)); do
  chunk=()
  for ((k=p; k<${#pairs[@]}; k+=P)); do chunk+=( "${pairs[k]}" ); done
  python brstm_img.py --modes stm --no-show --repeat 2 2 --savepng --separate --skip-existing \
    --plot-type height --zs "$ZS" --weights "$WEIGHT" \
    --biases "${chunk[@]}" > "out/stm_img_${SLURM_JOB_ID}_part${p}.log" 2>&1 &
done
wait

n_done=$(ls images/stm/stm_height_zs${ZS}_V*_s${WEIGHT}.png 2>/dev/null | wc -l)
echo "Done: ${n_done} / ${#pairs[@]} images"
