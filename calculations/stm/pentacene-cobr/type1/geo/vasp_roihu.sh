#!/bin/bash -l
#SBATCH --job-name=pent1
#SBATCH --account=project_2001912
#SBATCH --output=out/%x_id_%j.out
#SBATCH --partition=medium
#SBATCH --time=36:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=384

cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/15.2.0 openmpi/5.0.10 python-data/3.12-31.03
export OMP_NUM_THREADS=1
ulimit -s unlimited

VASP_BIN="/projappl/project_2001912/vasp-env/view/bin/vasp_std"

# ---------------- settings ----------------
WALLTIME_H=36        # must match --time above
SAFETY_MIN=60        # don't start a new attempt with less than this left
STALE_LIMIT=1800     # seconds without OUTCAR/OSZICAR update = hung
MAX_ATTEMPTS=20
# ------------------------------------------

JOB_START=$(date +%s)
JOB_END=$(( JOB_START + WALLTIME_H*3600 ))

newest_output() {
    local newest=0 t f
    for f in OUTCAR OSZICAR; do
        if [ -f "$f" ]; then
            t=$(stat -c %Y "$f")
            [ "$t" -gt "$newest" ] && newest=$t
        fi
    done
    echo "$newest"
}

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    left=$(( JOB_END - $(date +%s) ))
    if [ "$left" -lt $(( SAFETY_MIN*60 )) ]; then
        echo "Less than ${SAFETY_MIN} min left, not starting attempt $attempt."
        break
    fi

    echo "=== Attempt $attempt started $(date) ==="
    start_attempt=$(date +%s)
    srun "$VASP_BIN" &
    VASP_PID=$!
    step=$((attempt - 1))          # Slurm step id of this srun
    hung=0

    while kill -0 "$VASP_PID" 2>/dev/null; do
        sleep 300
        newest=$(newest_output)
        [ "$newest" -lt "$start_attempt" ] && newest=$start_attempt
        if [ $(( $(date +%s) - newest )) -gt "$STALE_LIMIT" ]; then
            echo "WATCHDOG: no output for >${STALE_LIMIT}s at $(date), killing step $step"
            hung=1
            scancel --quiet "${SLURM_JOB_ID}.${step}"
            sleep 60
            kill -9 "$VASP_PID" 2>/dev/null
            break
        fi
    done
    wait "$VASP_PID" 2>/dev/null

    # keep the output of every attempt
    for f in OUTCAR OSZICAR; do
        [ -f "$f" ] && cp "$f" "${f}_attempt${attempt}"
    done

    if [ "$hung" -eq 0 ]; then
        echo "VASP finished normally at $(date)."
        grep -q "reached required accuracy" OUTCAR && echo "Relaxation CONVERGED."
        break
    fi

    # ---- prepare restart after a hang ----
    if [ -s CONTCAR ] && [ "$(wc -l < CONTCAR)" -ge "$(wc -l < POSCAR)" ]; then
        cp POSCAR "POSCAR_attempt${attempt}"
        cp CONTCAR POSCAR
        echo "Restarting from CONTCAR."
    else
        echo "CONTCAR empty/incomplete, restarting from the same POSCAR."
    fi
    rm -f WAVECAR CHGCAR                       # never written cleanly after a kill
    sed -i 's/^\s*ISTART\s*=.*/ISTART = 0/I' INCAR
    grep -qi '^\s*ISTART' INCAR || echo "ISTART = 0" >> INCAR
done

echo "Job script finished at $(date) after $attempt attempt(s)."
