#!/bin/bash
#
# Chorin worker (8x7 QTT grid): long unpreconditioned Chorin time evolution
#
# Submits a single-task SLURM job that runs worker.jl and then plots the saved snapshots
# with plot_snapshots.jl.
#
# Usage: bash experiments/inse/chorin_worker_8x8/submit.sh [nocond]
#
# By default, local condition-number estimates are collected during the evolution and results go
# to results/. Pass "nocond" to skip them and write to results_nocond/ instead.
#

SCRIPT_DIR="experiments/inse/chorin_worker_8x8"

NOCOND_ARG=""
RESULTS_DIR="${SCRIPT_DIR}/results"

if [ "$1" == "nocond" ]; then
    NOCOND_ARG="nocond"
    RESULTS_DIR="${SCRIPT_DIR}/results_nocond"
fi

TASKS_DIR="${RESULTS_DIR}/tasks"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "${TASKS_DIR}"
mkdir -p "${LOGS_DIR}"

# ---- Submit job array ----
ARRAY_JOB_ID=$(sbatch --parsable \
    --output="${LOGS_DIR}/task_%03a.out" \
    --error="${LOGS_DIR}/task_%03a.err" <<ARRAY_EOF
#!/bin/bash
#SBATCH --job-name=chorin_worker_8x8
#SBATCH --account=bsc21
#SBATCH --chdir=.
#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01-12:00:00
#SBATCH --qos=ngp_bsccase

module load julia/1.12.0 

echo "Task \${SLURM_ARRAY_TASK_ID} / 1 — Job \${SLURM_JOB_ID} on \${HOSTNAME}"

julia --project=experiments -t 1 --heap-size-hint=28G ${SCRIPT_DIR}/worker.jl \${SLURM_ARRAY_TASK_ID} ${NOCOND_ARG}

echo "Generating snapshot plots..."
julia --project=experiments -t 1 ${SCRIPT_DIR}/plot_snapshots.jl
ARRAY_EOF
)

if [ -n "${NOCOND_ARG}" ]; then
    echo "Mode: nocond (no condition-number estimates)"
else
    echo "Mode: default (with condition-number estimates)"
fi

echo "Submitted array job: ${ARRAY_JOB_ID} (1 task)"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
