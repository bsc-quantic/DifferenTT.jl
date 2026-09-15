#!/bin/bash
#
# Preconditioner Benchmark: Parallel Sweep for chorin_pipeline
#
# Submits a SLURM job array and a dependent collector job.
#
# Usage: bash experiments/inse/chorin_worker_12x12/submit.sh [nocond]
#
# By default, condition-number estimates are collected only for (8,8) and (10,10).
# Pass "nocond" to skip condition-number estimates for all grid sizes.
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
#SBATCH --job-name=chorin_pipeline
#SBATCH --account=bsc21
#SBATCH --chdir=.
#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01-12:00:00
#SBATCH --qos=ngp_bsccase

module load julia/1.12.0 

echo "Task \${SLURM_ARRAY_TASK_ID} / 1 — Job \${SLURM_JOB_ID} on \${HOSTNAME}"

julia --project=. -t 1 --heap-size-hint=28G ${SCRIPT_DIR}/worker.jl \${SLURM_ARRAY_TASK_ID} ${NOCOND_ARG}

echo "Generating snapshot plots..."
julia --project=. -t 1 ${SCRIPT_DIR}/plot_snapshots.jl
ARRAY_EOF
)

if [ -n "${NOCOND_ARG}" ]; then
    echo "Mode: nocond (no condition-number estimates)"
else
    echo "Mode: default (condition-number estimates on 8x8 and 10x10 only)"
fi

echo "Submitted array job: ${ARRAY_JOB_ID} (1 task)"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
