#!/bin/bash
#
# Preconditioner Benchmark: Parallel Sweep for chorin_pipeline
#
# Submits a SLURM job array and a dependent collector job.
#
# Usage: bash experiments/preconditioner_benchmarks/chorin_pipeline/submit.sh [nocond]
#
# By default, condition-number estimates are collected only for (8,8) and (10,10).
# Pass "nocond" to skip condition-number estimates for all grid sizes.
#

SCRIPT_DIR="experiments/preconditioner_benchmarks/chorin_pipeline"

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
#SBATCH --array=1-111
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01-12:00:00
#SBATCH --qos=ngp_bsccase

module load julia/1.12.0 

echo "Task \${SLURM_ARRAY_TASK_ID} / 111 — Job \${SLURM_JOB_ID} on \${HOSTNAME}"

julia --project=. -t 1 --heap-size-hint=35G ${SCRIPT_DIR}/worker.jl \${SLURM_ARRAY_TASK_ID} ${NOCOND_ARG}
ARRAY_EOF
)

if [ -n "${NOCOND_ARG}" ]; then
    echo "Mode: nocond (no condition-number estimates)"
else
    echo "Mode: default (condition-number estimates on 8x8 and 10x10 only)"
fi

echo "Submitted array job: ${ARRAY_JOB_ID} (111 tasks)"

# ---- Submit collector job ----
COLLECT_JOB_ID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} \
    --output="${LOGS_DIR}/collect.out" \
    --error="${LOGS_DIR}/collect.err" <<COLLECT_EOF
#!/bin/bash
#SBATCH --job-name=chorin_collect
#SBATCH --account=bsc21
#SBATCH --chdir=.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --qos=ngp_bsccase

module load julia/1.12.0 2>/dev/null || true

echo "Collecting results from array job..."

julia --project=. ${SCRIPT_DIR}/collect.jl ${NOCOND_ARG}
COLLECT_EOF
)

echo "Submitted collector job: ${COLLECT_JOB_ID} (depends on ${ARRAY_JOB_ID})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
