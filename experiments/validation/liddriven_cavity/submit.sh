#!/bin/bash
#
# Lid-driven cavity worker (7x7 QTT grid, 128² interior nodes): long unpreconditioned ψ–ω time evolution
#
# Submits a single-task SLURM job that runs worker.jl, then plots the saved snapshots with
# plot_snapshots.jl and compares the last snapshot with Ghia et al. (1982) with ghia_comparison.jl.
#
# Usage: [CAVITY_LIDS=1|2] bash experiments/validation/liddriven_cavity/submit.sh [nocond]
#
# By default, local condition-number estimates are collected during the evolution and results go
# to results/. Pass "nocond" to skip them and write to results_nocond/ instead.
#
# With CAVITY_LIDS=2 (doubly-driven cavity) the folders get the _doublelid suffix and the Ghia comparison is
# skipped (it has no doubly-driven reference data).
#

SCRIPT_DIR="experiments/validation/liddriven_cavity"

# Number of moving lids: 1 (lid-driven cavity) or 2 (doubly-driven cavity, bottom lid moving in -x).
# Override with CAVITY_LIDS=2; the results then go to the *_doublelid folders.
CAVITY_LIDS="${CAVITY_LIDS:-1}"
if [ "${CAVITY_LIDS}" != "1" ] && [ "${CAVITY_LIDS}" != "2" ]; then
    echo "CAVITY_LIDS must be 1 or 2, got ${CAVITY_LIDS}"
    exit 1
fi
RESULTS_SUFFIX=""
if [ "${CAVITY_LIDS}" == "2" ]; then
    RESULTS_SUFFIX="_doublelid"
fi

NOCOND_ARG=""
RESULTS_DIR="${SCRIPT_DIR}/results${RESULTS_SUFFIX}"

if [ "$1" == "nocond" ]; then
    NOCOND_ARG="nocond"
    RESULTS_DIR="${SCRIPT_DIR}/results_nocond${RESULTS_SUFFIX}"
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
#SBATCH --job-name=liddriven_cavity${RESULTS_SUFFIX}
#SBATCH --account=bsc21
#SBATCH --chdir=.
#SBATCH --array=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01-12:00:00
#SBATCH --qos=ngp_bsccase

module load julia/1.12.0 

export CAVITY_LIDS="${CAVITY_LIDS}"

echo "Task \${SLURM_ARRAY_TASK_ID} / 1 — Job \${SLURM_JOB_ID} on \${HOSTNAME}"

julia --project=experiments -t 1 --heap-size-hint=28G ${SCRIPT_DIR}/worker.jl \${SLURM_ARRAY_TASK_ID} ${NOCOND_ARG}

echo "Generating snapshot plots..."
julia --project=experiments -t 1 ${SCRIPT_DIR}/plot_snapshots.jl ${NOCOND_ARG}

if [ "${CAVITY_LIDS}" == "1" ]; then
    echo "Comparing with Ghia et al. (1982)..."
    julia --project=experiments -t 1 ${SCRIPT_DIR}/ghia_comparison.jl ${NOCOND_ARG}
fi
ARRAY_EOF
)

if [ -n "${NOCOND_ARG}" ]; then
    echo "Mode: nocond (no condition-number estimates)"
else
    echo "Mode: default (with condition-number estimates)"
fi

echo "Lids: ${CAVITY_LIDS}"
echo "Submitted array job: ${ARRAY_JOB_ID} (1 task)"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
