#!/bin/bash
#
# Preconditioner Benchmark: Parallel Sweep for liddriven_pipeline_globalcg
#
# Submits a SLURM job array and a dependent collector job.
#
# Usage: [CAVITY_LIDS=1|2] bash experiments/fai_benchmarks/liddriven_pipeline_globalcg/submit.sh
#
# With CAVITY_LIDS=2 (doubly-driven cavity) the results folder gets the _doublelid suffix.
#

SCRIPT_DIR="experiments/fai_benchmarks/liddriven_pipeline_globalcg"

# Cluster settings: override from the environment, e.g. SLURM_ACCOUNT=<project> bash experiments/fai_benchmarks/liddriven_pipeline_globalcg/submit.sh
SLURM_ACCOUNT="${SLURM_ACCOUNT:-bsc21}"
SLURM_QOS="${SLURM_QOS:-ngp_bsccase}"
JULIA_MODULE="${JULIA_MODULE:-julia/1.12.0}"

# GC memory ballast (GB); see experiments/common/gc_ballast.jl. Override with GC_BALLAST_GB=0 to disable.
GC_BALLAST_GB="${GC_BALLAST_GB:-7}"

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

RESULTS_DIR="${SCRIPT_DIR}/results${RESULTS_SUFFIX}"

TASKS_DIR="${RESULTS_DIR}/tasks"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "${TASKS_DIR}"
mkdir -p "${LOGS_DIR}"

# ---- Submit job array ----
ARRAY_JOB_ID=$(sbatch --parsable \
    --output="${LOGS_DIR}/task_%03a.out" \
    --error="${LOGS_DIR}/task_%03a.err" <<ARRAY_EOF
#!/bin/bash
#SBATCH --job-name=liddriven_globalcg${RESULTS_SUFFIX}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --chdir=.
# #SBATCH --array=1-111
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01-12:00:00
#SBATCH --qos=${SLURM_QOS}
#SBATCH --exclusive

module load ${JULIA_MODULE}

if [ -n "\${SLURM_MEM_PER_NODE}" ]; then
    TASK_MEM_MB=\${SLURM_MEM_PER_NODE}
else
    TASK_MEM_MB=\$(( \${SLURM_MEM_PER_CPU:-2000} * SLURM_CPUS_PER_TASK ))
fi
HEAP_HINT_MB=\$(( TASK_MEM_MB * 80 / 100 ))

export GC_BALLAST_GB="${GC_BALLAST_GB}"
export CAVITY_LIDS="${CAVITY_LIDS}"

# echo "Task \${SLURM_ARRAY_TASK_ID} / 111 — Job \${SLURM_JOB_ID} on \${HOSTNAME}"
echo "Task \${SLURM_ARRAY_TASK_ID} / 10 — Job \${SLURM_JOB_ID} on \${HOSTNAME}"
echo "GC ballast ${GC_BALLAST_GB} GB, task memory \${TASK_MEM_MB} MB, heap hint \${HEAP_HINT_MB} MB"

julia --project=experiments -t 1 --heap-size-hint=\${HEAP_HINT_MB}M ${SCRIPT_DIR}/worker.jl \${SLURM_ARRAY_TASK_ID}
ARRAY_EOF
)

# echo "Submitted array job: ${ARRAY_JOB_ID} (111 tasks)"
echo "Lids: ${CAVITY_LIDS}"
echo "Submitted array job: ${ARRAY_JOB_ID} (10 tasks)"

# ---- Submit collector job ----
COLLECT_JOB_ID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} \
    --output="${LOGS_DIR}/collect.out" \
    --error="${LOGS_DIR}/collect.err" <<COLLECT_EOF
#!/bin/bash
#SBATCH --job-name=liddriven_collect_globalcg
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --chdir=.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --qos=${SLURM_QOS}

module load ${JULIA_MODULE} 2>/dev/null || true

export CAVITY_LIDS="${CAVITY_LIDS}"

echo "Collecting results from array job..."

julia --project=experiments ${SCRIPT_DIR}/collect.jl
COLLECT_EOF
)

echo "Submitted collector job: ${COLLECT_JOB_ID} (depends on ${ARRAY_JOB_ID})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
