#!/bin/bash

SCRIPT_DIR="experiments/preconditioner_benchmarks/tt_poisson_global"

# GC memory ballast (GB); see chorin_gc_report.md. Override with GC_BALLAST_GB=0 to disable.
GC_BALLAST_GB="${GC_BALLAST_GB:-7}"

RESULTS_DIR="${SCRIPT_DIR}/results"

TASKS_DIR="${RESULTS_DIR}/tasks"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "${TASKS_DIR}"
mkdir -p "${LOGS_DIR}"

# ---- Submit job array ----
ARRAY_JOB_ID=$(sbatch --parsable \
    --output="${LOGS_DIR}/task_%03a.out" \
    --error="${LOGS_DIR}/task_%03a.err" <<ARRAY_EOF
#!/bin/bash
#SBATCH --job-name=tt_glob
#SBATCH --account=bsc21
#SBATCH --chdir=.
# #SBATCH --array=1-88
#SBATCH --array=1-50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01-12:00:00
#SBATCH --qos=ngp_bsccase

module load julia/1.12.0 

if [ -n "\${SLURM_MEM_PER_NODE}" ]; then
    TASK_MEM_MB=\${SLURM_MEM_PER_NODE}
else
    TASK_MEM_MB=\$(( \${SLURM_MEM_PER_CPU:-2000} * SLURM_CPUS_PER_TASK ))
fi
HEAP_HINT_MB=\$(( TASK_MEM_MB * 80 / 100 ))

export GC_BALLAST_GB="${GC_BALLAST_GB}"

echo "Task \${SLURM_ARRAY_TASK_ID} / 50 — Job \${SLURM_JOB_ID} on \${HOSTNAME}"
echo "GC ballast ${GC_BALLAST_GB} GB, task memory \${TASK_MEM_MB} MB, heap hint \${HEAP_HINT_MB} MB"

julia --project=. -t 1 --heap-size-hint=\${HEAP_HINT_MB}M ${SCRIPT_DIR}/worker.jl \${SLURM_ARRAY_TASK_ID}
ARRAY_EOF
)

echo "Submitted array job: ${ARRAY_JOB_ID} (50 tasks)"

# ---- Submit collector job ----
COLLECT_JOB_ID=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB_ID} \
    --output="${LOGS_DIR}/collect.out" \
    --error="${LOGS_DIR}/collect.err" <<COLLECT_EOF
#!/bin/bash
#SBATCH --job-name=tt_collect
#SBATCH --account=bsc21
#SBATCH --chdir=.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --qos=ngp_bsccase

module load julia/1.12.0 2>/dev/null || true

echo "Collecting results from array job..."

julia --project=. ${SCRIPT_DIR}/collect.jl
COLLECT_EOF
)

echo "Submitted collector job: ${COLLECT_JOB_ID} (depends on ${ARRAY_JOB_ID})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
