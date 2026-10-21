#!/bin/bash
#
# Linsolve preconditioner SPD benchmark: a single job that runs every section of worker.jl and writes all results to
# results/linsolve_spd.jld2. Cases already in that file are skipped, so a timed-out job can simply be resubmitted.
#
# Usage: bash experiments/paper/linsolve_spd/submit.sh [section1,section2,...]
#
# Sections: chorin_linsolve_dense, chorin_truncinv_dense, chorin_linsolve_large (default: all, in this order).
# Follow the progress with: tail -f experiments/paper/linsolve_spd/results/logs/linsolve_spd_<jobid>.out
#

SCRIPT_DIR="experiments/paper/linsolve_spd"

# Cluster settings: override from the environment, e.g. SLURM_ACCOUNT=<project> bash experiments/paper/linsolve_spd/submit.sh
SLURM_ACCOUNT="${SLURM_ACCOUNT:-bsc21}"
SLURM_QOS="${SLURM_QOS:-ngp_bsccase}"
JULIA_MODULE="${JULIA_MODULE:-julia/1.12.0}"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "${LOGS_DIR}"

SECTIONS_ENV=""
if [ -n "$1" ]; then
    SECTIONS_ENV="export LINSOLVE_SPD_SECTIONS=$1"
fi

JOB_ID=$(sbatch --parsable \
    --output="${LOGS_DIR}/linsolve_spd_%j.out" \
    --error="${LOGS_DIR}/linsolve_spd_%j.err" <<JOB_EOF
#!/bin/bash
#SBATCH --job-name=linsolve_spd
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --chdir=.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01-12:00:00
#SBATCH --qos=${SLURM_QOS}

module load ${JULIA_MODULE}

# 2 GB per core on MN5: leave 20% headroom for BLAS buffers and memory the GC does not track
if [ -n "\${SLURM_MEM_PER_NODE}" ]; then
    TASK_MEM_MB=\${SLURM_MEM_PER_NODE}
else
    TASK_MEM_MB=\$(( \${SLURM_MEM_PER_CPU:-2000} * SLURM_CPUS_PER_TASK ))
fi
HEAP_HINT_MB=\$(( TASK_MEM_MB * 80 / 100 ))

${SECTIONS_ENV}

echo "Job \${SLURM_JOB_ID} on \${HOSTNAME}, task memory \${TASK_MEM_MB} MB, heap hint \${HEAP_HINT_MB} MB"
echo "Sections: \${LINSOLVE_SPD_SECTIONS:-all}"

julia --project=experiments -t 1 --heap-size-hint=\${HEAP_HINT_MB}M ${SCRIPT_DIR}/worker.jl
JOB_EOF
)

echo "Submitted job: ${JOB_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOGS_DIR}/linsolve_spd_${JOB_ID}.out"
