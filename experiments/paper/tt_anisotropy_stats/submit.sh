#!/bin/bash
#
# TT Anisotropy Benchmark (local ResidualALS), statistics variant: Parallel Sweep
#
# The 132 tasks of tt_anisotropy/submit.sh (12 (N, F) cases x 11 methods), each repeated, one
# result file per (task, repeat): tasks/task_<id>_rep_<r>.jld2. The problem is built once per job
# (and shared by every method of a block); each repeat draws a fresh initial guess (and, for FAI,
# a fresh randomized preconditioner). Submits a SLURM job array and a dependent collector job.
# Every submission queues a collector; the one that runs last sees every finished unit.
#
# Usage: bash experiments/paper/tt_anisotropy_stats/submit.sh [nocond]   (μ = ‖A‖_∞, as in the paper)
#        FAI_MU_RULE=fixed bash experiments/paper/tt_anisotropy_stats/submit.sh nocond   (μ = 1e7, earlier drafts)
#
# Pass "nocond" for the timing runs (no condition-number estimation overhead): plot_paper.jl
# reads its timings and iterations from results_nocond_mugersh (results_nocond with FAI_MU_RULE=fixed).
#
# --- Layout: why one job = one block -------------------------------------------------------
# About 1 allocation in 6 runs 24-37% slower for its whole lifetime (measured on the Chorin
# benchmarks, in shared AND exclusive runs, so --exclusive does not help). With the old per-task
# layout (all repeats of one configuration in one job), such a slowdown shifted that configuration
# as a whole: it did not average out over its repeats, it was invisible in the error bars, and the
# "paired" speedups compared a baseline and a configuration that ran on different nodes.
#
# BENCH_LAYOUT=block (default): array index -> (group, repeat), and the job runs EVERY method of
# that group at that repeat index, in a seeded random order (no method is systematically first or
# last), all with the same per-repeat seed. So
#   * speedups (baseline / method at the same repeat) are ratios measured on the same node, and a
#     node effect cancels in them;
#   * different repeats run in different jobs, so the absolute times sample node-to-node
#     variability, and the collector estimates it directly (block effects in
#     variability_summary.md).
# Array indices are repeat-major (1..NGROUPS = repeat 1 of every group, ...), so the jobs that
# start first give a usable early dataset. Do not bundle several repeats of a group into one job:
# that brings back exactly the problem above.
# BENCH_LAYOUT=task: the old layout (array index = task id, all repeats inside the job).
#
# Results of the two layouts must not share a tasks/ folder (same file names); the LAYOUT marker
# file enforces this. Before the first block run, move the old per-task results aside:
#     mv <results dir> <script dir>/olddata_pertask/
#
# Repeat configuration (environment variables, exported to the workers and echoed in the job log):
#   BENCH_LAYOUT         block | task                                     (default block)
#   BENCH_N_REPEATS      block: repeat indices covered by this array; task: repeats per job (default 10)
#   BENCH_REPEAT_OFFSET  global index of the first repeat - 1           (default 0)
#   BENCH_BASE_SEED      base of the per-repeat seeds                     (default 20260917)
#   BENCH_TIME_BUDGET_S  stop STARTING units past this wall time          (default 158400 = 44 h)
# More repeats later, without touching the existing ones (seeds use the GLOBAL repeat index, so
# pairing is preserved): BENCH_REPEAT_OFFSET=10 BENCH_N_REPEATS=5 bash .../submit.sh nocond
#
# --- Wall clock ----------------------------------------------------------------------------
# A block costs the sum of its methods' per-repeat times:
#     N=10: F=1,2,4,10,30,100 blocks  0.4, 0.7, 0.8, 0.9, 1.5, 7.5 h
#     N=20: F=1,2,4,10,30,100 blocks  0.7, 0.9, 8.9, 2.3, 3.2, 9.5 h
# (mean per-repeat times of the Oct 2026 nocond run). 10 repeats are 120 jobs, ~370 job-hours.
# Sub-second solves are sensitive to residual JIT cost in the first unit of a job, which in this
# layout is a different random method in every block: see BENCH_DISCARD_FIRST_REPEAT in collect.jl.
# The guard stops a block that is running out of time before it starts a unit that is not expected
# to fit; every finished unit is already on disk. Run the SAME command again: units whose file
# exists are skipped, so it only fills the gaps. The refilled units run in a different job (usually
# a different node); the collector counts the pairs that are still same-job (n_pairs_samejob) and
# BENCH_SAMEJOB_PAIRS_ONLY=1 restricts the speedups to those.
#
# A longer --time may mean a longer queue wait.
#
#

SCRIPT_DIR="experiments/paper/tt_anisotropy_stats"

# Cluster settings: override from the environment, e.g. SLURM_ACCOUNT=<project> bash experiments/paper/tt_anisotropy_stats/submit.sh
SLURM_ACCOUNT="${SLURM_ACCOUNT:-bsc21}"
SLURM_QOS="${SLURM_QOS:-ngp_bsccase}"
JULIA_MODULE="${JULIA_MODULE:-julia/1.12.0}"

# Must equal length(build_task_list()) in worker.jl:
# len(system_dims) * len(floor_values) * (1 + len(sinc_M_values) + len(fai_configs) + len(fai_L0_configs))
# = 2 * 6 * (1 + 5 + 4 + 1)
NTASKS=132
# Number of block groups, must equal length(task_groups(task_group.(build_task_list()))) in
# worker.jl: one per (N, F) case.
NGROUPS=12

# GC memory ballast (GB); see experiments/common/gc_ballast.jl. Override with GC_BALLAST_GB=0 to disable.
GC_BALLAST_GB="${GC_BALLAST_GB:-7}"

# Right-hand side: ones (default), random or manufactured. See anisotropy_problem.jl.
TT_ANISOTROPY_RHS="${TT_ANISOTROPY_RHS:-ones}"

BENCH_LAYOUT="${BENCH_LAYOUT:-block}"
BENCH_N_REPEATS="${BENCH_N_REPEATS:-10}"
BENCH_REPEAT_OFFSET="${BENCH_REPEAT_OFFSET:-0}"
BENCH_BASE_SEED="${BENCH_BASE_SEED:-20260917}"
# 44 h of a 48 h limit: leaves 4 h for warmup, the final save, and a slow repeat that the
# estimator did not foresee.
BENCH_TIME_BUDGET_S="${BENCH_TIME_BUDGET_S:-158400}"

NOCOND_ARG=""
RESULTS_DIR="${SCRIPT_DIR}/results"

if [ "$1" == "nocond" ]; then
    NOCOND_ARG="nocond"
    RESULTS_DIR="${SCRIPT_DIR}/results_nocond"
fi

# Penalty μ of the FAI build (experiments/common/mu_rules.jl): gershgorin (default, μ = ‖A‖_∞ of each problem, times
# FAI_MU_SCALE, as in the paper) or fixed (the μ = 1e7 of the worker, as in the earlier drafts). A rule-based run
# writes under <results dir>_mugersh[_x<scale>] and passes that folder to the worker and the collector through
# TT_ANISOTROPY_STATS_RESULTS_DIR, so it never mixes with fixed-μ data.
FAI_MU_RULE="${FAI_MU_RULE:-gershgorin}"
FAI_MU_SCALE="${FAI_MU_SCALE:-1.0}"
MU_EXPORTS=""
if [ "${FAI_MU_RULE}" == "gershgorin" ]; then
    MU_TAG="gersh"
    if ! awk -v s="${FAI_MU_SCALE}" 'BEGIN { exit !(s + 0 == 1) }'; then
        MU_TAG="${MU_TAG}_x$(awk -v s="${FAI_MU_SCALE}" 'BEGIN { printf "%g", s }')"   # %g, as mu_rule_tag
    fi
    RESULTS_DIR="${RESULTS_DIR}_mu${MU_TAG}"
    MU_EXPORTS="export FAI_MU_RULE=${FAI_MU_RULE} FAI_MU_SCALE=${FAI_MU_SCALE} TT_ANISOTROPY_STATS_RESULTS_DIR=${RESULTS_DIR}"
elif [ "${FAI_MU_RULE}" != "fixed" ]; then
    echo "ERROR: FAI_MU_RULE must be 'fixed' or 'gershgorin', got '${FAI_MU_RULE}'" >&2
    exit 1
elif ! awk -v s="${FAI_MU_SCALE}" 'BEGIN { exit !(s + 0 == 1) }'; then
    echo "ERROR: FAI_MU_SCALE only applies to a rule, not to FAI_MU_RULE=fixed" >&2
    exit 1
fi

TASKS_DIR="${RESULTS_DIR}/tasks"
LOGS_DIR="${RESULTS_DIR}/logs"
mkdir -p "${TASKS_DIR}"
mkdir -p "${LOGS_DIR}"

# ---- Layout ----
if [ "${BENCH_LAYOUT}" == "block" ]; then
    NARRAY=$(( NGROUPS * BENCH_N_REPEATS ))
    LOG_PREFIX="block"
    ARRAY_DESC="${NGROUPS} groups x ${BENCH_N_REPEATS} repeats = ${NARRAY} blocks (offset ${BENCH_REPEAT_OFFSET})"
elif [ "${BENCH_LAYOUT}" == "task" ]; then
    NARRAY=${NTASKS}
    LOG_PREFIX="task"
    ARRAY_DESC="${NTASKS} tasks x ${BENCH_N_REPEATS} repeats each (offset ${BENCH_REPEAT_OFFSET})"
else
    echo "ERROR: BENCH_LAYOUT must be 'block' or 'task', got '${BENCH_LAYOUT}'" >&2
    exit 1
fi

# Both layouts write the same file names (task_<id>_rep_<r>.jld2), so a block run into a folder
# holding per-task results would silently skip every existing file and mix same-node repeats with
# node-paired ones. The LAYOUT marker refuses that.
LAYOUT_MARKER="${TASKS_DIR}/LAYOUT"
if [ -f "${LAYOUT_MARKER}" ]; then
    if [ "$(cat "${LAYOUT_MARKER}")" != "${BENCH_LAYOUT}" ]; then
        echo "ERROR: ${TASKS_DIR} holds '$(cat "${LAYOUT_MARKER}")'-layout results; refusing to add BENCH_LAYOUT=${BENCH_LAYOUT} results to them." >&2
        exit 1
    fi
elif [ -n "$(ls -A "${TASKS_DIR}")" ]; then
    echo "ERROR: ${TASKS_DIR} already holds results but no LAYOUT marker, so they come from the old per-task layout." >&2
    echo "       Move ${RESULTS_DIR} aside first (e.g. mv ${RESULTS_DIR} ${SCRIPT_DIR}/olddata_pertask/), or, to add" >&2
    echo "       per-task repeats to them on purpose: echo task > ${LAYOUT_MARKER}; BENCH_LAYOUT=task bash $0 $*" >&2
    exit 1
else
    echo "${BENCH_LAYOUT}" > "${LAYOUT_MARKER}"
fi

# ---- Submit job array ----
# Log names carry the layout and the chunk (offset, repeats) so that several submissions
# (BENCH_REPEAT_OFFSET chunks) do not overwrite each other; append mode keeps a resubmission's log too.
ARRAY_JOB_ID=$(sbatch --parsable --open-mode=append \
    --output="${LOGS_DIR}/${LOG_PREFIX}_%03a_off${BENCH_REPEAT_OFFSET}_n${BENCH_N_REPEATS}.out" \
    --error="${LOGS_DIR}/${LOG_PREFIX}_%03a_off${BENCH_REPEAT_OFFSET}_n${BENCH_N_REPEATS}.err" <<ARRAY_EOF
#!/bin/bash
#SBATCH --job-name=tt_aniso_stats
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --chdir=.
#SBATCH --array=1-${NARRAY}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02-00:00:00
#SBATCH --qos=${SLURM_QOS}

module load ${JULIA_MODULE}

if [ -n "\${SLURM_MEM_PER_NODE}" ]; then
    TASK_MEM_MB=\${SLURM_MEM_PER_NODE}
else
    TASK_MEM_MB=\$(( \${SLURM_MEM_PER_CPU:-2000} * SLURM_CPUS_PER_TASK ))
fi
HEAP_HINT_MB=\$(( TASK_MEM_MB * 80 / 100 ))

export GC_BALLAST_GB="${GC_BALLAST_GB}"
export TT_ANISOTROPY_RHS="${TT_ANISOTROPY_RHS}"
export BENCH_LAYOUT="${BENCH_LAYOUT}"
export BENCH_N_REPEATS="${BENCH_N_REPEATS}"
export BENCH_REPEAT_OFFSET="${BENCH_REPEAT_OFFSET}"
export BENCH_BASE_SEED="${BENCH_BASE_SEED}"
export BENCH_TIME_BUDGET_S="${BENCH_TIME_BUDGET_S}"

echo "Array index \${SLURM_ARRAY_TASK_ID} / ${NARRAY} (${BENCH_LAYOUT} layout) — Job \${SLURM_JOB_ID} on \${HOSTNAME}"
echo "GC ballast ${GC_BALLAST_GB} GB, task memory \${TASK_MEM_MB} MB, heap hint \${HEAP_HINT_MB} MB, rhs ${TT_ANISOTROPY_RHS}"
echo "Repeats: LAYOUT=${BENCH_LAYOUT} N_REPEATS=${BENCH_N_REPEATS} REPEAT_OFFSET=${BENCH_REPEAT_OFFSET} BASE_SEED=${BENCH_BASE_SEED} TIME_BUDGET_S=${BENCH_TIME_BUDGET_S}"

${MU_EXPORTS}
echo "FAI μ rule: ${FAI_MU_RULE} (scale ${FAI_MU_SCALE}), results in ${RESULTS_DIR}"

julia --project=experiments -t 1 --heap-size-hint=\${HEAP_HINT_MB}M ${SCRIPT_DIR}/worker.jl \${SLURM_ARRAY_TASK_ID} ${NOCOND_ARG}
ARRAY_EOF
)

echo "Submitted array job: ${ARRAY_JOB_ID} (${BENCH_LAYOUT} layout: ${ARRAY_DESC})"

# ---- Submit collector job ----
COLLECT_JOB_ID=$(sbatch --parsable --open-mode=append --dependency=afterany:${ARRAY_JOB_ID} \
    --output="${LOGS_DIR}/collect_off${BENCH_REPEAT_OFFSET}_n${BENCH_N_REPEATS}.out" \
    --error="${LOGS_DIR}/collect_off${BENCH_REPEAT_OFFSET}_n${BENCH_N_REPEATS}.err" <<COLLECT_EOF
#!/bin/bash
#SBATCH --job-name=tt_aniso_st_coll
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --chdir=.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --qos=${SLURM_QOS}

module load ${JULIA_MODULE} 2>/dev/null || true

echo "Collecting results from array job..."

${MU_EXPORTS}
julia --project=experiments ${SCRIPT_DIR}/collect.jl ${NOCOND_ARG}
COLLECT_EOF
)

echo "Submitted collector job: ${COLLECT_JOB_ID} (depends on ${ARRAY_JOB_ID})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
