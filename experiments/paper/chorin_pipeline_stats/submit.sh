#!/bin/bash
#
# Preconditioner Benchmark, statistics variant: Parallel Sweep for chorin_pipeline_stats
#
# The 20 tasks of chorin_pipeline/submit.sh (2 system sizes x (baseline + FAI chi=2..10)), each
# repeated, one result file per (task, repeat): tasks/task_<id>_rep_<r>.jld2. Submits up to two
# SLURM job arrays (one per layout, see below) and a collector job that depends on both. Every
# submission queues a collector; the one that runs last sees every finished unit.
#
# Usage: bash experiments/paper/chorin_pipeline_stats/submit.sh [nocond]
#
# By default, condition-number estimates are collected for (8,8) and (10,10). Pass "nocond" to
# skip them. Timings/iteration statistics should come from a "nocond" run; a condition-number run
# is much slower per repeat, so give it fewer repeats (e.g. BENCH_N_REPEATS=5).
#
# --- Layout: blocks for (8,8), single units for (10,10) ------------------------------------
# About 1 allocation in 6 runs 24-37% slower for its whole lifetime (measured on the Chorin
# benchmarks, in shared AND exclusive runs, so --exclusive does not help). How the units are
# spread over jobs decides where that slowdown ends up (details in experiments/common/stats_harness.jl):
#
# BLOCK_SIZES (block layout, default 8x8): one job = one block = every method of one system size
#   at one repeat index, in a seeded random order. Speedups are within-node ratios and the node
#   effect cancels in them; the collector estimates it (block effects in variability_summary.md).
#   Array indices are repeat-major. Do not bundle several repeats of a group into one job.
# UNIT_SIZES (unit layout, default 10x10): one job = ONE (method, repeat) unit, because a (10,10)
#   unit takes ~10 h and a 10-method block would need ~4 jobs of 48 h. Every unit runs on its own
#   node, all in parallel, so the whole sweep finishes in one submission. The price: the speedups
#   pair runs on different nodes, so the node effect is not cancelled but widens their spread.
#   Array indices are repeat-major (the first 10 jobs are repeat 1 of every method).
# A size goes in at most one of the two lists; an empty list skips that array.
#
# The tasks/ folder records the split in its LAYOUT marker and refuses results of another split
# (same file names, different pairing). Results from the old all-block layout, or from another
# n_steps/dt, must be moved aside first:
#     mv <results dir> <script dir>/olddata_<something>/
#
# Repeat configuration (environment variables, exported to the workers and echoed in the job log):
#   BENCH_N_REPEATS      repeat indices covered by this submission       (default 10)
#   BENCH_REPEAT_OFFSET  global index of the first repeat - 1           (default 0)
#   BENCH_BASE_SEED      base of the per-repeat seeds                     (default 20260917)
#   BENCH_TIME_BUDGET_S  block layout: stop STARTING units past this wall time (default 158400 = 44 h)
#   BLOCK_SIZES / UNIT_SIZES   see above (comma-separated, e.g. "8x8,10x10")
#   BLOCK_TIME / UNIT_TIME     SLURM --time of each array (default 02-00:00:00 / 01-00:00:00)
#   CPUS_PER_TASK        cores per job (default 20); changing it changes the timings, so do not
#                        mix runs with different values in one results folder
# More repeats later, without touching the existing ones (seeds use the GLOBAL repeat index, so
# pairing is preserved): BENCH_REPEAT_OFFSET=10 BENCH_N_REPEATS=5 bash .../submit.sh nocond
#
# --- Wall clock and queue footprint --------------------------------------------------------
# Per-unit times (Oct 2026 nocond run: 500 steps (8,8), 250 steps (10,10), 20 cores):
#     (8,8)   500 steps: ~1.2 h per unit  -> a 10-method block ~12 h
#     (10,10) 250 steps: ~4.3 h per unit, ~62 s/step, dominated by RK4/masking, not the solve;
#             the bond dimension is at the 200 cap from step ~50, so 500 steps ~ 9-12 h per unit.
# UNIT_TIME=1 day leaves a ~2x margin for (10,10) at 500 steps; raise it if n_steps grows.
# Cores held at once: 10 repeats -> 10 block jobs + 100 unit jobs = 110 x 20 = 2200 cores
# (5 repeats: 1100), well inside the 4608-core limit of the QOS.
# A block stopped by the guard, or a failed/killed unit: run the SAME command again, units whose
# file exists are skipped. Refilled block units run in another job, so their pairs are no longer
# within-node; the collector counts them (n_pairs_samejob), BENCH_SAMEJOB_PAIRS_ONLY=1 drops them.
#
# The physics/solver configuration is identical to chorin_pipeline/worker.jl, and the masks are
# read from the masks of fai_benchmarks/chorin_pipeline/ (masks.jld2); the worker stops if they were not
# sampled on its grid, so regenerate them first (fai_benchmarks/chorin_pipeline/generate_masks.jl).
#

SCRIPT_DIR="experiments/paper/chorin_pipeline_stats"

# Cluster settings: override from the environment, e.g. SLURM_ACCOUNT=<project> bash experiments/paper/chorin_pipeline_stats/submit.sh
SLURM_ACCOUNT="${SLURM_ACCOUNT:-bsc21}"
SLURM_QOS="${SLURM_QOS:-ngp_bsccase}"
JULIA_MODULE="${JULIA_MODULE:-julia/1.12.0}"

# Tasks per system size: baseline + FAI chi=2..10, must match build_task_list() in worker.jl.
TASKS_PER_SIZE=10

# GC memory ballast (GB); see experiments/common/gc_ballast.jl. Override with GC_BALLAST_GB=0 to disable.
GC_BALLAST_GB="${GC_BALLAST_GB:-7}"

BLOCK_SIZES="${BLOCK_SIZES-8x8}"
UNIT_SIZES="${UNIT_SIZES-10x10}"
BLOCK_TIME="${BLOCK_TIME:-02-00:00:00}"
UNIT_TIME="${UNIT_TIME:-01-00:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-20}"

BENCH_N_REPEATS="${BENCH_N_REPEATS:-10}"
BENCH_REPEAT_OFFSET="${BENCH_REPEAT_OFFSET:-0}"
BENCH_BASE_SEED="${BENCH_BASE_SEED:-20260917}"
# 44 h of a 48 h limit: leaves 4 h for warmup, the final save, and a slow repeat that the
# estimator did not foresee. Only the block layout uses it (a unit job runs one unit).
BENCH_TIME_BUDGET_S="${BENCH_TIME_BUDGET_S:-158400}"

NOCOND_ARG=""
RESULTS_DIR="${SCRIPT_DIR}/results"

if [ "$1" == "nocond" ]; then
    NOCOND_ARG="nocond"
    RESULTS_DIR="${SCRIPT_DIR}/results_nocond"
fi

TASKS_DIR="${RESULTS_DIR}/tasks"
LOGS_DIR="${RESULTS_DIR}/logs"

# ---- Layout split ----
count_sizes() { [ -z "$1" ] && echo 0 || echo "$1" | tr ',' '\n' | grep -c .; }
NBLOCK_SIZES=$(count_sizes "${BLOCK_SIZES}")
NUNIT_SIZES=$(count_sizes "${UNIT_SIZES}")
if [ $(( NBLOCK_SIZES + NUNIT_SIZES )) -eq 0 ]; then
    echo "ERROR: BLOCK_SIZES and UNIT_SIZES are both empty" >&2
    exit 1
fi
for s in $(echo "${BLOCK_SIZES}" | tr ',' ' '); do
    if echo ",${UNIT_SIZES}," | grep -q ",${s},"; then
        echo "ERROR: size ${s} is in both BLOCK_SIZES and UNIT_SIZES" >&2
        exit 1
    fi
done
NBLOCK_ARRAY=$(( NBLOCK_SIZES * BENCH_N_REPEATS ))
NUNIT_ARRAY=$(( NUNIT_SIZES * TASKS_PER_SIZE * BENCH_N_REPEATS ))

# Block and unit results have the same file names (task_<id>_rep_<r>.jld2) but pair differently,
# so a folder holds one split only. The old all-block folders carry the marker "block".
LAYOUT_TAG="block:${BLOCK_SIZES};unit:${UNIT_SIZES}"
mkdir -p "${TASKS_DIR}" "${LOGS_DIR}"
LAYOUT_MARKER="${TASKS_DIR}/LAYOUT"
if [ -f "${LAYOUT_MARKER}" ]; then
    if [ "$(cat "${LAYOUT_MARKER}")" != "${LAYOUT_TAG}" ]; then
        echo "ERROR: ${TASKS_DIR} holds '$(cat "${LAYOUT_MARKER}")' results; refusing to add '${LAYOUT_TAG}' results to them." >&2
        echo "       Move ${RESULTS_DIR} aside first (e.g. mv ${RESULTS_DIR} ${SCRIPT_DIR}/olddata_<something>/)." >&2
        exit 1
    fi
elif [ -n "$(ls -A "${TASKS_DIR}")" ]; then
    echo "ERROR: ${TASKS_DIR} already holds results but no LAYOUT marker, so they come from the old per-task layout." >&2
    echo "       Move ${RESULTS_DIR} aside first (e.g. mv ${RESULTS_DIR} ${SCRIPT_DIR}/olddata_pertask/)." >&2
    exit 1
else
    echo "${LAYOUT_TAG}" > "${LAYOUT_MARKER}"
fi

# ---- Submit one job array ----
# $1 layout (block|unit), $2 sizes, $3 number of array indices, $4 --time
# Log names carry the layout and the chunk (offset, repeats) so that several submissions
# (BENCH_REPEAT_OFFSET chunks) do not overwrite each other; append mode keeps a resubmission's log too.
submit_array() {
    local layout="$1" sizes="$2" narray="$3" walltime="$4"
    sbatch --parsable --open-mode=append \
        --output="${LOGS_DIR}/${layout}_%03a_off${BENCH_REPEAT_OFFSET}_n${BENCH_N_REPEATS}.out" \
        --error="${LOGS_DIR}/${layout}_%03a_off${BENCH_REPEAT_OFFSET}_n${BENCH_N_REPEATS}.err" <<ARRAY_EOF
#!/bin/bash
#SBATCH --job-name=chorin_stats_${layout}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --chdir=.
#SBATCH --array=1-${narray}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${walltime}
#SBATCH --qos=${SLURM_QOS}

module load ${JULIA_MODULE}

if [ -n "\${SLURM_MEM_PER_NODE}" ]; then
    TASK_MEM_MB=\${SLURM_MEM_PER_NODE}
else
    TASK_MEM_MB=\$(( \${SLURM_MEM_PER_CPU:-2000} * SLURM_CPUS_PER_TASK ))
fi
HEAP_HINT_MB=\$(( TASK_MEM_MB * 80 / 100 ))

export GC_BALLAST_GB="${GC_BALLAST_GB}"
export BENCH_LAYOUT="${layout}"
export CHORIN_STATS_SIZES="${sizes}"
export BENCH_N_REPEATS="${BENCH_N_REPEATS}"
export BENCH_REPEAT_OFFSET="${BENCH_REPEAT_OFFSET}"
export BENCH_BASE_SEED="${BENCH_BASE_SEED}"
export BENCH_TIME_BUDGET_S="${BENCH_TIME_BUDGET_S}"

echo "Array index \${SLURM_ARRAY_TASK_ID} / ${narray} (${layout} layout, sizes ${sizes}) — Job \${SLURM_JOB_ID} on \${HOSTNAME}"
echo "GC ballast ${GC_BALLAST_GB} GB, task memory \${TASK_MEM_MB} MB, heap hint \${HEAP_HINT_MB} MB"
echo "Repeats: LAYOUT=${layout} SIZES=${sizes} N_REPEATS=${BENCH_N_REPEATS} REPEAT_OFFSET=${BENCH_REPEAT_OFFSET} BASE_SEED=${BENCH_BASE_SEED} TIME_BUDGET_S=${BENCH_TIME_BUDGET_S}"

julia --project=experiments -t 1 --heap-size-hint=\${HEAP_HINT_MB}M ${SCRIPT_DIR}/worker.jl \${SLURM_ARRAY_TASK_ID} ${NOCOND_ARG}
ARRAY_EOF
}

if [ -n "${NOCOND_ARG}" ]; then
    echo "Mode: nocond (no condition-number estimates)"
else
    echo "Mode: default (condition-number estimates on 8x8 and 10x10)"
fi

ARRAY_JOB_IDS=""
if [ "${NBLOCK_ARRAY}" -gt 0 ]; then
    BLOCK_JOB_ID=$(submit_array block "${BLOCK_SIZES}" "${NBLOCK_ARRAY}" "${BLOCK_TIME}")
    echo "Submitted block array: ${BLOCK_JOB_ID} (sizes ${BLOCK_SIZES}: ${NBLOCK_SIZES} groups x ${BENCH_N_REPEATS} repeats = ${NBLOCK_ARRAY} jobs, offset ${BENCH_REPEAT_OFFSET}, --time ${BLOCK_TIME})"
    ARRAY_JOB_IDS="${ARRAY_JOB_IDS}:${BLOCK_JOB_ID}"
fi
if [ "${NUNIT_ARRAY}" -gt 0 ]; then
    UNIT_JOB_ID=$(submit_array unit "${UNIT_SIZES}" "${NUNIT_ARRAY}" "${UNIT_TIME}")
    echo "Submitted unit array: ${UNIT_JOB_ID} (sizes ${UNIT_SIZES}: ${NUNIT_SIZES} x ${TASKS_PER_SIZE} tasks x ${BENCH_N_REPEATS} repeats = ${NUNIT_ARRAY} jobs, offset ${BENCH_REPEAT_OFFSET}, --time ${UNIT_TIME})"
    ARRAY_JOB_IDS="${ARRAY_JOB_IDS}:${UNIT_JOB_ID}"
fi
echo "Cores requested at once: $(( (NBLOCK_ARRAY + NUNIT_ARRAY) * CPUS_PER_TASK ))"

# ---- Submit collector job ----
COLLECT_JOB_ID=$(sbatch --parsable --open-mode=append --dependency=afterany${ARRAY_JOB_IDS} \
    --output="${LOGS_DIR}/collect_off${BENCH_REPEAT_OFFSET}_n${BENCH_N_REPEATS}.out" \
    --error="${LOGS_DIR}/collect_off${BENCH_REPEAT_OFFSET}_n${BENCH_N_REPEATS}.err" <<COLLECT_EOF
#!/bin/bash
#SBATCH --job-name=chorin_st_coll
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --chdir=.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --qos=${SLURM_QOS}

module load ${JULIA_MODULE} 2>/dev/null || true

echo "Collecting results from array job(s)${ARRAY_JOB_IDS}..."

julia --project=experiments ${SCRIPT_DIR}/collect.jl ${NOCOND_ARG}
COLLECT_EOF
)

echo "Submitted collector job: ${COLLECT_JOB_ID} (depends on${ARRAY_JOB_IDS})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
