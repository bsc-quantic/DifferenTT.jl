"""
    TT anisotropy benchmark, statistics variant (paper Figs. 9-11): collector

The workers of this benchmark write one file per task and repeat, and the plotter needs them as one table. We read every
`tasks/task_NNN_rep_RRR.jld2` of the results folder with `collect_dataset` (`common/stats_utils.jl`), which checks that the
repeats are consistent (hosts, seeds, blocks), and save a long table with one row per (task, repeat) (`"df"`) and a table
with the statistics of every configuration (`"df_stats"`). A configuration is a pair (case, key), where the case is (N, F):
the baseline that a configuration is compared with is the one of the same dimension and floor. The long table has
several rows per key, so the plotter of the non-statistics benchmark, `fai_benchmarks/tt_anisotropy/plot_results.jl`,
cannot read it: use `plot_paper.jl` of this folder.

# Inputs and outputs

Input: the task files in the `tasks/` folder of `results/` (or `results_nocond/` with `nocond`), in this folder, or in
`TT_ANISOTROPY_STATS_RESULTS_DIR` if set. Output: `collected_results.jld2` next to `tasks/` (keys `"df"`, `"df_stats"`,
`"df_dropped"`). Two environment variables change the statistics: `BENCH_DISCARD_FIRST_REPEAT=1` drops the first repeat
that each worker process ran (with a single solve per repeat it can still carry residual JIT cost, which matters for
solves of less than a second; the summary then reports the numbers before and after), and `BENCH_SAMEJOB_PAIRS_ONLY=1`
computes the speedups only from pairs that ran in the same job, that is on the same node (a block that was refilled in a
later job is dropped); by default all the pairs are used and `n_pairs_samejob` counts the ones of the same job.

In smoke mode (`DIFFERENTT_SMOKE=1`) it reads the folder `DIFFERENTT_SMOKE_DIR` that the smoke worker wrote, never the
committed results; `experiments/smoke.jl` sets it.

# How to run

From the root of the repository:

    julia --project=experiments experiments/paper/tt_anisotropy_stats/collect.jl [nocond]

`submit.sh` queues it as a collector job after each submission of the worker array (4 CPUs, 30 minutes); the one that
runs last sees every finished unit.

# Paper

Figs. 9-11 of Sec. 5.1 (through `plot_paper.jl`).
"""

using JLD2
using DataFrames
using Statistics

include(joinpath(@__DIR__, "..", "..", "common", "stats_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
# Smoke mode reads (and writes) the folder DIFFERENTT_SMOKE_DIR instead of this folder.
const OUT_ROOT = output_root(@__DIR__)

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
default_base_folder = if use_nocond
    joinpath(OUT_ROOT, "results_nocond")
else
    joinpath(OUT_ROOT, "results")
end
base_folder = get(ENV, "TT_ANISOTROPY_STATS_RESULTS_DIR", default_base_folder)
discard_first = get(ENV, "BENCH_DISCARD_FIRST_REPEAT", "0") in ("1", "true", "yes")
samejob_only = get(ENV, "BENCH_SAMEJOB_PAIRS_ONLY", "0") in ("1", "true", "yes")

function postprocess(result)
    haskey(result, "compute_cond") || (result["compute_cond"] = !use_nocond)
    result["case"] = (Int(result["N"]), Float64(result["F"]))
    return result
end

collect_dataset(
    base_folder;
    postprocess,
    size_col=:case,
    solve_col=:solve_time,
    total_col=:total_time,
    iters_col=:total_iters,
    scalar_cols=[
        :solve_time,
        :total_time,
        :precond_build_time,
        :relative_residual,
        :relative_error,
        :solution_maxdim,
        :total_iters,
        :nsweeps,
        :avg_cond,
        :avg_pc_cond,
        :compile_time,
        :gc_time,
        :precond_maxlinkdim,
        :repeat_wall_time,
    ],
    vector_cols=[:iters_per_site, :conds_per_site, :pc_conds_per_site],
    discard_first,
    samejob_only,
)
