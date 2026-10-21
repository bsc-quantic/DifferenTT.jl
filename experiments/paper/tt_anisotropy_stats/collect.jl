# Collector for tt_anisotropy_stats: one row per (task, repeat) plus one row per configuration.
#
# The saved "df" has SEVERAL rows per key (one per repeat). The original
# tt_anisotropy/plot_results.jl expects one row per key and produces nonsense from it: use this
# directory's plot_results.jl.
#
# A configuration is a (case, key) pair, where `case = (N, F)`: the baseline that a configuration's
# speedups are paired against is the one at the same dimension AND floor.
#
# Usage: julia --project=. experiments/preconditioner_benchmarks/tt_anisotropy_stats/collect.jl [nocond]
#
# Environment:
#   TT_ANISOTROPY_STATS_RESULTS_DIR  override the results folder (same variable as the worker)
#   BENCH_DISCARD_FIRST_REPEAT=1     drop the first repeat each worker process ran. With a single
#                                    solve per repeat (N_SOLVE_REPEATS = 1) that repeat can still
#                                    carry residual JIT cost, which matters for sub-second solves;
#                                    default 0. The variability summary then reports before/after.
#   BENCH_SAMEJOB_PAIRS_ONLY=1       block layout: compute the speedups only from pairs that ran
#                                    in the same job (same node), dropping those of blocks refilled
#                                    in a later job; default 0 (all pairs, n_pairs_samejob counted).

using JLD2
using DataFrames
using Statistics

include(joinpath(@__DIR__, "..", "stats_utils.jl"))

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
default_base_folder = if use_nocond
    "experiments/preconditioner_benchmarks/tt_anisotropy_stats/results_nocond"
else
    "experiments/preconditioner_benchmarks/tt_anisotropy_stats/results"
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
