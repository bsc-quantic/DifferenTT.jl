# Collector for chorin_pipeline_stats: one row per (task, repeat) plus one row per configuration.
#
# The saved "df" has SEVERAL rows per key (one per repeat). The original chorin_pipeline/plot_results.jl
# expects one row per key and produces nonsense from it: use this directory's plot_paper.jl.
#
# Usage: julia --project=experiments experiments/paper/chorin_pipeline_stats/collect.jl [nocond]
#
# Environment:
#   CHORIN_STATS_RESULTS_DIR      override the results folder (same variable as the worker)
#   BENCH_DISCARD_FIRST_REPEAT=1  drop the first repeat each worker process ran (may still carry
#                                 JIT cost); default 0. The plot script's variability summary then
#                                 reports before/after.
#   BENCH_SAMEJOB_PAIRS_ONLY=1    block layout: compute the speedups only from pairs that ran
#                                 in the same job (same node), dropping those of blocks refilled
#                                 in a later job; default 0 (all pairs, n_pairs_samejob counted).

using JLD2
using DataFrames
# Needed to deserialize the "last_snapshot" entry of the task files (which is then discarded)
using ITensors
using ITensorMPS
using DifferenTT
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
base_folder = get(ENV, "CHORIN_STATS_RESULTS_DIR", default_base_folder)
discard_first = get(ENV, "BENCH_DISCARD_FIRST_REPEAT", "0") in ("1", "true", "yes")
samejob_only = get(ENV, "BENCH_SAMEJOB_PAIRS_ONLY", "0") in ("1", "true", "yes")

const COND_NODES = [(8, 8), (10, 10), (12, 12)]

function should_compute_cond(nodes::Tuple{Int,Int}, use_nocond::Bool)::Bool
    return !use_nocond && nodes in COND_NODES
end

# Scalar summaries derived from the per-step data, so the stats frame can treat them like any
# other scalar metric. `iters` is (steps x sites); the plots want the per-step sum.
function postprocess(result)
    # The final fluid state is only for the snapshot plots, which read the task files directly
    delete!(result, "last_snapshot")
    if !haskey(result, "compute_cond")
        result["compute_cond"] = should_compute_cond(Tuple(result["nodes"]), use_nocond)
    end
    result["total_time"] = result["total_solve_time"] + result["precond_build_time"]
    per_step = Float64[sum(step) for step in result["iters"]]
    result["iters_per_step"] = per_step
    result["total_iters"] = sum(per_step)
    result["residual_max"] = maximum(result["residuals"])
    result["solution_maxdim"] = maximum(result["solution_maxdims"])
    # Results written before the divergence was tracked have no div_ratios
    haskey(result, "div_ratios") && (result["div_ratio_max"] = maximum(result["div_ratios"]))
    return result
end

collect_dataset(
    base_folder;
    postprocess,
    size_col=:nodes,
    solve_col=:total_solve_time,
    total_col=:total_time,
    iters_col=:total_iters,
    scalar_cols=[
        :total_solve_time,
        :total_time,
        :precond_build_time,
        :total_iters,
        :residual_max,
        :solution_maxdim,
        :div_ratio_max,
        :avg_cond,
        :avg_pc_cond,
        :total_compile_time,
        :total_gc_time,
        :repeat_wall_time,
    ],
    vector_cols=[
        :solve_times,
        :residuals,
        :solution_maxdims,
        :iters_per_step,
        :iters_per_site,
        :conds_per_site,
        :pc_conds_per_site,
        :div_ratios,
        :div_norms_before,
        :div_norms_after,
    ],
    discard_first,
    samejob_only,
)
