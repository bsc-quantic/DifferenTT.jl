"""
    Chorin pipeline benchmark, statistics variant (paper Table 3 and Figs. 11-12): collector

The workers of this benchmark write one file per task and repeat, and the plotter needs them as one table. We read every
`tasks/task_NNN_rep_RRR.jld2` of the results folder with `collect_dataset` (`common/stats_utils.jl`), which checks that the
repeats are consistent (hosts, seeds, blocks), and save a long table with one row per (task, repeat) (`"df"`) and a table
with the statistics of every configuration (`"df_stats"`). The long table has several rows per key, so the plotter of the
non-statistics benchmark, `fai_benchmarks/chorin_pipeline/plot_results.jl`, cannot read it: use `plot_paper.jl` of this
folder. Per task we add the total time (solve plus preconditioner build), the iterations per step, the largest residual,
the largest solution bond dimension and the largest ratio of the divergence after and before the projection.

# Inputs and outputs

Input: the task files in the `tasks/` folder of `results_mugersh/` (or `results_nocond_mugersh/` with `nocond`), in this
folder, or in `CHORIN_STATS_RESULTS_DIR` if set: the folders of the default μ rule, `gershgorin` (see `worker.jl`); with
`FAI_MU_RULE=fixed`, the folders `results/` and `results_nocond/` of the fixed μ. The files of repeat 1 also hold the
final fluid state, which we load (this needs DifferenTT and ITensors) and drop, since only the snapshot plots use it.
Output: `collected_results.jld2` next to `tasks/` (keys `"df"`, `"df_stats"`, `"df_dropped"`). Two environment variables
change the statistics: `BENCH_DISCARD_FIRST_REPEAT=1` drops the first repeat that each worker process ran (it can still
carry residual JIT cost), and `BENCH_SAMEJOB_PAIRS_ONLY=1` computes the speedups only from pairs that ran in the same
job, that is on the same node (a block that was refilled in a later job is dropped); by default all the pairs are used
and `n_pairs_samejob` counts the ones of the same job.

In smoke mode (`DIFFERENTT_SMOKE=1`) it reads the folder `DIFFERENTT_SMOKE_DIR` that the smoke worker wrote, never the
committed results; `experiments/smoke.jl` sets it.

# How to run

From the root of the repository:

    julia --project=experiments experiments/paper/chorin_pipeline_stats/collect.jl [nocond]

`submit.sh` queues it as a collector job after each submission of the worker arrays (4 CPUs, 30 minutes); the one that
runs last sees every finished unit.

# Paper

Figs. 11-12 of Sec. 5.2 (through `plot_paper.jl`); the parameters of Table 3 are the constants of the worker.
"""

using JLD2
using DataFrames
# Needed to deserialize the "last_snapshot" entry of the task files (which is then discarded)
using ITensors
using ITensorMPS
using DifferenTT
using Statistics

include(joinpath(@__DIR__, "..", "..", "common", "stats_utils.jl"))
include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
include(joinpath(@__DIR__, "..", "..", "common", "mu_rules.jl"))
# Smoke mode reads (and writes) the folder DIFFERENTT_SMOKE_DIR instead of this folder.
const OUT_ROOT = output_root(@__DIR__)

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
const MU_RULE, MU_SCALE = fai_mu_rule()
default_base_folder = joinpath(
    OUT_ROOT, mu_results_folder(use_nocond ? "results_nocond" : "results", MU_RULE, MU_SCALE)
)
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
        :precond_build_total_iters,
        :precond_build_nsweeps,
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
        :precond_build_iters_per_site,
    ],
    discard_first,
    samejob_only,
)
