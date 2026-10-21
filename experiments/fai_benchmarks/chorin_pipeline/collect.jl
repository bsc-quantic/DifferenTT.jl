"""
    Chorin pipeline benchmark (local ResidualALS): collector

The workers of this benchmark write one file per task, and the plotter needs them as one table. We read every
`tasks/task_NNN.jld2` of the results folder and stack the `"result"` dictionaries into a DataFrame with one row per task
(missing fields are filled with `missing`). The final fluid state of every task (`"last_snapshot"`) is dropped: the
plotter reads it from the task files. Tasks whose file cannot be read are reported and skipped.

# Inputs and outputs

Input: the task files in the `tasks/` folder of `results/` (or `results_nocond/` with `nocond`), in this folder. Output:
`collected_results.jld2` (key `"df"`) next to `tasks/`.

In smoke mode (`DIFFERENTT_SMOKE=1`) it reads the folder `DIFFERENTT_SMOKE_DIR` that the smoke worker wrote, never the
committed results; `experiments/smoke.jl` sets it.

# How to run

From the root of the repository:

    julia --project=experiments experiments/fai_benchmarks/chorin_pipeline/collect.jl [nocond]

`submit.sh` submits it as a collector job that depends on the worker array; the job script requests 4 CPUs for 10
minutes.
"""

using JLD2
using DataFrames
# Needed to deserialize the "last_snapshot" entry of the task files (which is then discarded)
using ITensors
using ITensorMPS
using DifferenTT

include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
# Smoke mode reads (and writes) the folder DIFFERENTT_SMOKE_DIR instead of this folder.
const OUT_ROOT = output_root(@__DIR__)

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = if use_nocond
    joinpath(OUT_ROOT, "results_nocond")
else
    joinpath(OUT_ROOT, "results")
end
tasks_folder = joinpath(base_folder, "tasks")

const COND_NODES = [(8, 8), (10, 10), (12, 12)]

function should_compute_cond(nodes::Tuple{Int,Int}, use_nocond::Bool)::Bool
    return !use_nocond && nodes in COND_NODES
end

df = DataFrame()

for file in readdir(tasks_folder)
    if endswith(file, ".jld2")
        filepath = joinpath(tasks_folder, file)
        try
            result = load(filepath, "result")
            # The final fluid state is only for the snapshot plots, which read the task files directly
            delete!(result, "last_snapshot")
            if !haskey(result, "compute_cond")
                result["compute_cond"] = should_compute_cond(Tuple(result["nodes"]), use_nocond)
            end

            result_sym = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in result)
            push!(df, result_sym; cols=:union)
        catch e
            println("Error loading $file: $e")
        end
    end
end

if nrow(df) > 0 && !hasproperty(df, :compute_cond)
    df.compute_cond = [should_compute_cond(Tuple(n), use_nocond) for n in df.nodes]
end

output_file = joinpath(base_folder, "collected_results.jld2")
save(output_file, "df", df)
n_with_cond = nrow(df) > 0 && hasproperty(df, :compute_cond) ? count(df.compute_cond) : 0
println("Collected $(nrow(df)) tasks into $output_file ($n_with_cond with condition-number estimates)")
