"""
    TT Poisson benchmark (global TT-CG): collector

The workers of this benchmark write one file per task, and the plotter needs them as one table. We read every
`tasks/task_NNN.jld2` of the results folder and stack the `"result"` dictionaries into a DataFrame with one row per task
(missing fields are filled with `missing`). Tasks whose file cannot be read are reported and skipped.

# Inputs and outputs

Input: the task files in the `tasks/` folder of `results/`, in this folder. Output: `collected_results.jld2` (key
`"df"`) next to `tasks/`.

In smoke mode (`DIFFERENTT_SMOKE=1`) it reads the folder `DIFFERENTT_SMOKE_DIR` that the smoke worker wrote, never the
committed results; `experiments/smoke.jl` sets it.

# How to run

From the root of the repository:

    julia --project=experiments experiments/fai_benchmarks/tt_poisson_global/collect.jl

`submit.sh` submits it as a collector job that depends on the worker array; the job script requests 4 CPUs for 10
minutes.
"""

using JLD2
using DataFrames

base_folder = joinpath(@__DIR__, "results")
tasks_folder = joinpath(base_folder, "tasks")

df = DataFrame()

for file in readdir(tasks_folder)
    if endswith(file, ".jld2")
        filepath = joinpath(tasks_folder, file)
        try
            result = load(filepath, "result")
            result_sym = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in result)
            push!(df, result_sym; cols=:union)
        catch e
            println("Error loading $file: $e")
        end
    end
end

output_file = joinpath(base_folder, "collected_results.jld2")
save(output_file, "df", df)
println("Collected $(nrow(df)) tasks into $output_file")
