using JLD2
using DataFrames

# Number of moving lids of the worker run (see worker.jl): 2 reads the doubly-driven cavity results
const CAVITY_LIDS = parse(Int, get(ENV, "CAVITY_LIDS", "1"))
CAVITY_LIDS in (1, 2) || error("CAVITY_LIDS must be 1 or 2, got $CAVITY_LIDS")
const RESULTS_SUFFIX = CAVITY_LIDS == 2 ? "_doublelid" : ""

base_folder = "experiments/preconditioner_benchmarks/liddriven_pipeline_globalcg/results" * RESULTS_SUFFIX
tasks_folder = joinpath(base_folder, "tasks")

df = DataFrame()

for file in readdir(tasks_folder)
    if endswith(file, ".jld2")
        filepath = joinpath(tasks_folder, file)
        try
            result = load(filepath, "result")
            result_sym = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in result)
            push!(df, result_sym, cols=:union)
        catch e
            println("Error loading $file: $e")
        end
    end
end

output_file = joinpath(base_folder, "collected_results.jld2")
save(output_file, "df", df)
println("Collected $(nrow(df)) tasks into $output_file")
