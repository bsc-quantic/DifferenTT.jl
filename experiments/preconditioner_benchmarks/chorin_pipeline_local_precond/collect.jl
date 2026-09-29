using JLD2
using DataFrames

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/chorin_pipeline_local_precond/results_nocond" : "experiments/preconditioner_benchmarks/chorin_pipeline_local_precond/results"
tasks_folder = joinpath(base_folder, "tasks")

const COND_NODES = [(8, 8), (10, 10), (12, 12)]

function should_compute_cond(nodes::Tuple{Int, Int}, use_nocond::Bool)::Bool
    return !use_nocond && nodes in COND_NODES
end

df = DataFrame()

for file in readdir(tasks_folder)
    if endswith(file, ".jld2")
        filepath = joinpath(tasks_folder, file)
        try
            result = load(filepath, "result")
            if !haskey(result, "compute_cond")
                result["compute_cond"] = should_compute_cond(Tuple(result["nodes"]), use_nocond)
            end
            
            result_sym = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in result)
            push!(df, result_sym, cols=:union)
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
