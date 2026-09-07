using JLD2
using DataFrames

base_folder = "experiments/preconditioner_benchmarks/tt_poisson_global/results"
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
