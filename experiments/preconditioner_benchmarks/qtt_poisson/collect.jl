using JLD2
using DataFrames

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/qtt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/qtt_poisson/results"
tasks_folder = joinpath(base_folder, "tasks")

df = DataFrame()

for file in readdir(tasks_folder)
    if endswith(file, ".jld2")
        filepath = joinpath(tasks_folder, file)
        try
            result = load(filepath, "result")
            push!(df, result, cols=:union)
        catch e
            println("Error loading $file: $e")
        end
    end
end

output_file = joinpath(base_folder, "collected_results.jld2")
save(output_file, "df", df)
println("Collected $(nrow(df)) tasks into $output_file")
