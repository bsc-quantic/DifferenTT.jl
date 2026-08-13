using JLD2
using DataFrames
using Plots

gr()
default(fmt=:png, dpi=300, size=(800, 600), fontfamily="Computer Modern", margin=5Plots.mm)

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/chorin_pipeline/results_nocond" : "experiments/preconditioner_benchmarks/chorin_pipeline/results"
input_file = joinpath(base_folder, "collected_results.jld2")

df = load(input_file, "df")

if ncol(df) == 0 || nrow(df) == 0
    println("Warning: Dataframe is empty or has no columns in $(input_file). Exiting gracefully.")
    exit(0)
end

plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

system_sizes = unique(df.nodes)

for nodes in system_sizes
    df_nodes = df[df.nodes .== Ref(nodes), :]
    
    # Plot Total Time vs mu for different bond dimensions
    p = plot(
        xlabel = "μ",
        ylabel = "Total Time (s)",
        title = "Chorin Pipeline Preconditioning - QTT $(nodes[1])x$(nodes[2])",
        xaxis = :log10,
        yaxis = :log10,
        legend = :outertopright,
        framestyle = :box
    )
    
    gdims = sort(unique(filter(!ismissing, df_nodes.G_maxdim)))
    colors = palette(:tab20)
    
    for (i, gdim) in enumerate(gdims)
        df_gdim = df_nodes[coalesce.(df_nodes.G_maxdim .== gdim, false), :]
        sort!(df_gdim, :mu)
        
        plot!(p, Float64.(df_gdim.mu), df_gdim.total_solve_time .+ df_gdim.precond_build_time, label="χ=$(gdim)", lw=2, marker=:rect, color=colors[mod1(i, length(colors))])
    end
    
    # Baseline
    df_base = df_nodes[df_nodes.key .== "baseline", :]
    if nrow(df_base) > 0
        hline!(p, [df_base.total_solve_time[1]], color=:black, ls=:dash, label="Baseline (No PC)")
    end
    
    savefig(p, joinpath(plots_folder, "total_time_per_bond_nodes_$(nodes[1]).png"))
end
