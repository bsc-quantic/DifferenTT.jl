using JLD2
using DataFrames
using CairoMakie

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/chorin_pipeline/results_nocond" : "experiments/preconditioner_benchmarks/chorin_pipeline/results"
input_file = joinpath(base_folder, "collected_results.jld2")

df = load(input_file, "df")

plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

system_sizes = unique(df.nodes)

for nodes in system_sizes
    df_nodes = df[df.nodes .== Ref(nodes), :]
    
    # Plot Total Time vs mu for different bond dimensions
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel = "μ",
        ylabel = "Total Time (s)",
        title = "Chorin Pipeline Preconditioning - QTT $(nodes[1])x$(nodes[2])",
        xscale = log10,
        yscale = log10
    )
    
    gdims = sort(unique(filter(!ismissing, df_nodes.G_maxdim)))
    
    for gdim in gdims
        df_gdim = df_nodes[coalesce.(df_nodes.G_maxdim .== gdim, false), :]
        sort!(df_gdim, :mu)
        
        lines!(ax, Float64.(df_gdim.mu), df_gdim.total_solve_time .+ df_gdim.precond_build_time, label="χ=$(gdim)", linewidth=2, marker=:rect)
    end
    
    # Baseline
    df_base = df_nodes[df_nodes.key .== "baseline", :]
    if nrow(df_base) > 0
        hlines!(ax, [df_base.total_solve_time[1]], color=:black, linestyle=:dash, label="Baseline (No PC)")
    end
    
    axislegend(ax, position=:lt)
    
    save(joinpath(plots_folder, "total_time_per_bond_nodes_$(nodes[1]).png"), fig)
end
