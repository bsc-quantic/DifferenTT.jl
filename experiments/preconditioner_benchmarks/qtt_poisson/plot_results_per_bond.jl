using JLD2
using DataFrames
using CairoMakie

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/qtt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/qtt_poisson/results"
input_file = joinpath(base_folder, "collected_results.jld2")

df = load(input_file, "df")
plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

system_sizes = unique(df.nodes)

for nodes in system_sizes
    df_nodes = df[df.nodes .== Ref(nodes), :]
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel = "μ",
        ylabel = "Relative Error to Exact Inverse",
        title = "FAI vs Exact Inverse (1D Poisson) - QTT L=$(nodes[1])",
        xscale = log10,
        yscale = log10
    )
    
    gdims = sort(unique(df_nodes.G_maxdim))
    
    for gdim in gdims
        df_gdim = df_nodes[df_nodes.G_maxdim .== gdim, :]
        sort!(df_gdim, :mu)
        
        lines!(ax, df_gdim.mu, df_gdim.relative_error, label="χ=$(gdim)", linewidth=2, marker=:rect)
    end
    
    axislegend(ax, position=:lt)
    
    save(joinpath(plots_folder, "rel_error_per_bond_nodes_$(nodes[1]).png"), fig)
end
