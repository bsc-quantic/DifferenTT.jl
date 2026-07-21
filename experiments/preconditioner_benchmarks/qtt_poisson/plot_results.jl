using JLD2
using DataFrames
using CairoMakie

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/qtt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/qtt_poisson/results"
input_file = joinpath(base_folder, "collected_results.jld2")

df = load(input_file, "df")

plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

# We want to plot relative_error vs gdim for different mu
system_sizes = unique(df.nodes)

for nodes in system_sizes
    df_nodes = df[df.nodes .== Ref(nodes), :]
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel = "FAI Bond Dimension",
        ylabel = "Relative Error to Exact Inverse",
        title = "FAI vs Exact Inverse (1D Poisson) - QTT L=$(nodes[1])",
        yscale = log10
    )
    
    mu_values = sort(unique(df_nodes.mu))
    
    for mu in mu_values
        df_mu = df_nodes[df_nodes.mu .== mu, :]
        sort!(df_mu, :G_maxdim)
        
        lines!(ax, df_mu.G_maxdim, df_mu.relative_error, label="μ=$(mu)", linewidth=2, marker=:circle)
    end
    
    axislegend(ax, position=:rt)
    
    save(joinpath(plots_folder, "rel_error_nodes_$(nodes[1]).png"), fig)
end
