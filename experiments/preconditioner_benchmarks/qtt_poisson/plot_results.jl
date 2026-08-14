using JLD2
using DataFrames
using Plots

gr()
default(fmt=:png, dpi=300, size=(800, 600), fontfamily="Computer Modern", margin=5Plots.mm)

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/qtt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/qtt_poisson/results"
input_file = joinpath(base_folder, "collected_results.jld2")

df = load(input_file, "df")

if ncol(df) == 0 || nrow(df) == 0
    println("Warning: Dataframe is empty or has no columns in $(input_file). Exiting gracefully.")
    exit(0)
end

plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

# We want to plot relative_error vs gdim for different mu
system_sizes = unique(df.nodes)

for nodes in system_sizes
    df_nodes = df[df.nodes .== Ref(nodes), :]
    
    p = plot(
        xlabel = "FAI Bond Dimension",
        ylabel = "Relative Error to Exact Inverse",
        title = "FAI vs Exact Inverse (1D Poisson) - QTT L=$(nodes[1])",
        yaxis = :log10,
        legend = :outertopright,
        framestyle = :box
    )
    
    mu_values = sort(unique(df_nodes.mu))
    colors = palette(:tab20)
    
    for (i, mu) in enumerate(mu_values)
        df_mu = df_nodes[df_nodes.mu .== mu, :]
        sort!(df_mu, :G_maxdim)
        
        plot!(p, df_mu.G_maxdim, df_mu.relative_error, label="μ=$(mu)", lw=2, marker=:circle, color=colors[mod1(i, length(colors))])
    end
    
    savefig(p, joinpath(plots_folder, "rel_error_nodes_$(nodes[1]).png"))
end
