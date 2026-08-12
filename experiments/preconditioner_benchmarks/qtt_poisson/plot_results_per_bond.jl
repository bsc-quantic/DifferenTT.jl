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

system_sizes = unique(df.nodes)

for nodes in system_sizes
    df_nodes = df[df.nodes .== Ref(nodes), :]
    
    p = plot(
        xlabel = "μ",
        ylabel = "Relative Error to Exact Inverse",
        title = "FAI vs Exact Inverse (1D Poisson) - QTT L=$(nodes[1])",
        xaxis = :log10,
        yaxis = :log10,
        legend = :outertopright,
        framestyle = :box
    )
    
    gdims = sort(unique(df_nodes.G_maxdim))
    colors = palette(:tab20)
    
    for (i, gdim) in enumerate(gdims)
        df_gdim = df_nodes[df_nodes.G_maxdim .== gdim, :]
        sort!(df_gdim, :mu)
        
        plot!(p, Float64.(df_gdim.mu), df_gdim.relative_error, label="χ=$(gdim)", lw=2, marker=:rect, color=colors[mod1(i, length(colors))])
    end
    
    savefig(p, joinpath(plots_folder, "rel_error_per_bond_nodes_$(nodes[1]).png"))
end
