"""
    QTT Poisson FAI accuracy vs the exact 1D inverse.

    This benchmark does not record Krylov condition-number stats
    (`conds_per_site` / `pc_conds_per_site`).
"""

using JLD2
using DataFrames
using Plots

gr()
default(fmt=:png, dpi=300, size=(800, 600), fontfamily="Computer Modern", margin=5Plots.mm)

base_folder = "experiments/preconditioner_benchmarks/qtt_poisson"
input_file = joinpath(base_folder, "results", "collected_results.jld2")

df = load(input_file, "df")

if ncol(df) == 0 || nrow(df) == 0
    println("Warning: Dataframe is empty or has no columns in $(input_file). Exiting gracefully.")
    exit(0)
end

plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

# We want to plot relative_error vs gdim for different mu
system_sizes = unique(df.nodes)

error_metrics = [
    (:relative_error, "Exact Inverse", "rel_error"),
    (:relative_error_iden, "Identity", "rel_error_iden")
]

for (err_col, err_name, err_prefix) in error_metrics
    if string(err_col) ∉ names(df)
        println("Warning: Column $err_col not found in dataframe, skipping plots.")
        continue
    end
    
    for nodes in system_sizes
        df_nodes = df[df.nodes .== Ref(nodes), :]
        
        p = plot(
            xlabel = "FAI Bond Dimension",
            ylabel = "Relative Error to $(err_name)",
            title = "FAI vs $(err_name) (1D Poisson) - QTT L=$(nodes[1])",
            legend = :outertopright,
            framestyle = :box
        )
        
        mu_values = sort(unique(df_nodes.mu))
        colors = palette(:tab20)
        
        for (i, mu) in enumerate(mu_values)
            df_mu = df_nodes[df_nodes.mu .== mu, :]
            sort!(df_mu, :G_maxdim)
            
            plot!(p, df_mu.G_maxdim, df_mu[!, err_col], label="μ=$(mu)", lw=2, marker=:circle, color=colors[mod1(i, length(colors))])
        end
        
        savefig(p, joinpath(plots_folder, "$(err_prefix)_nodes_$(nodes[1]).png"))
    end
end

# --- Plot 2: relative_error vs system_size for different G_maxdim ---
mu_values_all = sort(unique(df.mu))
gdims_all = sort(unique(df.G_maxdim))

for (err_col, err_name, err_prefix) in error_metrics
    if string(err_col) ∉ names(df)
        continue
    end

    for mu in mu_values_all
        df_mu = df[df.mu .== mu, :]
        
        p2 = plot(
            xlabel = "System Size (L)",
            ylabel = "Relative Error to $(err_name)",
            title = "FAI vs $(err_name) (1D Poisson) - QTT μ=$(mu)",
            legend = :outertopright,
            framestyle = :box,
            xticks = [n[1] for n in sort(unique(df.nodes))]
        )
        
        colors = palette(:tab20)
        
        for (i, gdim) in enumerate(gdims_all)
            df_gdim = df_mu[df_mu.G_maxdim .== gdim, :]
            sort!(df_gdim, :nodes)
            
            if nrow(df_gdim) > 0
                x_vals = [n[1] for n in df_gdim.nodes]
                y_vals = df_gdim[!, err_col]
                
                plot!(p2, x_vals, y_vals, label="G_maxdim=$(gdim)", lw=2, marker=:circle, color=colors[mod1(i, length(colors))])
            end
        end
        
        savefig(p2, joinpath(plots_folder, "$(err_prefix)_syssize_mu_$(mu).png"))
    end
end
