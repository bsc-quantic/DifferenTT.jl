using JLD2
using DataFrames
using CairoMakie
using Statistics

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/chorin_pipeline/results_nocond" : "experiments/preconditioner_benchmarks/chorin_pipeline/results"
input_file = joinpath(base_folder, "collected_results.jld2")

df = load(input_file, "df")
plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

system_sizes = unique(df.nodes)

for nodes in system_sizes
    df_nodes = df[df.nodes .== Ref(nodes), :]
    
    df_base = df_nodes[df_nodes.key .== "baseline", :]
    base_t = nrow(df_base) > 0 ? df_base.total_solve_time[1] : 0.0
    
    df_fai = df_nodes[df_nodes.key .!= "baseline", :]
    if nrow(df_fai) == 0
        continue
    end
    
    mu_values = sort(unique(filter(!ismissing, df_fai.mu)))
    gdim_values = sort(unique(filter(!ismissing, df_fai.G_maxdim)))
    
    # 1. Heatmap (Solve Speedup)
    speedup_mat = fill(NaN, length(mu_values), length(gdim_values))
    for (i, mu) in enumerate(mu_values)
        for (j, gdim) in enumerate(gdim_values)
            row = df_fai[(df_fai.mu .== mu) .& (df_fai.G_maxdim .== gdim), :]
            if nrow(row) > 0
                speedup_mat[i, j] = base_t / row.total_solve_time[1]
            end
        end
    end
    
    fig_heat = Figure(size=(800, 600))
    ax_heat = Axis(fig_heat[1, 1], xlabel = "G_maxdim (χ)", ylabel = "mu (μ)", title = "Solve Speedup Heatmap - $(nodes[1])x$(nodes[2])",
                   xticks = (1:length(gdim_values), string.(gdim_values)), 
                   yticks = (1:length(mu_values), string.(mu_values)))
    
    hm = heatmap!(ax_heat, speedup_mat', colormap=:viridis)
    Colorbar(fig_heat[1, 2], hm, label="Speedup Factor")
    save(joinpath(plots_folder, "heatmap_speedup_$(nodes[1]).png"), fig_heat)
    
    # 2. Pareto Front (Build vs Solve)
    fig_pareto = Figure(size=(800, 600))
    ax_pareto = Axis(fig_pareto[1, 1], xlabel = "Preconditioner Build Time (s)", ylabel = "Solve Time (s)", title = "Pareto Front - $(nodes[1])x$(nodes[2])")
    hlines!(ax_pareto, [base_t], color=:red, linestyle=:dash, label="Baseline Solve Time")
    scatter!(ax_pareto, df_fai.precond_build_time, df_fai.total_solve_time, color=:blue, markersize=12, label="FAI Preconditioners")
    axislegend(ax_pareto)
    save(joinpath(plots_folder, "pareto_$(nodes[1]).png"), fig_pareto)
    
    # 3. Iteration Profile
    if !use_nocond
        fig_iter = Figure(size=(900, 500))
        ax_iter = Axis(fig_iter[1, 1], xlabel = "MPS Site", ylabel = "Total Krylov Iterations", title = "Site-by-Site Iteration Profile - $(nodes[1])x$(nodes[2])")
        
        if nrow(df_base) > 0 && hasproperty(df_base, :iters_per_site)
            iters_b = df_base.iters_per_site[1]
            lines!(ax_iter, 1:length(iters_b), iters_b, label="Baseline", color=:black, linewidth=3)
        end
        
        if hasproperty(df_fai, :iters_per_site)
            best_fai = df_fai[argmin(df_fai.total_solve_time), :]
            iters_f = best_fai.iters_per_site
            lines!(ax_iter, 1:length(iters_f), iters_f, label="Best FAI (μ=$(best_fai.mu), χ=$(best_fai.G_maxdim))", color=:blue, linewidth=3)
            if nrow(df_base) > 0
                band!(ax_iter, 1:length(iters_f), iters_f, df_base.iters_per_site[1], color=(:blue, 0.2))
            end
        end
        axislegend(ax_iter)
        save(joinpath(plots_folder, "iters_profile_$(nodes[1]).png"), fig_iter)
    end
end
