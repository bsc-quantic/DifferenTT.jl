using JLD2
using DataFrames
using Plots
using Statistics

gr()
default(fmt=:png, dpi=300, size=(900, 500), fontfamily="Computer Modern", margin=5Plots.mm)

base_folder = "experiments/preconditioner_benchmarks/chorin_pipeline_globalcg/results"
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
    
    df_base = df_nodes[df_nodes.key .== "baseline", :]
    base_t = nrow(df_base) > 0 ? df_base.total_solve_time[1] : 0.0
    
    df_fai = df_nodes[df_nodes.key .!= "baseline", :]
    if nrow(df_fai) == 0
        continue
    end
    
    mu_values = sort(unique(filter(!ismissing, df_fai.mu)))
    gdim_values = sort(unique(filter(!ismissing, df_fai.G_maxdim)))
    
    # 1. Heatmap (Solve Speedup)
    speedup_mat = fill(NaN, length(gdim_values), length(mu_values))
    for (i, mu) in enumerate(mu_values)
        for (j, gdim) in enumerate(gdim_values)
            row = df_fai[(df_fai.mu .== mu) .& (df_fai.G_maxdim .== gdim), :]
            if nrow(row) > 0
                speedup_mat[j, i] = base_t / row.total_solve_time[1]
            end
        end
    end
    
    valid_speedups = filter(!isnan, speedup_mat)
    if !isempty(valid_speedups)
        hm = heatmap(string.(mu_values), string.(gdim_values), speedup_mat,
            title = "Solve Speedup - $(nodes[1])x$(nodes[2])",
            xlabel = "mu (μ)",
            ylabel = "G_maxdim (χ)",
            color = :viridis,
            clim = (minimum(valid_speedups), maximum(valid_speedups)),
            right_margin = 5Plots.mm)
        savefig(hm, joinpath(plots_folder, "heatmap_speedup_$(nodes[1]).png"))
    end
    
    # 2. Pareto Front (Build vs Solve)
    pf_plot = plot(title="Pareto Front - $(nodes[1])x$(nodes[2])",
        xlabel="Preconditioner Build Time (s)",
        ylabel="Solve Time (s)",
        legend=:outertopright,
        framestyle=:box)
        
    hline!(pf_plot, [base_t], label="Baseline Solve Time", color=:red, lw=2, ls=:dash)
    
    # Shade the "profit" region (total time < baseline time)
    if base_t > 0
        plot!(pf_plot, [0, base_t], [base_t, 0],
            fillrange=0, fillalpha=0.1, fillcolor=:green,
            linealpha=0, label="Net Speedup Region")
    end
    
    pf_colors = palette(:tab20)
    for (i, gdim) in enumerate(gdim_values)
        df_g = df_fai[coalesce.(df_fai.G_maxdim .== gdim, false), :]
        if nrow(df_g) > 0
            scatter!(pf_plot, df_g.precond_build_time, df_g.total_solve_time, label="χ=$gdim", color=pf_colors[mod1(i, length(pf_colors))], markersize=6)
        end
    end
    
    xlims!(pf_plot, 0, maximum(df_fai.precond_build_time) * 1.05)
    ylims!(pf_plot, 0, max(maximum(df_fai.total_solve_time), base_t) * 1.05)
    
    savefig(pf_plot, joinpath(plots_folder, "pareto_$(nodes[1]).png"))
    
    # 3. Iteration Profile
    p_iter = plot(title="Site-by-Site Iteration Profile - $(nodes[1])x$(nodes[2])",
        xlabel="MPS Site", ylabel="Total Krylov Iterations", legend=:outertopright)
        
    if nrow(df_base) > 0 && hasproperty(df_base, :iters_per_site) && !ismissing(df_base.iters_per_site[1])
        iters_b = df_base.iters_per_site[1]
        plot!(p_iter, 1:length(iters_b), iters_b, label="Baseline", color=:black, lw=3, marker=:circle)
    end
    
    if hasproperty(df_fai, :iters_per_site)
        best_fai = df_fai[argmin(df_fai.total_solve_time), :]
        iters_f = best_fai.iters_per_site
        if !ismissing(iters_f)
            plot!(p_iter, 1:length(iters_f), iters_f, label="Best FAI (μ=$(best_fai.mu), χ=$(best_fai.G_maxdim))", color=:blue, lw=3, marker=:square)
            if nrow(df_base) > 0 && !ismissing(df_base.iters_per_site[1])
                plot!(p_iter, 1:length(iters_f), df_base.iters_per_site[1], fillrange=iters_f, fillalpha=0.2, fillcolor=:green, label="Iterations Saved", linealpha=0)
            end
        end
    end
    savefig(p_iter, joinpath(plots_folder, "iters_profile_$(nodes[1]).png"))
end
