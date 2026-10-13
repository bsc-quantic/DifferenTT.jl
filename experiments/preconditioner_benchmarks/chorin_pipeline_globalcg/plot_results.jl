"""
    FAI Preconditioner Benchmark: Evolution Plotting Script

    Reads the evolution distributed JLD2 results and generates comprehensive
    visualizations using Plots.jl, including:
    1. Heatmaps of Aggregate Solve/Total Speedup (μ vs G_maxdim)
    2. Pareto fronts of Build Time vs Total Solve Time
    3. Step-by-step Iteration profiles
    4. Step-by-step Speedup profiles
    5. Step-by-step relative residual of every configuration
    6. Step-by-step divergence norm ratio (after / before the pressure correction) of every configuration
    7. Step-by-step bond dimension of the pressure field of every configuration
    8. Scaling behavior across system sizes

    Global CG does not record local Krylov condition numbers.
"""

using JLD2
using DataFrames
using Plots
using Statistics
using Printf
using LaTeXStrings

# Use GR backend
gr()
default(fmt=:png, dpi=300, size=(900, 500), fontfamily="Computer Modern", margin=5Plots.mm)


# Configuration
ENV["GKSwstype"] = "100"
base_folder = "experiments/preconditioner_benchmarks/chorin_pipeline_globalcg/results"
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

include(joinpath(@__DIR__, "..", "plot_cond_utils.jl"))

results_file = joinpath(base_folder, "collected_results.jld2")
@info "Loading dataframe from $results_file"
df = load(results_file, "df")

results = Dict()
for row in eachrow(df)
    nodes = row.nodes
    if !haskey(results, nodes)
        results[nodes] = Dict()
    end
    k = row.key
    d = Dict(
        "mu" => hasproperty(row, :mu) && !ismissing(row.mu) ? row.mu : NaN,
        "G_maxdim" => hasproperty(row, :G_maxdim) && !ismissing(row.G_maxdim) ? row.G_maxdim : 0,
        "precond_build_time" => row.precond_build_time,
        "total_solve_time" => row.total_solve_time,
        "solve_times" => row.solve_times,
        "iters" => row.global_iters,
        "residuals" => row.residuals,
        "div_ratios" => row_vec(row, :div_ratios),
        "solution_maxdims" => row_vec(row, :solution_maxdims),
        "n_steps" => length(row.solve_times)
    )
    results[nodes][k] = d
end

system_sizes = sort(collect(keys(results)))

max_total_speedups = Float64[]
max_solve_speedups = Float64[]
grid_labels = String[]

for nodes in system_sizes
    @info "Generating plots for system size $nodes"

    grid_data = results[nodes]

    has_baseline = haskey(grid_data, "baseline")
    
    if !has_baseline
        @warn "Missing baseline data for $nodes. Absolute times will be plotted instead of speedups."
        first_v = first(values(grid_data))
        n_steps = first_v["n_steps"]
        base_time = NaN
        base_iters = fill(NaN, n_steps)
        base_solve_times = fill(NaN, n_steps)
        base_residuals = fill(NaN, n_steps)
    else
        base = grid_data["baseline"]
        base_time = base["total_solve_time"]
        base_iters = base["iters"]
        base_solve_times = base["solve_times"]
        base_residuals = base["residuals"]
        n_steps = base["n_steps"]
    end

    mus = Float64[]
    gdims = Int[]
    solve_speedups = Float64[]
    total_speedups = Float64[]
    build_times = Float64[]
    total_solve_times = Float64[]
    keys_list = String[]

    for (k, v) in grid_data
        if k == "baseline"
            continue
        end
        push!(keys_list, k)
        push!(mus, v["mu"])
        push!(gdims, v["G_maxdim"])
        push!(build_times, v["precond_build_time"])
        push!(total_solve_times, v["total_solve_time"])
        if has_baseline
            sp_solve = base_time / v["total_solve_time"]
            sp_total = base_time / (v["total_solve_time"] + v["precond_build_time"])
        else
            sp_solve = v["total_solve_time"]
            sp_total = v["total_solve_time"] + v["precond_build_time"]
        end
        
        push!(solve_speedups, sp_solve)
        push!(total_speedups, sp_total)
    end

    if isempty(mus)
        @warn "No preconditioned runs found for $nodes"
        continue
    end

    # Track for scaling plot
    if has_baseline
        push!(max_total_speedups, maximum(total_speedups))
        push!(max_solve_speedups, maximum(solve_speedups))
        push!(grid_labels, "$(nodes[1])x$(nodes[2])")
    end

    unique_mus = sort(unique(mus))
    unique_gdims = sort(unique(gdims))

    Z_solve = fill(NaN, length(unique_gdims), length(unique_mus))
    Z_total = fill(NaN, length(unique_gdims), length(unique_mus))

    for i in 1:length(mus)
        μ_idx = findfirst(==(mus[i]), unique_mus)
        g_idx = findfirst(==(gdims[i]), unique_gdims)
        Z_solve[g_idx, μ_idx] = solve_speedups[i]
        Z_total[g_idx, μ_idx] = total_speedups[i]
    end

    hm_solve_title = has_baseline ? "Aggregate Solve Speedup ($nodes, N=$n_steps)" : "Aggregate Solve Time [s] ($nodes, N=$n_steps)"
    # Heatmap: Aggregate Solve Speedup/Time
    hm_solve = heatmap(string.(unique_mus), string.(unique_gdims), Z_solve,
        title=hm_solve_title,
        xlabel="Regularization parameter (μ)",
        ylabel=L"Bond Dimension ($\chi_{max}$)",
        color=:viridis,
        clim=(minimum(filter(!isnan, Z_solve)), maximum(filter(!isnan, Z_solve))),
        right_margin=5Plots.mm)

    savefig(hm_solve, joinpath(plots_dir, "heatmap_solve_$(nodes[1])x$(nodes[2]).png"))

    hm_total_title = has_baseline ? "Aggregate Total Speedup ($nodes, N=$n_steps)" : "Aggregate Total Time [s] ($nodes, N=$n_steps)"
    # Heatmap: Aggregate Total Speedup/Time
    hm_total = heatmap(string.(unique_mus), string.(unique_gdims), Z_total,
        title=hm_total_title,
        xlabel="Regularization parameter (μ)",
        ylabel=L"Bond Dimension ($\chi_{max}$)",
        color=:viridis,
        clim=(minimum(filter(!isnan, Z_total)), maximum(filter(!isnan, Z_total))),
        right_margin=5Plots.mm)

    savefig(hm_total, joinpath(plots_dir, "heatmap_total_$(nodes[1])x$(nodes[2]).png"))

    # Pareto Front
    pf_plot = plot(title="Trade-off: Build vs Total Solve Time ($nodes, N=$n_steps)",
        xlabel="Preconditioner Build Time (s)",
        ylabel="Aggregate Linear Solve Time (s)",
        legend=:outertopright,
        legendtitle=L"$\chi_{max}$",
        framestyle=:box)
    pf_colors = palette(:tab20)
    for (i, gdim) in enumerate(unique_gdims)
        idx = findall(x -> x == gdim, gdims)
        scatter!(pf_plot, build_times[idx], total_solve_times[idx],
            label=latexstring("FAI \$\\chi_{max}=$(gdim)\$"),
            color=pf_colors[mod1(i, length(pf_colors))], marker=:circle)
    end

    if has_baseline
        hline!(pf_plot, [base_time], label="Baseline Total Solve", color=:red, lw=2, ls=:dash)
        plot!(pf_plot, [0, base_time], [base_time, 0],
            fillrange=0, fillalpha=0.1, fillcolor=:green,
            linealpha=0, label="Net Speedup Region")
    
        xlims!(pf_plot, 0, maximum(build_times) * 1.05)
        ylims!(pf_plot, 0, max(maximum(total_solve_times), base_time) * 1.05)
    else
        if maximum(build_times) > 0
            xlims!(pf_plot, 0, maximum(build_times) * 1.05)
        end
        if maximum(total_solve_times) > 0
            ylims!(pf_plot, 0, maximum(total_solve_times) * 1.05)
        end
    end

    savefig(pf_plot, joinpath(plots_dir, "pareto_$(nodes[1])x$(nodes[2]).png"))

    # Step-by-step profile
    steps_range = 1:n_steps
    colors = palette(:tab20)

    # Iterations over time
    p_iters = plot(title=latexstring("Krylov Iterations vs Time Step ($nodes)\nBest μ per \$\\chi_{max}\$"),
        xlabel="Chorin Time Step",
        ylabel="Avg Krylov Iterations per Site",
        legend=:outertopright,
        yaxis=:log10)

    if has_baseline
        plot!(p_iters, steps_range, base_iters,
            label="Baseline", color=:black, lw=3, marker=:circle)
    end

    # Speedup over time
    p_sp_title = has_baseline ? latexstring("Solve Speedup vs Time Step ($nodes)\nBest μ per \$\\chi_{max}\$") : latexstring("Solve Time vs Time Step ($nodes)\nBest μ per \$\\chi_{max}\$")
    p_sp_ylabel = has_baseline ? "Speedup Ratio" : "Solve Time (s)"

    p_sp = plot(title=p_sp_title,
        xlabel="Chorin Time Step",
        ylabel=p_sp_ylabel,
        legend=:outertopright,
        yaxis=:log10)

    if has_baseline
        hline!(p_sp, [1.0], color=:black, lw=3, ls=:dash, label="Break-even (1.0x)")
    end

    # Residuals over time
    p_res = plot(title=latexstring("Relative Residual vs Time Step ($nodes)\nBest μ per \$\\chi_{max}\$"),
        xlabel="Chorin Time Step",
        ylabel="Relative Residual",
        legend=:outertopright,
        yaxis=:log10)

    if has_baseline
        plot!(p_res, steps_range, base_residuals,
            label="Baseline", color=:black, lw=3, marker=:circle)
    end

    for (i, gdim) in enumerate(unique_gdims)
        best_idx = -1
        if has_baseline
            best_val = -Inf
            for idx in 1:length(solve_speedups)
                if gdims[idx] == gdim && solve_speedups[idx] > best_val
                    best_val = solve_speedups[idx]
                    best_idx = idx
                end
            end
        else
            best_val = Inf
            for idx in 1:length(solve_speedups)
                if gdims[idx] == gdim && solve_speedups[idx] < best_val
                    best_val = solve_speedups[idx]
                    best_idx = idx
                end
            end
        end

        best_idx == -1 && continue
        
        best_v = grid_data[keys_list[best_idx]]
        c_color = colors[mod1(i, length(colors))]
        label_str = latexstring("FAI \$\\chi_{max}=$(gdim)\$ (μ=$(best_v["mu"]))")
        
        plot!(p_iters, steps_range, best_v["iters"], label=label_str, color=c_color, lw=2, marker=:square)
        plot!(p_res, steps_range, best_v["residuals"], label=label_str, color=c_color, lw=2, marker=:star5)
        
        if has_baseline
            per_step_metric = base_solve_times ./ best_v["solve_times"]
        else
            per_step_metric = best_v["solve_times"]
        end
        plot!(p_sp, steps_range, per_step_metric, label=label_str, color=c_color, lw=2, marker=:diamond)
    end

    savefig(p_iters, joinpath(plots_dir, "step_iterations_$(nodes[1])x$(nodes[2]).png"))
    savefig(p_sp, joinpath(plots_dir, "step_speedup_$(nodes[1])x$(nodes[2]).png"))
    savefig(p_res, joinpath(plots_dir, "step_residuals_$(nodes[1])x$(nodes[2]).png"))

    # Per-step residual, divergence reduction and pressure bond dimension of every configuration
    step_keys = sort(keys_list, by=k -> (grid_data[k]["G_maxdim"], grid_data[k]["mu"]))
    function step_series(metric)
        series = Any[]
        has_baseline && push!(series, (; label="Baseline", values=grid_data["baseline"][metric], color=:black, lw=3))
        for (j, k) in enumerate(step_keys)
            v = grid_data[k]
            push!(series, (; label=latexstring("\$\\chi=$(v["G_maxdim"]), \\mu=$(v["mu"])\$"),
                values=v[metric], color=colors[mod1(j, length(colors))], lw=1.5))
        end
        return series
    end
    p_res_all = step_profile_plot(step_series("residuals");
        title="Relative Residual vs Time Step ($nodes)\nAll Preconditioner Setups",
        ylabel="Relative Residual")
    p_div_all = step_profile_plot(step_series("div_ratios");
        title="Divergence Reduction by the Pressure Correction ($nodes)\nAll Preconditioner Setups",
        ylabel="Divergence Norm Ratio (after / before)")
    p_bond_all = step_profile_plot(step_series("solution_maxdims");
        title="Pressure Field Bond Dimension vs Time Step ($nodes)\nAll Preconditioner Setups",
        ylabel="Max Bond Dimension of p", yaxis=:identity)
    isnothing(p_res_all) || savefig(p_res_all, joinpath(plots_dir, "step_residuals_all_setups_$(nodes[1])x$(nodes[2]).png"))
    isnothing(p_div_all) || savefig(p_div_all, joinpath(plots_dir, "step_div_ratio_all_setups_$(nodes[1])x$(nodes[2]).png"))
    isnothing(p_bond_all) || savefig(p_bond_all, joinpath(plots_dir, "step_pressure_bonddim_all_setups_$(nodes[1])x$(nodes[2]).png"))

end

# Scaling View
if !isempty(max_total_speedups) && length(max_total_speedups) > 1
    p_scale = plot(1:length(grid_labels), max_solve_speedups,
        label="Max Agg. Solve Speedup", color=:blue, lw=3, marker=:circle, markersize=8,
        title="Scaling of Maximum Achievable Speedups (Evolution)",
        xlabel="System Size",
        ylabel="Speedup Factor",
        xticks=(1:length(grid_labels), grid_labels),
        legend=:topleft,
        framestyle=:box)

    plot!(p_scale, 1:length(grid_labels), max_total_speedups,
        label="Max Agg. Total Speedup", color=:green, lw=3, marker=:square, markersize=8)

    hline!(p_scale, [1.0], label="Break-even (1.0x)", color=:red, lw=2, ls=:dash)

    savefig(p_scale, joinpath(plots_dir, "scaling_summary_evolution.png"))
end

@info "All evolution plots successfully generated in $plots_dir"
