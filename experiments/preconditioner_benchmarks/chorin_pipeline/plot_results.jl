"""
    FAI Preconditioner Benchmark: Evolution Plotting Script

    Reads the evolution distributed JLD2 results and generates comprehensive
    visualizations using Plots.jl, including:
    1. Heatmaps of Aggregate Solve/Total Speedup (μ vs G_maxdim)
    2. Pareto fronts of Build Time vs Total Solve Time
    3. Step-by-step Iteration profiles
    4. Step-by-step Speedup profiles
    5. Scaling behavior across system sizes
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
base_folder = "experiments/preconditioner_benchmarks/chorin_pipeline"
nocond_folder = joinpath(base_folder, "results_nocond")
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

cond_folder = joinpath(base_folder, "results")

results_file_nocond = joinpath(nocond_folder, "collected_results.jld2")
@info "Loading nocond dataframe from $results_file_nocond"
df_nocond = load(results_file_nocond, "df")

results_file_cond = joinpath(cond_folder, "collected_results.jld2")
if isfile(results_file_cond)
    @info "Loading cond dataframe from $results_file_cond"
    df_cond = load(results_file_cond, "df")
else
    @warn "No cond dataframe found at $results_file_cond. Condition numbers will not be plotted."
    df_cond = DataFrame()
end

results_nocond = Dict()
for row in eachrow(df_nocond)
    nodes = row.nodes
    if !haskey(results_nocond, nodes)
        results_nocond[nodes] = Dict()
    end
    k = row.key
    d = Dict(
        "mu" => ismissing(row.mu) ? NaN : row.mu,
        "G_maxdim" => ismissing(row.G_maxdim) ? 0 : row.G_maxdim,
        "precond_build_time" => row.precond_build_time,
        "total_solve_time" => row.total_solve_time,
        "solve_times" => row.solve_times,
        "iters" => [mean(step_iters) for step_iters in row.iters],
        "iters_per_site" => hasproperty(row, :iters_per_site) ? row.iters_per_site : nothing,
        "n_steps" => length(row.solve_times)
    )
    results_nocond[nodes][k] = d
end

results_cond = Dict()
if !isempty(df_cond)
    for row in eachrow(df_cond)
        nodes = row.nodes
        if !haskey(results_cond, nodes)
            results_cond[nodes] = Dict()
        end
        k = row.key
        d_cond = Dict(
            "mu" => ismissing(row.mu) ? NaN : row.mu,
            "G_maxdim" => ismissing(row.G_maxdim) ? 0 : row.G_maxdim,
            "avg_cond" => hasproperty(row, :avg_cond) ? row.avg_cond : NaN,
            "avg_cond_improv" => hasproperty(row, :avg_cond_improv) ? row.avg_cond_improv : NaN,
            "conds_per_site" => hasproperty(row, :conds_per_site) ? row.conds_per_site : Float64[],
            "cond_improv_per_site" => hasproperty(row, :cond_improv_per_site) ? row.cond_improv_per_site : Float64[]
        )
        results_cond[nodes][k] = d_cond
    end
end

system_sizes = sort(collect(keys(results_nocond)))

max_total_speedups = Float64[]
max_solve_speedups = Float64[]
grid_labels = String[]

for nodes in system_sizes
    @info "Generating plots for system size $nodes"

    grid_data_nocond = results_nocond[nodes]
    grid_data_cond = get(results_cond, nodes, Dict())

    if !haskey(grid_data_nocond, "baseline")
        @warn "Skipping $nodes: Missing baseline data in nocond results"
        continue
    end

    base = grid_data_nocond["baseline"]
    base_time = base["total_solve_time"]
    base_iters = base["iters"]
    base_solve_times = base["solve_times"]
    n_steps = base["n_steps"]

    base_conds = nothing
    if haskey(grid_data_cond, "baseline")
        base_conds = get(grid_data_cond["baseline"], "conds_per_site", nothing)
    end

    mus = Float64[]
    gdims = Int[]
    solve_speedups = Float64[]
    total_speedups = Float64[]
    build_times = Float64[]
    total_solve_times = Float64[]
    keys_list = String[]

    avg_cond_improvements = Float64[]
    
    for (k, v) in grid_data_nocond
        if k == "baseline"
            continue
        end
        push!(keys_list, k)
        push!(mus, v["mu"])
        push!(gdims, v["G_maxdim"])
        push!(build_times, v["precond_build_time"])
        push!(total_solve_times, v["total_solve_time"])
        
        sp_solve = base_time / v["total_solve_time"]
        sp_total = base_time / (v["total_solve_time"] + v["precond_build_time"])
        
        push!(solve_speedups, sp_solve)
        push!(total_speedups, sp_total)

        cond_v = nothing
        for (ck, cv) in grid_data_cond
            if ck != "baseline" && cv["mu"] == v["mu"] && cv["G_maxdim"] == v["G_maxdim"]
                cond_v = cv
                break
            end
        end

        if !isnothing(cond_v) && haskey(cond_v, "avg_cond_improv")
            push!(avg_cond_improvements, cond_v["avg_cond_improv"])
        elseif !isnothing(cond_v) && haskey(cond_v, "avg_cond")
            push!(avg_cond_improvements, cond_v["avg_cond"])
        else
            push!(avg_cond_improvements, NaN)
        end
    end

    if isempty(mus)
        @warn "No preconditioned runs found for $nodes"
        continue
    end

    # Track for scaling plot
    push!(max_total_speedups, maximum(total_speedups))
    push!(max_solve_speedups, maximum(solve_speedups))
    push!(grid_labels, "$(nodes[1])x$(nodes[2])")

    unique_mus = sort(unique(mus))
    unique_gdims = sort(unique(gdims))

    Z_solve = fill(NaN, length(unique_gdims), length(unique_mus))
    Z_total = fill(NaN, length(unique_gdims), length(unique_mus))
    Z_cond = fill(NaN, length(unique_gdims), length(unique_mus))

    for i in 1:length(mus)
        μ_idx = findfirst(==(mus[i]), unique_mus)
        g_idx = findfirst(==(gdims[i]), unique_gdims)
        Z_solve[g_idx, μ_idx] = solve_speedups[i]
        Z_total[g_idx, μ_idx] = total_speedups[i]
        Z_cond[g_idx, μ_idx] = avg_cond_improvements[i]
    end

    # Heatmap: Aggregate Solve Speedup
    hm_solve = heatmap(string.(unique_mus), string.(unique_gdims), Z_solve,
        title="Aggregate Solve Speedup ($nodes, N=$n_steps)",
        xlabel="Regularization parameter (μ)",
        ylabel=L"Bond Dimension ($\chi_{max}$)",
        color=:viridis,
        clim=(minimum(filter(!isnan, Z_solve)), maximum(filter(!isnan, Z_solve))),
        right_margin=5Plots.mm)

    savefig(hm_solve, joinpath(plots_dir, "heatmap_solve_$(nodes[1])x$(nodes[2]).png"))

    # Heatmap: Aggregate Total Speedup
    hm_total = heatmap(string.(unique_mus), string.(unique_gdims), Z_total,
        title="Aggregate Total Speedup ($nodes, N=$n_steps)",
        xlabel="Regularization parameter (μ)",
        ylabel=L"Bond Dimension ($\chi_{max}$)",
        color=:viridis,
        clim=(minimum(filter(!isnan, Z_total)), maximum(filter(!isnan, Z_total))),
        right_margin=5Plots.mm)

    savefig(hm_total, joinpath(plots_dir, "heatmap_total_$(nodes[1])x$(nodes[2]).png"))

    valid_conds = filter(!isnan, Z_cond)
    if !isempty(valid_conds)
        hm_cond = heatmap(string.(unique_mus), string.(unique_gdims), Z_cond,
            title="Avg Cond Improvement ($nodes, N=$n_steps)",
            xlabel="Regularization parameter (μ)",
            ylabel=L"Bond Dimension ($\chi_{max}$)",
            color=:viridis,
            clim=(minimum(valid_conds), maximum(valid_conds)),
            right_margin=5Plots.mm)

        savefig(hm_cond, joinpath(plots_dir, "heatmap_cond_improvement_$(nodes[1])x$(nodes[2]).png"))
    else
        @info "Skipping condition heatmap for $nodes (no cond data found)"
    end

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
        scatter!(pf_plot, build_times[idx], total_solve_times[idx], label=string(gdim), color=pf_colors[mod1(i, length(pf_colors))])
    end

    hline!(pf_plot, [base_time], label="Baseline Total Solve", color=:red, lw=2, ls=:dash)
    plot!(pf_plot, [0, base_time], [base_time, 0],
        fillrange=0, fillalpha=0.1, fillcolor=:green,
        linealpha=0, label="Net Speedup Region")

    xlims!(pf_plot, 0, maximum(build_times) * 1.05)
    ylims!(pf_plot, 0, max(maximum(total_solve_times), base_time) * 1.05)

    savefig(pf_plot, joinpath(plots_dir, "pareto_$(nodes[1])x$(nodes[2]).png"))

    # Step-by-step profile
    best_idx = argmax(solve_speedups)
    best_key = keys_list[best_idx]
    best_v = grid_data_nocond[best_key]

    steps_range = 1:n_steps

    # Iterations over time
    p_iters = plot(steps_range, base_iters,
        label="Baseline", color=:red, lw=2, marker=:circle,
        title=latexstring("Krylov Iterations vs Time Step ($nodes)\nBest config: μ=$(best_v["mu"]), \$\\chi_{max}=$(best_v["G_maxdim"])\$"),
        xlabel="Chorin Time Step",
        ylabel="Avg Krylov Iterations per Site",
        legend=:outertopright)

    plot!(p_iters, steps_range, best_v["iters"],
        label="Preconditioned", color=:blue, lw=2, marker=:square)

    plot!(p_iters, steps_range, base_iters, fillrange=best_v["iters"],
        fillalpha=0.2, fillcolor=:green, label="Iterations Saved", linealpha=0)

    savefig(p_iters, joinpath(plots_dir, "step_iterations_$(nodes[1])x$(nodes[2]).png"))

    # Speedup over time
    per_step_speedup = base_solve_times ./ best_v["solve_times"]
    p_sp = plot(steps_range, per_step_speedup,
        label="Per-Step Solve Speedup", color=:purple, lw=2, marker=:diamond,
        title=latexstring("Solve Speedup vs Time Step ($nodes)\nBest config: μ=$(best_v["mu"]), \$\\chi_{max}=$(best_v["G_maxdim"])\$"),
        xlabel="Chorin Time Step",
        ylabel="Speedup Ratio",
        legend=:outertopright)

    hline!(p_sp, [1.0], color=:red, lw=2, ls=:dash, label="Break-even (1.0x)")
    
    savefig(p_sp, joinpath(plots_dir, "step_speedup_$(nodes[1])x$(nodes[2]).png"))

    # Cond per site profile
    best_v_cond = nothing
    for (ck, cv) in grid_data_cond
        if ck != "baseline" && cv["mu"] == best_v["mu"] && cv["G_maxdim"] == best_v["G_maxdim"]
            best_v_cond = cv
            break
        end
    end

    if !isnothing(base_conds) && !isnothing(best_v_cond)
        sites_range_cond = 1:length(base_conds)
        base_conds_clean = [c > 0 ? c : NaN for c in base_conds]
        best_cond_key = haskey(best_v_cond, "cond_improv_per_site") && !isempty(best_v_cond["cond_improv_per_site"]) ? "cond_improv_per_site" : "conds_per_site"
        best_conds_clean = [c > 0 ? c : NaN for c in best_v_cond[best_cond_key]]

        precond_absolute = base_conds_clean ./ best_conds_clean

        p_conds = plot(sites_range_cond, base_conds_clean,
            label="Baseline", color=:red, lw=2, marker=:circle, yaxis=:log,
            title="Local Condition Number Profile ($nodes)",
            xlabel="MPS Site Index",
            ylabel="Condition Number (Log Scale)",
            legend=:outertopright)

        plot!(p_conds, sites_range_cond, precond_absolute,
            label="Preconditioned", color=:blue, lw=2, marker=:square)

        savefig(p_conds, joinpath(plots_dir, "site_condition_$(nodes[1])x$(nodes[2]).png"))
    end

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
