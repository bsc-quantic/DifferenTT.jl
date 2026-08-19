"""
    FAI Preconditioner Benchmark: Evolution Plotting Script

    Reads the evolution distributed JLD2 results and generates comprehensive
    visualizations using Plots.jl, including:
    1. Heatmaps of Aggregate Solve/Total Speedup (μ vs G_maxdim)
    2. Pareto fronts of Build Time vs Total Solve Time
    3. Step-by-step Iteration profiles
    4. Step-by-step Speedup profiles
    5. Site-by-site condition-number profiles (κ absolute, not a ratio)
    6. Scaling behavior across system sizes
    Uses nocond results for accurate timings, and cond results for
    condition-number profiles.

    Note: `pc_conds_per_site` is the absolute local κ(P^{-1}A), not a ratio.
"""

using JLD2
using DataFrames
using Plots
using Statistics
using Printf
using LaTeXStrings

gr()
default(fmt=:png, dpi=300, size=(900, 500), fontfamily="Computer Modern", margin=5Plots.mm)

ENV["GKSwstype"] = "100"
base_folder = "experiments/preconditioner_benchmarks/chorin_pipeline"
nocond_folder = joinpath(base_folder, "results_nocond")
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

cond_folder = joinpath(base_folder, "results")

include(joinpath(@__DIR__, "..", "plot_cond_utils.jl"))

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
        "key" => k,
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
            "key" => k,
            "mu" => ismissing(row.mu) ? NaN : row.mu,
            "G_maxdim" => ismissing(row.G_maxdim) ? 0 : row.G_maxdim,
            "avg_cond" => hasproperty(row, :avg_cond) ? row.avg_cond : NaN,
            "avg_pc_cond" => hasproperty(row, :avg_pc_cond) ? row.avg_pc_cond : NaN,
            "conds_per_site" => row_vec(row, :conds_per_site),
            "pc_conds_per_site" => row_vec(row, :pc_conds_per_site)
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
    base_avg_cond = NaN
    if haskey(grid_data_cond, "baseline")
        base_conds = get(grid_data_cond["baseline"], "conds_per_site", nothing)
        ac = get(grid_data_cond["baseline"], "avg_cond", NaN)
        if ac isa Number && isfinite(Float64(ac))
            base_avg_cond = Float64(ac)
        end
    end

    mus = Float64[]
    gdims = Int[]
    solve_speedups = Float64[]
    total_speedups = Float64[]
    build_times = Float64[]
    total_solve_times = Float64[]
    keys_list = String[]
    cond_improvement_factors = Float64[]

    for (k, v) in grid_data_nocond
        k == "baseline" && continue
        push!(keys_list, k)
        push!(mus, v["mu"])
        push!(gdims, v["G_maxdim"])
        push!(build_times, v["precond_build_time"])
        push!(total_solve_times, v["total_solve_time"])

        push!(solve_speedups, base_time / v["total_solve_time"])
        push!(total_speedups, base_time / (v["total_solve_time"] + v["precond_build_time"]))

        cond_v = match_cond(grid_data_cond, v)
        push!(cond_improvement_factors, isnothing(cond_v) ? NaN : cond_improvement_factor(cond_v; baseline_avg=base_avg_cond))
    end

    if isempty(mus)
        @warn "No preconditioned runs found for $nodes"
        continue
    end

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
        Z_cond[g_idx, μ_idx] = cond_improvement_factors[i]
    end

    hm_solve = heatmap(string.(unique_mus), string.(unique_gdims), Z_solve,
        title="Aggregate Solve Speedup ($nodes, N=$n_steps)",
        xlabel="Regularization parameter (μ)",
        ylabel=L"Bond Dimension ($\chi_{max}$)",
        color=:viridis,
        clim=(minimum(filter(!isnan, Z_solve)), maximum(filter(!isnan, Z_solve))),
        right_margin=5Plots.mm)
    savefig(hm_solve, joinpath(plots_dir, "heatmap_solve_$(nodes[1])x$(nodes[2]).png"))

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
        @info "Skipping condition heatmap for $nodes (no FAI κ(P^{-1}A) data found)"
    end

    pf_plot = plot(title="Trade-off: Build vs Total Solve Time ($nodes, N=$n_steps)",
        xlabel="Preconditioner Build Time (s)",
        ylabel="Aggregate Linear Solve Time (s)",
        legend=:outertopright,
        framestyle=:box)
    pf_colors = palette(:tab20)
    for (i, gdim) in enumerate(unique_gdims)
        idx = findall(x -> x == gdim, gdims)
        scatter!(pf_plot, build_times[idx], total_solve_times[idx],
            label=latexstring("FAI \$\\chi_{max}=$(gdim)\$"),
            color=pf_colors[mod1(i, length(pf_colors))], marker=:circle)
    end

    hline!(pf_plot, [base_time], label="Baseline Total Solve", color=:red, lw=2, ls=:dash)
    plot!(pf_plot, [0, base_time], [base_time, 0],
        fillrange=0, fillalpha=0.1, fillcolor=:green,
        linealpha=0, label="Net Speedup Region")

    xlims!(pf_plot, 0, maximum(build_times) * 1.05)
    ylims!(pf_plot, 0, max(maximum(total_solve_times), base_time) * 1.05)
    savefig(pf_plot, joinpath(plots_dir, "pareto_$(nodes[1])x$(nodes[2]).png"))

    best_idx = argmax(solve_speedups)
    best_key = keys_list[best_idx]
    best_v = grid_data_nocond[best_key]
    steps_range = 1:n_steps

    p_iters = plot(steps_range, base_iters,
        label="Baseline", color=:red, lw=2, marker=:circle,
        title=latexstring("Krylov Iterations vs Time Step ($nodes)\nBest FAI: μ=$(best_v["mu"]), \$\\chi_{max}=$(best_v["G_maxdim"])\$"),
        xlabel="Chorin Time Step",
        ylabel="Avg Krylov Iterations per Site",
        legend=:outertopright)
    plot!(p_iters, steps_range, best_v["iters"],
        label="FAI", color=:blue, lw=2, marker=:square)
    plot!(p_iters, steps_range, base_iters, fillrange=best_v["iters"],
        fillalpha=0.12, fillcolor=:blue, label="", linealpha=0)
    savefig(p_iters, joinpath(plots_dir, "step_iterations_$(nodes[1])x$(nodes[2]).png"))

    per_step_speedup = base_solve_times ./ best_v["solve_times"]
    p_sp = plot(steps_range, per_step_speedup,
        label="FAI", color=:purple, lw=2, marker=:diamond,
        title=latexstring("Solve Speedup vs Time Step ($nodes)\nBest FAI: μ=$(best_v["mu"]), \$\\chi_{max}=$(best_v["G_maxdim"])\$"),
        xlabel="Chorin Time Step",
        ylabel="Speedup Ratio",
        legend=:outertopright)
    hline!(p_sp, [1.0], color=:red, lw=2, ls=:dash, label="Break-even (1.0x)")
    savefig(p_sp, joinpath(plots_dir, "step_speedup_$(nodes[1])x$(nodes[2]).png"))

    best_v_cond = match_cond(grid_data_cond, best_v)
    fai_kpc = isnothing(best_v_cond) ? nothing : kappa_pc(best_v_cond)

    if nonempty_numeric_vec(base_conds)
        sites_range_cond = 1:length(base_conds)
        base_conds_clean = clean_kappa(base_conds)

        p_conds = plot(sites_range_cond, base_conds_clean,
            label="Baseline", color=:red, lw=2, marker=:circle, yaxis=:log,
            title="Local Condition Number Profile ($nodes)",
            xlabel="MPS Site Index",
            ylabel="Condition Number (Log Scale)",
            legend=:outertopright)
        if !isnothing(fai_kpc)
            plot!(p_conds, sites_range_cond, fai_kpc, label="FAI", color=:blue, lw=2, marker=:square)
        else
            @info "No FAI κ(P^{-1}A) for $nodes; omitting FAI from condition profile"
        end
        savefig(p_conds, joinpath(plots_dir, "site_condition_$(nodes[1])x$(nodes[2]).png"))

        p_cond_imp = plot(
            title="Condition Number Improvement ($nodes)\nRatio: unpreconditioned / preconditioned",
            xlabel="MPS Site Index",
            ylabel="Improvement Factor",
            legend=:outertopright)
        hline!(p_cond_imp, [1.0], color=:red, lw=2, ls=:dash, label="No improvement")
        if !isnothing(fai_kpc)
            plot!(p_cond_imp, sites_range_cond, base_conds_clean ./ fai_kpc,
                label="FAI", color=:blue, lw=2, marker=:square)
        end
        savefig(p_cond_imp, joinpath(plots_dir, "site_cond_improvement_$(nodes[1])x$(nodes[2]).png"))
    end
end

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
