"""
    FAI Preconditioner Benchmark: Evolution Plotting Script

    Reads the evolution distributed JLD2 results and generates comprehensive
    visualizations using Plots.jl, including:
    1. Heatmaps of Aggregate Solve/Total Speedup (μ vs G_maxdim)
    2. Pareto fronts of Build Time vs Total Solve Time
    3. Step-by-step Iteration profiles
    4. Step-by-step Speedup profiles
    5. Scaling behavior across system sizes

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
        "mu" => ismissing(row.mu) ? NaN : row.mu,
        "G_maxdim" => ismissing(row.G_maxdim) ? 0 : row.G_maxdim,
        "precond_build_time" => row.precond_build_time,
        "total_solve_time" => row.total_solve_time,
        "solve_times" => row.solve_times,
        "iters" => row.global_iters,
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

    if !haskey(grid_data, "baseline")
        @warn "Skipping $nodes: Missing baseline data"
        continue
    end

    base = grid_data["baseline"]
    base_time = base["total_solve_time"]
    base_iters = base["iters"]
    base_solve_times = base["solve_times"]
    n_steps = base["n_steps"]

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
        
        sp_solve = base_time / v["total_solve_time"]
        sp_total = base_time / (v["total_solve_time"] + v["precond_build_time"])
        
        push!(solve_speedups, sp_solve)
        push!(total_speedups, sp_total)
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

    for i in 1:length(mus)
        μ_idx = findfirst(==(mus[i]), unique_mus)
        g_idx = findfirst(==(gdims[i]), unique_gdims)
        Z_solve[g_idx, μ_idx] = solve_speedups[i]
        Z_total[g_idx, μ_idx] = total_speedups[i]
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
    best_v = grid_data[best_key]

    steps_range = 1:n_steps

    # Iterations over time
    p_iters = plot(steps_range, base_iters,
        label="Baseline", color=:red, lw=2, marker=:circle,
        title=latexstring("Krylov Iterations vs Time Step ($nodes)\nBest FAI: μ=$(best_v["mu"]), \$\\chi_{max}=$(best_v["G_maxdim"])\$"),
        xlabel="Chorin Time Step",
        ylabel="Avg Krylov Iterations per Site",
        legend=:outertopright)

    plot!(p_iters, steps_range, best_v["iters"],
        label="FAI", color=:blue, lw=2, marker=:square)

    plot!(p_iters, steps_range, base_iters, fillrange=best_v["iters"],
        fillalpha=0.2, fillcolor=:green, label="Iterations Saved", linealpha=0)

    savefig(p_iters, joinpath(plots_dir, "step_iterations_$(nodes[1])x$(nodes[2]).png"))

    # Speedup over time
    per_step_speedup = base_solve_times ./ best_v["solve_times"]
    p_sp = plot(steps_range, per_step_speedup,
        label="Per-Step Solve Speedup", color=:purple, lw=2, marker=:diamond,
        title=latexstring("Solve Speedup vs Time Step ($nodes)\nBest FAI: μ=$(best_v["mu"]), \$\\chi_{max}=$(best_v["G_maxdim"])\$"),
        xlabel="Chorin Time Step",
        ylabel="Speedup Ratio",
        legend=:outertopright)

    hline!(p_sp, [1.0], color=:red, lw=2, ls=:dash, label="Break-even (1.0x)")
    
    savefig(p_sp, joinpath(plots_dir, "step_speedup_$(nodes[1])x$(nodes[2]).png"))

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
