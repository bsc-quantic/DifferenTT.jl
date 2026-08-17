"""
    FAI Preconditioner Benchmark: Evolution Plotting Script per Bond Dimension
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
use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/chorin_pipeline/results_nocond" : "experiments/preconditioner_benchmarks/chorin_pipeline/results"
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
        "iters" => [mean(step_iters) for step_iters in row.iters],
        "n_steps" => length(row.solve_times)
    )
    results[nodes][k] = d
end

system_sizes = sort(collect(keys(results)))

for nodes in system_sizes
    @info "Generating per-bond-dimension plots for system size $nodes"

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
    keys_list = String[]

    for (k, v) in grid_data
        if k == "baseline"
            continue
        end
        push!(keys_list, k)
        push!(mus, v["mu"])
        push!(gdims, v["G_maxdim"])
        push!(solve_speedups, base_time / v["total_solve_time"])
    end

    if isempty(mus)
        @warn "No preconditioned runs found for $nodes"
        continue
    end

    unique_gdims = sort(unique(gdims))
    steps_range = 1:n_steps

    # Initialize combined plots
    p_iters_combined = plot(
        title=latexstring("Krylov Iterations vs Time Step ($nodes)\nBest μ per \$\\chi_{max}\$"),
        xlabel="Chorin Time Step", ylabel="Avg Krylov Iterations", legend=:outertopright
    )
    p_sp_combined = plot(
        title=latexstring("Solve Speedup vs Time Step ($nodes)\nBest μ per \$\\chi_{max}\$"),
        xlabel="Chorin Time Step", ylabel="Speedup Ratio", legend=:outertopright
    )

    plot!(p_iters_combined, steps_range, base_iters, label="Baseline", color=:black, lw=3, marker=:circle)
    hline!(p_sp_combined, [1.0], color=:black, lw=3, ls=:dash, label="Break-even (1.0x)")

    colors = palette(:tab20)

    for (i, gdim) in enumerate(unique_gdims)
        best_idx = -1
        best_speedup = -1.0

        for idx in 1:length(solve_speedups)
            if gdims[idx] == gdim && solve_speedups[idx] > best_speedup
                best_speedup = solve_speedups[idx]
                best_idx = idx
            end
        end

        if best_idx == -1
            continue
        end

        best_key = keys_list[best_idx]
        best_v = grid_data[best_key]

        out_dir = joinpath(plots_dir, "$(nodes[1])x$(nodes[2])", "chi_$gdim")
        mkpath(out_dir)

        # Iterations Profile
        p_iters = plot(steps_range, base_iters,
            label="Baseline", color=:red, lw=2, marker=:circle,
            title=latexstring("Krylov Iterations vs Time Step ($nodes)\nBest config for \$\\chi_{max}=$(gdim)\$: μ=$(best_v["mu"])"),
            xlabel="Chorin Time Step",
            ylabel="Avg Krylov Iterations",
            legend=:outertopright)

        plot!(p_iters, steps_range, best_v["iters"],
            label="Preconditioned", color=:blue, lw=2, marker=:square)

        plot!(p_iters, steps_range, base_iters, fillrange=best_v["iters"],
            fillalpha=0.2, fillcolor=:green, label="Iterations Saved", linealpha=0)

        savefig(p_iters, joinpath(out_dir, "step_iterations.png"))

        # Speedup Profile
        per_step_speedup = base_solve_times ./ best_v["solve_times"]
        p_sp = plot(steps_range, per_step_speedup,
            label="Preconditioned", color=:purple, lw=2, marker=:diamond,
            title=latexstring("Solve Speedup vs Time Step ($nodes)\nBest config for \$\\chi_{max}=$(gdim)\$: μ=$(best_v["mu"])"),
            xlabel="Chorin Time Step",
            ylabel="Speedup Ratio",
            legend=:outertopright)
        
        hline!(p_sp, [1.0], color=:red, lw=2, ls=:dash, label="Break-even (1.0x)")

        savefig(p_sp, joinpath(out_dir, "step_speedup.png"))

        # Add to combined plots
        c_color = colors[mod1(i, length(colors))]
        plot!(p_iters_combined, steps_range, best_v["iters"],
            label=latexstring("\$\\chi_{max}=$(gdim)\$ (μ=$(best_v["mu"]))"), color=c_color, lw=2, marker=:square)

        plot!(p_sp_combined, steps_range, per_step_speedup,
            label=latexstring("\$\\chi_{max}=$(gdim)\$ (μ=$(best_v["mu"]))"), color=c_color, lw=2, marker=:diamond)
    end

    out_dir_combined = joinpath(plots_dir, "$(nodes[1])x$(nodes[2])")
    mkpath(out_dir_combined)

    savefig(p_iters_combined, joinpath(out_dir_combined, "combined_step_iterations.png"))
    savefig(p_sp_combined, joinpath(out_dir_combined, "combined_step_speedup.png"))
end

@info "All per-bond evolution plots successfully generated in $plots_dir"
