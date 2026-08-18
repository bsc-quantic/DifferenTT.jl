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
base_folder = "experiments/preconditioner_benchmarks/chorin_pipeline"
nocond_folder = joinpath(base_folder, "results_nocond")
cond_folder = joinpath(base_folder, "results")
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

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

for nodes in system_sizes
    @info "Generating per-bond-dimension plots for system size $nodes"

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
    keys_list = String[]

    for (k, v) in grid_data_nocond
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
    p_conds_combined = plot(
        yaxis=:log, title=latexstring("Local Condition Number Profile ($nodes)\nBest μ per \$\\chi_{max}\$"),
        xlabel="MPS Site Index", ylabel="Condition Number (Log Scale)", legend=:outertopright
    )

    plot!(p_iters_combined, steps_range, base_iters, label="Baseline", color=:black, lw=3, marker=:circle)
    hline!(p_sp_combined, [1.0], color=:black, lw=3, ls=:dash, label="Break-even (1.0x)")

    sites_range_cond = !isnothing(base_conds) ? (1:length(base_conds)) : []
    if !isnothing(base_conds)
        base_conds_clean = [c > 0 ? c : NaN for c in base_conds]
        plot!(p_conds_combined, sites_range_cond, base_conds_clean, label="Baseline", color=:black, lw=3, marker=:circle)
    end

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
        best_v = grid_data_nocond[best_key]

        best_v_cond = nothing
        for (ck, cv) in grid_data_cond
            if ck != "baseline" && cv["mu"] == best_v["mu"] && cv["G_maxdim"] == best_v["G_maxdim"]
                best_v_cond = cv
                break
            end
        end

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

        if !isnothing(base_conds) && !isnothing(best_v_cond)
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

            savefig(p_conds, joinpath(out_dir, "site_condition.png"))

            plot!(p_conds_combined, sites_range_cond, precond_absolute,
                label=latexstring("\$\\chi_{max}=$(gdim)\$ (μ=$(best_v["mu"]))"), color=c_color, lw=2, marker=:square)
        end
    end

    out_dir_combined = joinpath(plots_dir, "$(nodes[1])x$(nodes[2])")
    mkpath(out_dir_combined)

    savefig(p_iters_combined, joinpath(out_dir_combined, "combined_step_iterations.png"))
    savefig(p_sp_combined, joinpath(out_dir_combined, "combined_step_speedup.png"))
    if !isnothing(base_conds)
        savefig(p_conds_combined, joinpath(out_dir_combined, "combined_site_condition.png"))
    end
end

@info "All per-bond evolution plots successfully generated in $plots_dir"
