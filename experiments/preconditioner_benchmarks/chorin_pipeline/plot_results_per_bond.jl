"""
    FAI Preconditioner Benchmark: Evolution Plotting Script per Bond Dimension

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
cond_folder = joinpath(base_folder, "results")
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

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
    base_conds_clean = nonempty_numeric_vec(base_conds) ? clean_kappa(base_conds) : nothing

    mus = Float64[]
    gdims = Int[]
    solve_speedups = Float64[]
    keys_list = String[]

    for (k, v) in grid_data_nocond
        k == "baseline" && continue
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
    sites_range_cond = !isnothing(base_conds_clean) ? (1:length(base_conds_clean)) : []

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
    p_cond_imp_combined = plot(
        title=latexstring("Condition Number Improvement ($nodes)\nBest μ per \$\\chi_{max}\$"),
        xlabel="MPS Site Index", ylabel="Improvement Factor", legend=:outertopright
    )

    plot!(p_iters_combined, steps_range, base_iters, label="Baseline", color=:black, lw=3, marker=:circle)
    hline!(p_sp_combined, [1.0], color=:black, lw=3, ls=:dash, label="Break-even (1.0x)")
    if !isnothing(base_conds_clean)
        plot!(p_conds_combined, sites_range_cond, base_conds_clean, label="Baseline", color=:black, lw=3, marker=:circle)
        hline!(p_cond_imp_combined, [1.0], color=:black, lw=3, ls=:dash, label="No improvement")
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
        best_idx == -1 && continue

        best_v = grid_data_nocond[keys_list[best_idx]]
        best_v_cond = match_cond(grid_data_cond, best_v)
        fai_kpc = isnothing(best_v_cond) ? nothing : kappa_pc(best_v_cond)

        out_dir = joinpath(plots_dir, "$(nodes[1])x$(nodes[2])", "chi_$gdim")
        mkpath(out_dir)

        p_iters = plot(steps_range, base_iters,
            label="Baseline", color=:red, lw=2, marker=:circle,
            title=latexstring("Krylov Iterations vs Time Step ($nodes)\nBest FAI for \$\\chi_{max}=$(gdim)\$: μ=$(best_v["mu"])"),
            xlabel="Chorin Time Step",
            ylabel="Avg Krylov Iterations",
            legend=:outertopright)
        plot!(p_iters, steps_range, best_v["iters"],
            label="FAI", color=:blue, lw=2, marker=:square)
        plot!(p_iters, steps_range, base_iters, fillrange=best_v["iters"],
            fillalpha=0.12, fillcolor=:blue, label="", linealpha=0)
        savefig(p_iters, joinpath(out_dir, "step_iterations.png"))

        per_step_speedup = base_solve_times ./ best_v["solve_times"]
        p_sp = plot(steps_range, per_step_speedup,
            label="FAI", color=:purple, lw=2, marker=:diamond,
            title=latexstring("Solve Speedup vs Time Step ($nodes)\nBest FAI for \$\\chi_{max}=$(gdim)\$: μ=$(best_v["mu"])"),
            xlabel="Chorin Time Step",
            ylabel="Speedup Ratio",
            legend=:outertopright)
        hline!(p_sp, [1.0], color=:red, lw=2, ls=:dash, label="Break-even (1.0x)")
        savefig(p_sp, joinpath(out_dir, "step_speedup.png"))

        c_color = colors[mod1(i, length(colors))]
        plot!(p_iters_combined, steps_range, best_v["iters"],
            label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ (μ=$(best_v["mu"]))"), color=c_color, lw=2, marker=:square)
        plot!(p_sp_combined, steps_range, per_step_speedup,
            label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ (μ=$(best_v["mu"]))"), color=c_color, lw=2, marker=:diamond)

        if !isnothing(base_conds_clean)
            p_conds = plot(sites_range_cond, base_conds_clean,
                label="Baseline", color=:red, lw=2, marker=:circle, yaxis=:log,
                title="Local Condition Number Profile ($nodes)",
                xlabel="MPS Site Index",
                ylabel="Condition Number (Log Scale)",
                legend=:outertopright)
            p_cond_imp = plot(
                title="Condition Number Improvement ($nodes)\nRatio: unpreconditioned / preconditioned",
                xlabel="MPS Site Index",
                ylabel="Improvement Factor",
                legend=:outertopright)
            hline!(p_cond_imp, [1.0], color=:red, lw=2, ls=:dash, label="No improvement")

            if !isnothing(fai_kpc)
                plot!(p_conds, sites_range_cond, fai_kpc, label="FAI", color=:blue, lw=2, marker=:square)
                plot!(p_cond_imp, sites_range_cond, base_conds_clean ./ fai_kpc,
                    label="FAI", color=:blue, lw=2, marker=:square)
                plot!(p_conds_combined, sites_range_cond, fai_kpc,
                    label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ (μ=$(best_v["mu"]))"), color=c_color, lw=2, marker=:square)
                plot!(p_cond_imp_combined, sites_range_cond, base_conds_clean ./ fai_kpc,
                    label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ (μ=$(best_v["mu"]))"), color=c_color, lw=2, marker=:square)
            else
                @info "No FAI κ(P^{-1}A) for $nodes χ=$gdim; omitting FAI from condition profile"
            end

            savefig(p_conds, joinpath(out_dir, "site_condition.png"))
            savefig(p_cond_imp, joinpath(out_dir, "site_cond_improvement.png"))
        end
    end

    out_dir_combined = joinpath(plots_dir, "$(nodes[1])x$(nodes[2])")
    mkpath(out_dir_combined)
    savefig(p_iters_combined, joinpath(out_dir_combined, "combined_step_iterations.png"))
    savefig(p_sp_combined, joinpath(out_dir_combined, "combined_step_speedup.png"))
    if !isnothing(base_conds_clean)
        savefig(p_conds_combined, joinpath(out_dir_combined, "combined_site_condition.png"))
        savefig(p_cond_imp_combined, joinpath(out_dir_combined, "combined_site_cond_improvement.png"))
    end
end

@info "All per-bond evolution plots successfully generated in $plots_dir"
