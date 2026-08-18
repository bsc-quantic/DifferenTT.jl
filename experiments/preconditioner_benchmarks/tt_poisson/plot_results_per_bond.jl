"""
    FAI Preconditioner Benchmark: Plotting Script per Bond Dimension

    Reads the extended grid search JLD2 results and generates comprehensive
    visualizations using Plots.jl, focusing on site-by-site Iteration and
    Condition Number profiles for the best μ at EACH bond dimension.
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
base_folder = "experiments/preconditioner_benchmarks/tt_poisson"
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
    if row.method == "sinc"
        continue
    end
    nodes = row.N
    if !haskey(results_nocond, nodes)
        results_nocond[nodes] = Dict()
    end
    k = row.key
    d_nocond = Dict(
        "time" => row.solve_time,
        "solve_time" => row.solve_time,
        "total_time" => row.total_time,
        "precond_build_time" => row.precond_build_time,
        "mu" => ismissing(row.mu) ? NaN : row.mu,
        "G_maxdim" => ismissing(row.G_maxdim) ? 0 : row.G_maxdim,
        "iters_per_site" => hasproperty(row, :iters_per_site) ? row.iters_per_site : nothing
    )
    results_nocond[nodes][k] = d_nocond
end

results_cond = Dict()
if !isempty(df_cond)
    for row in eachrow(df_cond)
        if row.method == "sinc"
            continue
        end
        nodes = row.N
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

    base_time = grid_data_nocond["baseline"]["time"]

    base_iters = nothing
    if haskey(grid_data_nocond, "baseline")
        base_iters = get(grid_data_nocond["baseline"], "iters_per_site", nothing)
    end

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
        push!(solve_speedups, base_time / v["solve_time"])
    end

    if isempty(mus)
        @warn "No preconditioned runs found for $nodes in nocond results"
        continue
    end

    unique_gdims = sort(unique(gdims))

    # Initialize combined plots for this system size
    sites_range_combined = !isnothing(base_iters) ? (1:length(base_iters)) : []
    p_iters_combined = plot(
        title=latexstring("Local Iterations Profile ($nodes)\nBest μ per \$\\chi_{max}\$"),
        xlabel="MPS Site Index", ylabel="Krylov Iterations", legend=:outertopright
    )
    p_conds_combined = plot(
        yaxis=:log, title=latexstring("Local Condition Number Profile ($nodes)\nBest μ per \$\\chi_{max}\$"),
        xlabel="MPS Site Index", ylabel="Condition Number (Log Scale)", legend=:outertopright
    )
    p_cond_imp_combined = plot(
        title=latexstring("Condition Number Improvement ($nodes)\nBest μ per \$\\chi_{max}\$"),
        xlabel="MPS Site Index", ylabel="Improvement Factor", legend=:outertopright
    )

    if !isnothing(base_iters)
        plot!(p_iters_combined, sites_range_combined, base_iters, label="Baseline", color=:black, lw=3, marker=:circle)
    end

    if !isnothing(base_conds)
        base_conds_clean = [c > 0 ? c : NaN for c in base_conds]
        plot!(p_conds_combined, sites_range_combined, base_conds_clean, label="Baseline", color=:black, lw=3, marker=:circle)
        hline!(p_cond_imp_combined, [1.0], color=:black, lw=3, ls=:dash, label="No improvement")
    end

    colors = palette(:tab20) # Distinct colors for each bond dimension

    for (i, gdim) in enumerate(unique_gdims)
        # Find the configuration with the best Solve Speedup for this gdim
        best_idx = -1
        best_speedup = -1.0

        for i in 1:length(solve_speedups)
            if gdims[i] != gdim
                continue
            end

            if solve_speedups[i] > best_speedup
                best_speedup = solve_speedups[i]
                best_idx = i
            end
        end

        if best_idx == -1 || isnothing(base_iters)
            @warn "Skipping site-by-site plots for $nodes and gdim=$gdim: No suitable preconditioned data found in nocond results"
            continue
        end

        best_key = keys_list[best_idx]
        best_v_nocond = grid_data_nocond[best_key]

        # Find the corresponding cond_v
        best_v_cond = nothing
        for (ck, cv) in grid_data_cond
            if ck != "baseline" && cv["mu"] == best_v_nocond["mu"] && cv["G_maxdim"] == best_v_nocond["G_maxdim"]
                best_v_cond = cv
                break
            end
        end

        sites_range = 1:length(base_iters)

        # Create output directory: plots / 8x8 / chi_10
        out_dir = joinpath(plots_dir, "N_$(nodes)", "chi_$gdim")
        mkpath(out_dir)

        # Iterations Profile
        p_iters = plot(sites_range, base_iters,
            label="Baseline", color=:red, lw=2, marker=:circle,
            title=latexstring("Local Iterations Profile ($nodes)\nBest config for \$\\chi_{max}=$(gdim)\$: μ=$(best_v_nocond["mu"])"),
            xlabel="MPS Site Index",
            ylabel="Krylov Iterations",
            legend=:outertopright)

        plot!(p_iters, sites_range, best_v_nocond["iters_per_site"],
            label="Preconditioned", color=:blue, lw=2, marker=:square)

        plot!(p_iters, sites_range, base_iters, fillrange=best_v_nocond["iters_per_site"],
            fillalpha=0.2, fillcolor=:green, label="Iterations Saved", linealpha=0)

        savefig(p_iters, joinpath(out_dir, "site_iterations.png"))

        # Add to combined plot
        c_color = colors[mod1(i, length(colors))]
        plot!(p_iters_combined, sites_range, best_v_nocond["iters_per_site"],
            label=latexstring("\$\\chi_{max}=$(gdim)\$ (μ=$(best_v_nocond["mu"]))"), color=c_color, lw=2, marker=:square)

        if !isnothing(base_conds) && !isnothing(best_v_cond)
            base_conds_clean = [c > 0 ? c : NaN for c in base_conds]
            best_cond_key = haskey(best_v_cond, "cond_improv_per_site") && !isempty(best_v_cond["cond_improv_per_site"]) ? "cond_improv_per_site" : "conds_per_site"
            best_conds_clean = [c > 0 ? c : NaN for c in best_v_cond[best_cond_key]]

            precond_absolute = base_conds_clean ./ best_conds_clean

            p_conds = plot(sites_range, base_conds_clean,
                label="Baseline", color=:red, lw=2, marker=:circle, yaxis=:log,
                title="Local Condition Number Profile ($nodes)",
                xlabel="MPS Site Index",
                ylabel="Condition Number (Log Scale)",
                legend=:outertopright)

            plot!(p_conds, sites_range, precond_absolute,
                label="Preconditioned", color=:blue, lw=2, marker=:square)

            savefig(p_conds, joinpath(out_dir, "site_condition.png"))

            p_cond_imp = bar(sites_range, best_conds_clean,
                title="Condition Number Improvement ($nodes)\nRatio: Unpreconditioned / Preconditioned",
                xlabel="MPS Site Index",
                ylabel="Improvement Factor",
                legend=false,
                color=:purple, alpha=0.7)

            hline!(p_cond_imp, [1.0], color=:red, lw=2, ls=:dash, label="No improvement")
            savefig(p_cond_imp, joinpath(out_dir, "site_cond_improvement.png"))

            # Add to combined condition and improvement plots
            plot!(p_conds_combined, sites_range, precond_absolute,
                label=latexstring("\$\\chi_{max}=$(gdim)\$ (μ=$(best_v_nocond["mu"]))"), color=c_color, lw=2, marker=:square)

            plot!(p_cond_imp_combined, sites_range, best_conds_clean,
                label=latexstring("\$\\chi_{max}=$(gdim)\$ (μ=$(best_v_nocond["mu"]))"), color=c_color, lw=2, marker=:square)
        end
    end

    # Save the combined plots in the system size root folder
    out_dir_combined = joinpath(plots_dir, "N_$(nodes)")
    mkpath(out_dir_combined)

    if !isnothing(base_iters)
        savefig(p_iters_combined, joinpath(out_dir_combined, "combined_site_iterations.png"))
    end
    if !isnothing(base_conds)
        savefig(p_conds_combined, joinpath(out_dir_combined, "combined_site_condition.png"))
        savefig(p_cond_imp_combined, joinpath(out_dir_combined, "combined_site_cond_improvement.png"))
    end
end

@info "All per-bond plots successfully generated in $plots_dir"
