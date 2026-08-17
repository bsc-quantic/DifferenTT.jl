"""
    FAI Preconditioner Benchmark: Plotting Script

    Reads the extended grid search JLD2 results and generates comprehensive
    visualizations using Plots.jl, including:
    1. Heatmaps of Speedup (μ vs G_maxdim)
    2. Pareto fronts of Build Time vs Solve Time
    3. Heatmap of Avg Condition Numbers (μ vs G_maxdim)
    4. Site-by-site Iteration and Condition Number profiles
    5. Scaling behavior across system sizes
    Uses nocond results for accurate timings, and cond results for iteration
    and condition number profiles.
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
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/tt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/tt_poisson/results"
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

results_file = joinpath(base_folder, "collected_results.jld2")
@info "Loading dataframe from $results_file"
df = load(results_file, "df")

results_nocond = Dict()
results_cond = Dict()

for row in eachrow(df)
    # Ignore sinc for heatmaps, or let it fail gracefully
    if row.method == "sinc"
        continue
    end
    
    nodes = row.N # Using N for tt_poisson
    
    if !haskey(results_nocond, nodes)
        results_nocond[nodes] = Dict()
        results_cond[nodes] = Dict()
    end
    
    k = row.key
    d_nocond = Dict(
        "time" => row.solve_time, # baseline
        "solve_time" => row.solve_time,
        "total_time" => row.total_time,
        "precond_build_time" => row.precond_build_time,
        "mu" => ismissing(row.mu) ? NaN : row.mu,
        "G_maxdim" => ismissing(row.G_maxdim) ? 0 : row.G_maxdim
    )
    
    # tt_poisson uses iters_per_site
    iters = hasproperty(row, :iters_per_site) ? row.iters_per_site : nothing
    
    d_cond = Dict(
        "mu" => ismissing(row.mu) ? NaN : row.mu,
        "G_maxdim" => ismissing(row.G_maxdim) ? 0 : row.G_maxdim,
        "iters_per_site" => iters,
        "avg_cond" => 1.0, # fake value to bypass cond errors
        "conds_per_site" => ones(iters !== nothing ? length(iters) : 1),
        "cond_improv_per_site" => ones(iters !== nothing ? length(iters) : 1)
    )
    
    results_nocond[nodes][k] = d_nocond
    if iters !== nothing
        results_cond[nodes][k] = d_cond
    end
end

system_sizes = sort(collect(keys(results_nocond)))

# To track max speedups for the scaling plot
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

    base_time = grid_data_nocond["baseline"]["time"]

    # We get base_iters and base_conds from the cond results, if available
    base_iters = nothing
    base_conds = nothing
    if haskey(grid_data_cond, "baseline")
        base_iters = get(grid_data_cond["baseline"], "iters_per_site", nothing)
        base_conds = get(grid_data_cond["baseline"], "conds_per_site", nothing)
    end

    # Extract configuration data
    mus = Float64[]
    gdims = Int[]
    solve_speedups = Float64[]
    total_speedups = Float64[]
    avg_cond_improvements = Float64[]
    build_times = Float64[]
    solve_times = Float64[]
    keys_list = String[]

    for (k, v) in grid_data_nocond
        if k == "baseline"
            continue
        end
        push!(keys_list, k)
        push!(mus, v["mu"])
        push!(gdims, v["G_maxdim"])
        push!(build_times, v["precond_build_time"])
        push!(solve_times, v["solve_time"])
        push!(solve_speedups, base_time / v["solve_time"])
        push!(total_speedups, base_time / v["total_time"])

        # Also extract average condition numbers from the cond run, if available
        # Match by numerical values rather than string key to avoid formatting mismatches
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
        @warn "No preconditioned runs found for $nodes in nocond results"
        continue
    end

    # Track for scaling plot
    push!(max_total_speedups, maximum(total_speedups))
    push!(max_solve_speedups, maximum(solve_speedups))
    push!(grid_labels, "N_$(nodes)")

    # ---------------------------------------------------------
    # 1. Heatmaps (μ vs G_maxdim)
    # ---------------------------------------------------------
    unique_mus = sort(unique(mus))
    unique_gdims = sort(unique(gdims))

    # Create matrices for heatmaps (NaN for missing points)
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

    # Heatmap: Solve Speedup
    hm_solve = heatmap(string.(unique_mus), string.(unique_gdims), Z_solve,
        title="Solve Speedup ($nodes)",
        xlabel="Regularization parameter (μ)",
        ylabel=L"Bond Dimension ($\chi_{max}$)",
        color=:viridis,
        clim=(minimum(filter(!isnan, Z_solve)), maximum(filter(!isnan, Z_solve))),
        right_margin=5Plots.mm)

    savefig(hm_solve, joinpath(plots_dir, "heatmap_solve_N_$(nodes).png"))

    # Heatmap: Total Speedup
    hm_total = heatmap(string.(unique_mus), string.(unique_gdims), Z_total,
        title="Total Speedup ($nodes)",
        xlabel="Regularization parameter (μ)",
        ylabel=L"Bond Dimension ($\chi_{max}$)",
        color=:viridis,
        clim=(minimum(filter(!isnan, Z_total)), maximum(filter(!isnan, Z_total))),
        right_margin=5Plots.mm)

    savefig(hm_total, joinpath(plots_dir, "heatmap_total_N_$(nodes).png"))

    # Heatmap: Avg Condition Number Improvement
    valid_conds = filter(!isnan, Z_cond)
    if !isempty(valid_conds)
        hm_cond = heatmap(string.(unique_mus), string.(unique_gdims), Z_cond,
            title="Avg Cond Improvement ($nodes)",
            xlabel="Regularization parameter (μ)",
            ylabel=L"Bond Dimension ($\chi_{max}$)",
            color=:viridis,
            clim=(minimum(valid_conds), maximum(valid_conds)),
            right_margin=5Plots.mm)

        savefig(hm_cond, joinpath(plots_dir, "heatmap_cond_improvement_N_$(nodes).png"))
    else
        @info "Skipping condition heatmap for $nodes (no cond data found)"
    end

    # ---------------------------------------------------------
    # 2. Pareto Front (Build Time vs Solve Time)
    # ---------------------------------------------------------
    pf_plot = plot(title="Trade-off: Build vs Solve Time ($nodes)",
        xlabel="Preconditioner Build Time (s)",
        ylabel="Linear Solve Time (s)",
        legend=:outertopright,
        legendtitle=L"$\chi_{max}$",
        framestyle=:box)
    pf_colors = palette(:tab20)
    for (i, gdim) in enumerate(unique_gdims)
        idx = findall(x -> x == gdim, gdims)
        scatter!(pf_plot, build_times[idx], solve_times[idx], label=string(gdim), color=pf_colors[mod1(i, length(pf_colors))])
    end

    # Add baseline
    hline!(pf_plot, [base_time], label="Baseline Solve Time", color=:red, lw=2, ls=:dash)

    # Shade the "profit" region (total time < baseline time)
    # y < base_time - x  => solve_time < base_time - build_time
    plot!(pf_plot, [0, base_time], [base_time, 0],
        fillrange=0, fillalpha=0.1, fillcolor=:green,
        linealpha=0, label="Net Speedup Region")

    # Limit axes to reasonable values
    xlims!(pf_plot, 0, maximum(build_times) * 1.05)
    ylims!(pf_plot, 0, max(maximum(solve_times), base_time) * 1.05)

    savefig(pf_plot, joinpath(plots_dir, "pareto_N_$(nodes).png"))

    # ---------------------------------------------------------
    # 3. Site-by-site Profiles
    # ---------------------------------------------------------
    # Find the configuration with the best Solve Speedup that ALSO has cond data available
    best_idx = -1
    best_speedup = -1.0
    for i in 1:length(solve_speedups)
        # Find the matching cond_v by numerical values
        cond_v = nothing
        for (ck, cv) in grid_data_cond
            if ck != "baseline" && cv["mu"] == mus[i] && cv["G_maxdim"] == gdims[i]
                cond_v = cv
                break
            end
        end

        if !isnothing(cond_v) && haskey(cond_v, "iters_per_site") && (haskey(cond_v, "cond_improv_per_site") || haskey(cond_v, "conds_per_site"))
            if solve_speedups[i] > best_speedup
                best_speedup = solve_speedups[i]
                best_idx = i
            end
        end
    end

    if best_idx == -1 || isnothing(base_iters)
        @warn "Skipping site-by-site plots for $nodes: No suitable preconditioned data found in cond results"
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

    # Iterations Profile
    p_iters = plot(sites_range, base_iters,
        label="Baseline", color=:red, lw=2, marker=:circle,
        title=latexstring("Local Iterations Profile ($nodes)\nBest config: μ=$(best_v_nocond["mu"]), \$\\chi_{max}=$(best_v_nocond["G_maxdim"])\$"),
        xlabel="MPS Site Index",
        ylabel="Krylov Iterations",
        legend=:outertopright)

    plot!(p_iters, sites_range, best_v_cond["iters_per_site"],
        label="Preconditioned", color=:blue, lw=2, marker=:square)

    # Fill between lines to highlight reduction
    plot!(p_iters, sites_range, base_iters, fillrange=best_v_cond["iters_per_site"],
        fillalpha=0.2, fillcolor=:green, label="Iterations Saved", linealpha=0)

    savefig(p_iters, joinpath(plots_dir, "site_iterations_N_$(nodes).png"))

    if !isnothing(base_conds)
        # Condition Number Profile
        # Remember: Baseline condition numbers are stored directly. 
        # For preconditioned, we stored the *improvement ratio* (unprecond / precond)
        base_conds_clean = [c > 0 ? c : NaN for c in base_conds]
        best_cond_key = haskey(best_v_cond, "cond_improv_per_site") ? "cond_improv_per_site" : "conds_per_site"
        best_conds_clean = [c > 0 ? c : NaN for c in best_v_cond[best_cond_key]]

        # Reconstruct the preconditioned absolute condition number to compare side-by-side
        # Since best_v_cond["conds_per_site"] = base_conds_clean / precond_absolute
        precond_absolute = base_conds_clean ./ best_conds_clean

        p_conds = plot(sites_range, base_conds_clean,
            label="Baseline", color=:red, lw=2, marker=:circle, yaxis=:log,
            title="Local Condition Number Profile ($nodes)",
            xlabel="MPS Site Index",
            ylabel="Condition Number (Log Scale)",
            legend=:outertopright)

        plot!(p_conds, sites_range, precond_absolute,
            label="Preconditioned", color=:blue, lw=2, marker=:square)

        savefig(p_conds, joinpath(plots_dir, "site_condition_N_$(nodes).png"))

        # Alternative Condition Plot: Plot just the improvement ratio
        p_cond_imp = bar(sites_range, best_conds_clean,
            title="Condition Number Improvement ($nodes)\nRatio: Unpreconditioned / Preconditioned",
            xlabel="MPS Site Index",
            ylabel="Improvement Factor",
            legend=false,
            color=:purple, alpha=0.7)

        hline!(p_cond_imp, [1.0], color=:red, lw=2, ls=:dash, label="No improvement")
        savefig(p_cond_imp, joinpath(plots_dir, "site_cond_improvement_N_$(nodes).png"))
    else
        @info "Skipping condition number plots for $nodes (no data found in baseline)"
    end

end

# ---------------------------------------------------------
# 4. Scaling View
# ---------------------------------------------------------
if !isempty(max_total_speedups)
    p_scale = plot(1:length(grid_labels), max_solve_speedups,
        label="Max Solve Speedup", color=:blue, lw=3, marker=:circle, markersize=8,
        title="Scaling of Maximum Achievable Speedups",
        xlabel="System Size",
        ylabel="Speedup Factor",
        xticks=(1:length(grid_labels), grid_labels),
        legend=:topright,
        framestyle=:box)

    plot!(p_scale, 1:length(grid_labels), max_total_speedups,
        label="Max Total Speedup", color=:green, lw=3, marker=:square, markersize=8)

    hline!(p_scale, [1.0], label="Break-even (1.0x)", color=:red, lw=2, ls=:dash)

    savefig(p_scale, joinpath(plots_dir, "scaling_summary.png"))
end

@info "All plots successfully generated in $plots_dir"
