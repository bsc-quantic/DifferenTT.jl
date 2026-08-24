"""
    FAI / Sinc Preconditioner Benchmark: Plotting Script

    Reads the extended grid search JLD2 results and generates comprehensive
    visualizations using Plots.jl, including:
    1. Heatmaps of Speedup (μ vs G_maxdim) for FAI
    2. Pareto fronts of Build Time vs Solve Time (FAI + sinc)
    3. Heatmap of Avg Condition Number Improvement (μ vs G_maxdim) for FAI
    4. Site-by-site Iteration and Condition Number profiles (baseline / FAI / sinc)
    5. Scaling behavior across system sizes
    Uses nocond results for accurate timings, and cond results for iteration
    and condition number profiles.

    Note: `pc_conds_per_site` is the absolute local κ(P⁻¹A), not a ratio.
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
    nodes = row.N
    if !haskey(results_nocond, nodes)
        results_nocond[nodes] = Dict()
    end
    k = row.key
    d_nocond = Dict(
        "key" => k,
        "method" => row.method,
        "time" => row.solve_time,
        "solve_time" => row.solve_time,
        "total_time" => row.total_time,
        "precond_build_time" => row.precond_build_time,
        "mu" => hasproperty(row, :mu) && !ismissing(row.mu) ? row.mu : NaN,
        "G_maxdim" => hasproperty(row, :G_maxdim) && !ismissing(row.G_maxdim) ? row.G_maxdim : 0,
        "M" => hasproperty(row, :M) && !ismissing(row.M) ? Int(row.M) : 0,
        "iters_per_site" => hasproperty(row, :iters_per_site) ? row.iters_per_site : nothing
    )
    results_nocond[nodes][k] = d_nocond
end

results_cond = Dict()
if !isempty(df_cond)
    for row in eachrow(df_cond)
        nodes = row.N
        if !haskey(results_cond, nodes)
            results_cond[nodes] = Dict()
        end
        k = row.key
        d_cond = Dict(
            "key" => k,
            "method" => row.method,
            "mu" => hasproperty(row, :mu) && !ismissing(row.mu) ? row.mu : NaN,
            "G_maxdim" => hasproperty(row, :G_maxdim) && !ismissing(row.G_maxdim) ? row.G_maxdim : 0,
            "M" => hasproperty(row, :M) && !ismissing(row.M) ? Int(row.M) : 0,
            "avg_cond" => hasproperty(row, :avg_cond) ? row.avg_cond : NaN,
            "avg_pc_cond" => hasproperty(row, :avg_pc_cond) ? row.avg_pc_cond : NaN,
            "conds_per_site" => row_vec(row, :conds_per_site),
            "pc_conds_per_site" => row_vec(row, :pc_conds_per_site)
        )
        results_cond[nodes][k] = d_cond
    end
end

system_sizes = sort(collect(keys(results_nocond)))

# To track max speedups for the scaling plot
max_total_speedups_fai = Float64[]
max_solve_speedups_fai = Float64[]
max_total_speedups_sinc = Float64[]
max_solve_speedups_sinc = Float64[]
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

    base_iters = nothing
    if haskey(grid_data_nocond, "baseline")
        base_iters = get(grid_data_nocond["baseline"], "iters_per_site", nothing)
    end

    base_conds = nothing
    if haskey(grid_data_cond, "baseline")
        base_conds = get(grid_data_cond["baseline"], "conds_per_site", nothing)
    end

    # FAI grid-search arrays
    mus = Float64[]
    gdims = Int[]
    fai_solve_speedups = Float64[]
    fai_total_speedups = Float64[]
    cond_improvement_factors = Float64[]
    fai_build_times = Float64[]
    fai_solve_times = Float64[]
    fai_keys = String[]

    sinc_Ms = Int[]
    sinc_solve_speedups = Float64[]
    sinc_total_speedups = Float64[]
    sinc_build_times = Float64[]
    sinc_solve_times = Float64[]
    sinc_keys = String[]

    for (k, v) in grid_data_nocond
        v["method"] == "baseline" && continue
        if v["method"] == "sinc"
            push!(sinc_keys, k)
            push!(sinc_Ms, v["M"])
            push!(sinc_build_times, v["precond_build_time"])
            push!(sinc_solve_times, v["solve_time"])
            push!(sinc_solve_speedups, base_time / v["solve_time"])
            push!(sinc_total_speedups, base_time / v["total_time"])
        elseif v["method"] == "fai"
            push!(fai_keys, k)
            push!(mus, v["mu"])
            push!(gdims, v["G_maxdim"])
            push!(fai_build_times, v["precond_build_time"])
            push!(fai_solve_times, v["solve_time"])
            push!(fai_solve_speedups, base_time / v["solve_time"])
            push!(fai_total_speedups, base_time / v["total_time"])

            cond_v = match_cond(grid_data_cond, v)
            push!(cond_improvement_factors, isnothing(cond_v) ? NaN : cond_improvement_factor(cond_v))
        end
    end

    if isempty(mus) && isempty(sinc_Ms)
        @warn "No preconditioned runs found for $nodes in nocond results"
        continue
    end

    # Track for scaling plot
    push!(grid_labels, "N_$(nodes)")
    push!(max_solve_speedups_fai, isempty(fai_solve_speedups) ? NaN : maximum(fai_solve_speedups))
    push!(max_total_speedups_fai, isempty(fai_total_speedups) ? NaN : maximum(fai_total_speedups))
    push!(max_solve_speedups_sinc, isempty(sinc_solve_speedups) ? NaN : maximum(sinc_solve_speedups))
    push!(max_total_speedups_sinc, isempty(sinc_total_speedups) ? NaN : maximum(sinc_total_speedups))

    unique_mus = sort(unique(mus))
    unique_gdims = sort(unique(gdims))

    # ---------------------------------------------------------
    # 1. Heatmaps (μ vs G_maxdim) — FAI only
    # ---------------------------------------------------------
    if !isempty(mus)
        Z_solve = fill(NaN, length(unique_gdims), length(unique_mus))
        Z_total = fill(NaN, length(unique_gdims), length(unique_mus))
        Z_cond = fill(NaN, length(unique_gdims), length(unique_mus))

        for i in 1:length(mus)
            μ_idx = findfirst(==(mus[i]), unique_mus)
            g_idx = findfirst(==(gdims[i]), unique_gdims)
            Z_solve[g_idx, μ_idx] = fai_solve_speedups[i]
            Z_total[g_idx, μ_idx] = fai_total_speedups[i]
            Z_cond[g_idx, μ_idx] = cond_improvement_factors[i]
        end

        hm_solve = heatmap(string.(unique_mus), string.(unique_gdims), Z_solve,
            title="Solve Speedup ($nodes)",
            xlabel="Regularization parameter (μ)",
            ylabel=L"Bond Dimension ($\chi_{max}$)",
            color=:viridis,
            clim=(minimum(filter(!isnan, Z_solve)), maximum(filter(!isnan, Z_solve))),
            right_margin=5Plots.mm)

        savefig(hm_solve, joinpath(plots_dir, "heatmap_solve_N_$(nodes).png"))

        hm_total = heatmap(string.(unique_mus), string.(unique_gdims), Z_total,
            title="Total Speedup ($nodes)",
            xlabel="Regularization parameter (μ)",
            ylabel=L"Bond Dimension ($\chi_{max}$)",
            color=:viridis,
            clim=(minimum(filter(!isnan, Z_total)), maximum(filter(!isnan, Z_total))),
            right_margin=5Plots.mm)

        savefig(hm_total, joinpath(plots_dir, "heatmap_total_N_$(nodes).png"))

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
            @info "Skipping condition heatmap for $nodes (no FAI κ(P⁻¹A) data found)"
        end
    else
        @info "Skipping FAI heatmaps for $nodes (no FAI runs)"
    end

    # ---------------------------------------------------------
    # 2. Pareto Front (Build Time vs Solve Time)
    # ---------------------------------------------------------
    all_build = vcat(fai_build_times, sinc_build_times)
    all_solve = vcat(fai_solve_times, sinc_solve_times)
    pf_plot = plot(title="Trade-off: Build vs Solve Time ($nodes)",
        xlabel="Preconditioner Build Time (s)",
        ylabel="Linear Solve Time (s)",
        legend=:outertopright,
        framestyle=:box)
    pf_colors = palette(:tab20)
    for (i, gdim) in enumerate(unique_gdims)
        idx = findall(x -> x == gdim, gdims)
        scatter!(pf_plot, fai_build_times[idx], fai_solve_times[idx],
            label=latexstring("FAI \$\\chi_{max}=$(gdim)\$"),
            color=pf_colors[mod1(i, length(pf_colors))], marker=:circle)
    end
    sinc_colors = palette(:Dark2)
    for (i, M) in enumerate(sort(unique(sinc_Ms)))
        idx = findall(x -> x == M, sinc_Ms)
        scatter!(pf_plot, sinc_build_times[idx], sinc_solve_times[idx],
            label="Sinc M=$(M)",
            color=sinc_colors[mod1(i, length(sinc_colors))], marker=:diamond, markersize=8)
    end

    hline!(pf_plot, [base_time], label="Baseline Solve Time", color=:red, lw=2, ls=:dash)

    plot!(pf_plot, [0, base_time], [base_time, 0],
        fillrange=0, fillalpha=0.1, fillcolor=:green,
        linealpha=0, label="Net Speedup Region")

    if !isempty(all_build)
        xlims!(pf_plot, 0, maximum(all_build) * 1.05)
        ylims!(pf_plot, 0, max(maximum(all_solve), base_time) * 1.05)
    end

    savefig(pf_plot, joinpath(plots_dir, "pareto_N_$(nodes).png"))

    # ---------------------------------------------------------
    # 3. Site-by-site Profiles
    # ---------------------------------------------------------
    if isnothing(base_iters)
        @warn "Skipping site-by-site plots for $nodes: no baseline iters_per_site"
        continue
    end

    best_fai = nothing
    if !isempty(fai_solve_speedups)
        best_fai = grid_data_nocond[fai_keys[argmax(fai_solve_speedups)]]
    end
    best_sinc = nothing
    if !isempty(sinc_solve_speedups)
        best_sinc = grid_data_nocond[sinc_keys[argmax(sinc_solve_speedups)]]
    end

    title_bits = String[]
    if !isnothing(best_fai)
        push!(title_bits, "FAI μ=$(best_fai["mu"]), \$\\chi_{max}=$(best_fai["G_maxdim"])\$")
    end
    if !isnothing(best_sinc)
        push!(title_bits, "Sinc M=$(best_sinc["M"])")
    end
    iters_title = isempty(title_bits) ? "Local Iterations Profile ($nodes)" :
        latexstring("Local Iterations Profile ($nodes)\nBest: $(join(title_bits, "; "))")

    sites_range = 1:length(base_iters)

    p_iters = plot(sites_range, base_iters,
        label="Baseline", color=:red, lw=2, marker=:circle,
        title=iters_title,
        xlabel="MPS Site Index",
        ylabel="Krylov Iterations",
        legend=:outertopright)

    if !isnothing(best_fai) && !isnothing(best_fai["iters_per_site"])
        plot!(p_iters, sites_range, best_fai["iters_per_site"],
            label="FAI", color=:blue, lw=2, marker=:square)
        plot!(p_iters, sites_range, base_iters, fillrange=best_fai["iters_per_site"],
            fillalpha=0.12, fillcolor=:blue, label="", linealpha=0)
    end
    if !isnothing(best_sinc) && !isnothing(best_sinc["iters_per_site"])
        plot!(p_iters, sites_range, best_sinc["iters_per_site"],
            label="Sinc", color=:darkorange, lw=2, marker=:diamond)
    end

    savefig(p_iters, joinpath(plots_dir, "site_iterations_N_$(nodes).png"))

    if isnothing(base_conds) || !nonempty_numeric_vec(base_conds)
        @info "Skipping condition number plots for $nodes (no baseline cond data)"
        continue
    end

    base_conds_clean = clean_kappa(base_conds)
    best_fai_cond = isnothing(best_fai) ? nothing : match_cond(grid_data_cond, best_fai)
    best_sinc_cond = isnothing(best_sinc) ? nothing : match_cond(grid_data_cond, best_sinc)
    fai_kpc = isnothing(best_fai_cond) ? nothing : kappa_pc(best_fai_cond)
    sinc_kpc = isnothing(best_sinc_cond) ? nothing : kappa_pc(best_sinc_cond)

    p_conds = plot(sites_range, base_conds_clean,
        label="Baseline", color=:red, lw=2, marker=:circle, yaxis=:log,
        title="Local Condition Number Profile ($nodes)",
        xlabel="MPS Site Index",
        ylabel="Condition Number (Log Scale)",
        legend=:outertopright)

    if !isnothing(fai_kpc)
        plot!(p_conds, sites_range, fai_kpc, label="FAI", color=:blue, lw=2, marker=:square)
    else
        @info "No FAI κ(P⁻¹A) for $nodes; omitting FAI from condition profile"
    end
    if !isnothing(sinc_kpc)
        plot!(p_conds, sites_range, sinc_kpc, label="Sinc", color=:darkorange, lw=2, marker=:diamond)
    else
        @info "No sinc κ(P⁻¹A) for $nodes; omitting sinc from condition profile"
    end

    savefig(p_conds, joinpath(plots_dir, "site_condition_N_$(nodes).png"))

    p_cond_imp = plot(
        title="Condition Number Improvement ($nodes)\nRatio: unpreconditioned / preconditioned",
        xlabel="MPS Site Index",
        ylabel="Improvement Factor",
        legend=:outertopright)

    hline!(p_cond_imp, [1.0], color=:red, lw=2, ls=:dash, label="No improvement")
    if !isnothing(fai_kpc)
        plot!(p_cond_imp, sites_range, base_conds_clean ./ fai_kpc,
            label="FAI", color=:blue, lw=2, marker=:square)
    end
    if !isnothing(sinc_kpc)
        plot!(p_cond_imp, sites_range, base_conds_clean ./ sinc_kpc,
            label="Sinc", color=:darkorange, lw=2, marker=:diamond)
    end
    savefig(p_cond_imp, joinpath(plots_dir, "site_cond_improvement_N_$(nodes).png"))
end

# ---------------------------------------------------------
# 4. Scaling View
# ---------------------------------------------------------
if !isempty(grid_labels)
    p_scale = plot(1:length(grid_labels), max_solve_speedups_fai,
        label="Max FAI Solve Speedup", color=:blue, lw=3, marker=:circle, markersize=8,
        title="Scaling of Maximum Achievable Speedups",
        xlabel="System Size",
        ylabel="Speedup Factor",
        xticks=(1:length(grid_labels), grid_labels),
        legend=:topright,
        framestyle=:box)

    plot!(p_scale, 1:length(grid_labels), max_total_speedups_fai,
        label="Max FAI Total Speedup", color=:green, lw=3, marker=:square, markersize=8)
    plot!(p_scale, 1:length(grid_labels), max_solve_speedups_sinc,
        label="Max Sinc Solve Speedup", color=:darkorange, lw=3, marker=:diamond, markersize=8)
    plot!(p_scale, 1:length(grid_labels), max_total_speedups_sinc,
        label="Max Sinc Total Speedup", color=:brown, lw=3, marker=:utriangle, markersize=8)

    hline!(p_scale, [1.0], label="Break-even (1.0x)", color=:red, lw=2, ls=:dash)

    savefig(p_scale, joinpath(plots_dir, "scaling_summary.png"))
end

@info "All plots successfully generated in $plots_dir"