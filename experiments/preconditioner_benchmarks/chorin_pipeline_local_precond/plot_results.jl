"""
    FAI Preconditioner Benchmark: Local Precond Plotting Script

    Reads the evolution distributed JLD2 results and generates visualizations:
    1. Pareto fronts of Build Time vs Total Solve Time
    2. Step-by-step Iteration profiles (total Krylov iterations)
    3. Step-by-step Speedup profiles
    4. Site-by-site iteration profiles (sum of iterations across full evolution)
    5. Step-by-step relative residual of every configuration
    6. Step-by-step divergence norm ratio (after / before the pressure correction) of every configuration
    7. Scaling behavior across system sizes
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

base_folder = "experiments/preconditioner_benchmarks/chorin_pipeline_local_precond"
nocond_folder = joinpath(base_folder, "results_nocond")
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

include(joinpath(@__DIR__, "..", "plot_cond_utils.jl"))

results_file_nocond = joinpath(nocond_folder, "collected_results.jld2")
if !isfile(results_file_nocond)
    alt_file = joinpath(base_folder, "results", "collected_results.jld2")
    if isfile(alt_file)
        results_file_nocond = alt_file
    end
end

if !isfile(results_file_nocond)
    @error "Results file not found at $results_file_nocond"
    exit(0)
end

@info "Loading nocond dataframe from $results_file_nocond"
df_nocond = load(results_file_nocond, "df")

results_nocond = Dict()
for row in eachrow(df_nocond)
    nodes = row.nodes
    if !haskey(results_nocond, nodes)
        results_nocond[nodes] = Dict()
    end
    k = row.key
    mode = hasproperty(row, :mode) ? row.mode : (k == "baseline" ? "baseline" : (startswith(k, "precond_all") ? "precond_all" : "precond_local"))
    
    d = Dict(
        "key" => k,
        "mode" => mode,
        "mu" => hasproperty(row, :mu) && !ismissing(row.mu) ? row.mu : NaN,
        "G_maxdim" => hasproperty(row, :G_maxdim) && !ismissing(row.G_maxdim) ? row.G_maxdim : 0,
        "precond_build_time" => row.precond_build_time,
        "total_solve_time" => row.total_solve_time,
        "solve_times" => row.solve_times,
        "iters" => [sum(step_iters) for step_iters in row.iters],
        "iters_per_site" => hasproperty(row, :iters_per_site) && !ismissing(row.iters_per_site) && !isnothing(row.iters_per_site) ?
                            row.iters_per_site :
                            (hasproperty(row, :iters) ? sum(row.iters) : Float64[]),
        "residuals" => row_vec(row, :residuals),
        "div_ratios" => row_vec(row, :div_ratios),
        "n_steps" => length(row.solve_times)
    )
    results_nocond[nodes][k] = d
end

system_sizes = sort(collect(keys(results_nocond)))

max_total_speedups = Float64[]
max_solve_speedups = Float64[]
grid_labels = String[]

for nodes in system_sizes
    @info "Generating plots for system size $nodes"
    push!(grid_labels, "$(nodes[1])x$(nodes[2])")

    grid_data = results_nocond[nodes]
    if !haskey(grid_data, "baseline")
        @warn "Skipping $nodes: Missing baseline data"
        push!(max_solve_speedups, NaN)
        push!(max_total_speedups, NaN)
        continue
    end

    base = grid_data["baseline"]
    base_time = base["total_solve_time"]
    base_iters = base["iters"]
    base_solve_times = base["solve_times"]
    n_steps = base["n_steps"]
    base_iters_per_site = get(base, "iters_per_site", Float64[])
    
    out_dir = joinpath(plots_dir, "$(nodes[1])x$(nodes[2])")
    mkpath(out_dir)

    gdims = unique([v["G_maxdim"] for (k, v) in grid_data if v["mode"] != "baseline"])
    sort!(gdims)
    
    build_times = Float64[]
    total_solve_times = Float64[]
    modes = String[]
    gdims_list = Int[]
    
    best_speedup_overall = 0.0
    best_total_speedup_overall = 0.0

    for (k, v) in grid_data
        if v["mode"] != "baseline"
            push!(build_times, v["precond_build_time"])
            push!(total_solve_times, v["total_solve_time"])
            push!(modes, v["mode"])
            push!(gdims_list, v["G_maxdim"])
            
            speedup_solve = sum(base_solve_times) / sum(v["solve_times"])
            speedup_total = base_time / (v["total_solve_time"] + v["precond_build_time"])
            best_speedup_overall = max(best_speedup_overall, speedup_solve)
            best_total_speedup_overall = max(best_total_speedup_overall, speedup_total)
        end
    end
    
    push!(max_solve_speedups, best_speedup_overall)
    push!(max_total_speedups, best_total_speedup_overall)
    
    # Pareto Plot
    pf_plot = plot(title="Pareto Front: Build Time vs Total Solve Time ($nodes)",
                   xlabel="Preconditioner Build Time (s)",
                   ylabel="Total Solve Time (s)", legend=:outertopright)
    
    colors = palette(:tab20)
    for (i, gdim) in enumerate(gdims)
        c = colors[mod1(i, length(colors))]
        for mode in ["precond_all", "precond_local"]
            idx = findfirst(x -> gdims_list[x] == gdim && modes[x] == mode, 1:length(gdims_list))
            if !isnothing(idx)
                mode_label = mode == "precond_all" ? "All Sites" : "Local"
                shape = mode == "precond_all" ? :circle : :square
                scatter!(pf_plot, [build_times[idx]], [total_solve_times[idx]],
                    label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ ($mode_label)"),
                    marker=shape, markersize=6, color=c)
            end
        end
    end
    hline!(pf_plot, [base_time], label="Baseline Total Solve", color=:red, lw=2, ls=:dash)
    plot!(pf_plot, [0, base_time], [base_time, 0], fillrange=0, fillalpha=0.1, fillcolor=:green, linealpha=0, label="Net Speedup Region")
    xlims!(pf_plot, 0, maximum(build_times) * 1.05)
    ylims!(pf_plot, 0, max(maximum(total_solve_times), base_time) * 1.05)
    savefig(pf_plot, joinpath(out_dir, "pareto.png"))
    
    # Combined Iterations, Speedup, and Site-by-Site Iterations across all configurations
    p_iters_comb = plot(title=latexstring("Total Krylov Iterations vs Time Step ($nodes)"),
                        xlabel="Chorin Time Step", ylabel="Total Krylov Iterations", legend=:outertopright)
    p_sp_comb = plot(title=latexstring("Solve Speedup vs Time Step ($nodes)"),
                     xlabel="Chorin Time Step", ylabel="Speedup Ratio", legend=:outertopright)
    
    sites_range_iters = nonempty_numeric_vec(base_iters_per_site) ? (1:length(base_iters_per_site)) : []
    p_site_comb = plot(title="Site-by-Site Sum of Iterations ($nodes)",
                       xlabel="MPS Site Index", ylabel="Total Krylov Iterations", legend=:outertopright)
    p_site_all_comb = plot(title="Site-by-Site Sum of Iterations ($nodes)\nAll Preconditioner Setups",
                           xlabel="MPS Site Index", ylabel="Total Krylov Iterations", legend=:outertopright,
                           size=(1050, 600), legendfont=font(8, "Computer Modern"))
    
    plot!(p_iters_comb, 1:n_steps, base_iters, label="Baseline", color=:black, lw=3)
    hline!(p_sp_comb, [1.0], color=:black, lw=3, ls=:dash, label="Break-even (1.0x)")
    if !isempty(sites_range_iters)
        plot!(p_site_comb, sites_range_iters, base_iters_per_site, label="Baseline", color=:black, lw=3, marker=:circle)
        plot!(p_site_all_comb, sites_range_iters, base_iters_per_site, label="Baseline", color=:black, lw=3, marker=:circle)
    end
    
    colors = palette(:tab20)
    for (i, gdim) in enumerate(gdims)
        c = colors[mod1(i, length(colors))]
        for mode in ["precond_all", "precond_local"]
            idx = findfirst(x -> gdims_list[x] == gdim && modes[x] == mode, 1:length(gdims_list))
            if !isnothing(idx)
                matching_keys = [k for (k, val) in grid_data if val["mode"] == mode && val["G_maxdim"] == gdim]
                if !isempty(matching_keys)
                    v = grid_data[matching_keys[1]]
                    mode_label = mode == "precond_all" ? "All" : "Local"
                    shape = mode == "precond_all" ? :circle : :square
                    
                    plot!(p_iters_comb, 1:n_steps, v["iters"], label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ ($mode_label)"), lw=1, marker=shape, markersize=3, color=c)
                    plot!(p_sp_comb, 1:n_steps, base_solve_times ./ v["solve_times"], label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ ($mode_label)"), lw=1, marker=shape, markersize=3, color=c)

                    if !isempty(sites_range_iters)
                        v_iters_site = get(v, "iters_per_site", nothing)
                        if nonempty_numeric_vec(v_iters_site)
                            plot!(p_site_comb, sites_range_iters, v_iters_site,
                                label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ ($mode_label)"),
                                lw=1.5, marker=shape, markersize=3, color=c)
                            plot!(p_site_all_comb, sites_range_iters, v_iters_site,
                                label=latexstring("FAI \$\\chi=$(gdim)\$ ($mode_label)"),
                                lw=1.5, marker=shape, markersize=3, color=c)
                        end
                    end
                end
            end
        end
    end
    savefig(p_iters_comb, joinpath(out_dir, "combined_step_iterations.png"))
    savefig(p_sp_comb, joinpath(out_dir, "combined_step_speedup.png"))
    if !isempty(sites_range_iters)
        savefig(p_site_comb, joinpath(out_dir, "combined_site_iterations.png"))
        savefig(p_site_all_comb, joinpath(out_dir, "combined_site_iterations_all_setups.png"))
    end

    # Per-step residual and divergence reduction of every configuration
    function step_series(metric)
        series = Any[(; label="Baseline", values=base[metric], color=:black, lw=3)]
        for (i, gdim) in enumerate(gdims), mode in ["precond_all", "precond_local"]
            matching_keys = [k for (k, val) in grid_data if val["mode"] == mode && val["G_maxdim"] == gdim]
            isempty(matching_keys) && continue
            v = grid_data[matching_keys[1]]
            mode_label = mode == "precond_all" ? "All" : "Local"
            push!(series, (; label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ ($mode_label)"), values=v[metric],
                color=colors[mod1(i, length(colors))], lw=1.5, ls=mode == "precond_all" ? :solid : :dash))
        end
        return series
    end
    p_res_all = step_profile_plot(step_series("residuals");
        title="Relative Residual vs Time Step ($nodes)\nAll Preconditioner Setups",
        ylabel="Relative Residual")
    p_div_all = step_profile_plot(step_series("div_ratios");
        title="Divergence Reduction by the Pressure Correction ($nodes)\nAll Preconditioner Setups",
        ylabel="Divergence Norm Ratio (after / before)")
    isnothing(p_res_all) || savefig(p_res_all, joinpath(out_dir, "combined_step_residuals_all_setups.png"))
    isnothing(p_div_all) || savefig(p_div_all, joinpath(out_dir, "combined_step_div_ratio_all_setups.png"))
    
    # Site-by-Site Iterations, Step Iterations, & Step Speedup per gdim
    for (i, gdim) in enumerate(gdims)
        p_site = plot(title=latexstring("Site-by-Site Sum of Iterations ($nodes)\nPreconditioner Rank \$\\chi_{max}=$(gdim)\$"),
                      xlabel="MPS Site Index", ylabel="Total Krylov Iterations", legend=:outertopright)
        
        if !isempty(sites_range_iters)
            plot!(p_site, sites_range_iters, base_iters_per_site, label="Baseline", color=:black, lw=2.5, marker=:circle)
        end
        
        p_iters_gdim = plot(title=latexstring("Total Krylov Iterations vs Time Step ($nodes)\nPreconditioner Rank \$\\chi_{max}=$(gdim)\$"),
                            xlabel="Chorin Time Step", ylabel="Total Krylov Iterations", legend=:outertopright)
        plot!(p_iters_gdim, 1:n_steps, base_iters, label="Baseline", color=:black, lw=2.5, marker=:circle)

        p_sp_gdim = plot(title=latexstring("Solve Speedup vs Time Step ($nodes)\nPreconditioner Rank \$\\chi_{max}=$(gdim)\$"),
                         xlabel="Chorin Time Step", ylabel="Speedup Ratio", legend=:outertopright)
        hline!(p_sp_gdim, [1.0], color=:black, lw=3, ls=:dash, label="Break-even (1.0x)")
        
        c_site = colors[mod1(i, length(colors))]
        for mode in ["precond_all", "precond_local"]
            idx = findfirst(x -> gdims_list[x] == gdim && modes[x] == mode, 1:length(gdims_list))
            if !isnothing(idx)
                matching_keys = [k for (k, val) in grid_data if val["mode"] == mode && val["G_maxdim"] == gdim]
                if !isempty(matching_keys)
                    v = grid_data[matching_keys[1]]
                    
                    v_iters_site = get(v, "iters_per_site", nothing)
                    mode_label = mode == "precond_all" ? "All Sites" : "Local"
                    shape = mode == "precond_all" ? :circle : :square
                    
                    if !isempty(sites_range_iters) && nonempty_numeric_vec(v_iters_site)
                        plot!(p_site, sites_range_iters, v_iters_site, label=mode_label, color=c_site, lw=2, marker=shape, ls=mode == "precond_all" ? :solid : :dash)
                    end
                    plot!(p_iters_gdim, 1:n_steps, v["iters"], label=mode_label, color=c_site, lw=2, marker=shape)
                    plot!(p_sp_gdim, 1:n_steps, base_solve_times ./ v["solve_times"], label=mode_label, color=c_site, lw=2, marker=shape, markersize=4)
                end
            end
        end
        savefig(p_site, joinpath(out_dir, "site_iterations_gdim_$(gdim).png"))
        savefig(p_iters_gdim, joinpath(out_dir, "step_iterations_gdim_$(gdim).png"))
        savefig(p_sp_gdim, joinpath(out_dir, "step_speedup_gdim_$(gdim).png"))

        chi_dir = joinpath(out_dir, "chi_$gdim")
        mkpath(chi_dir)
        savefig(p_site, joinpath(chi_dir, "site_iterations.png"))
        savefig(p_iters_gdim, joinpath(chi_dir, "step_iterations.png"))
        savefig(p_sp_gdim, joinpath(chi_dir, "step_speedup.png"))
    end
end

if !isempty(max_total_speedups) && length(max_total_speedups) > 1
    p_scale = plot(1:length(grid_labels), max_solve_speedups,
        label="Max Agg. Solve Speedup", color=:blue, lw=3, marker=:circle, markersize=8,
        title="Scaling of Maximum Achievable Speedups", xlabel="System Size", ylabel="Speedup Factor",
        xticks=(1:length(grid_labels), grid_labels), legend=:topleft)
    plot!(p_scale, 1:length(grid_labels), max_total_speedups, label="Max Agg. Total Speedup", color=:green, lw=3, marker=:square, markersize=8)
    hline!(p_scale, [1.0], label="Break-even", color=:red, lw=2, ls=:dash)
    savefig(p_scale, joinpath(plots_dir, "scaling_summary.png"))
end

@info "Plots successfully generated!"
