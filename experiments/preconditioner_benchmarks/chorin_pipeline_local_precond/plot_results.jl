"""
    FAI Preconditioner Benchmark: Local Precond Plotting Script
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

results_file_nocond = joinpath(nocond_folder, "collected_results.jld2")
@info "Loading nocond dataframe from \$results_file_nocond"
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
        "iters" => [mean(step_iters) for step_iters in row.iters],
        "iters_per_site" => hasproperty(row, :iters_per_site) ? row.iters_per_site : nothing,
        "n_steps" => length(row.solve_times)
    )
    results_nocond[nodes][k] = d
end

system_sizes = sort(collect(keys(results_nocond)))

max_total_speedups = Float64[]
max_solve_speedups = Float64[]
grid_labels = String[]

for nodes in system_sizes
    @info "Generating plots for system size \$nodes"
    push!(grid_labels, "\$(nodes[1])x\$(nodes[2])")

    grid_data = results_nocond[nodes]
    if !haskey(grid_data, "baseline")
        @warn "Skipping \$nodes: Missing baseline data"
        push!(max_solve_speedups, NaN)
        push!(max_total_speedups, NaN)
        continue
    end

    base = grid_data["baseline"]
    base_time = base["total_solve_time"]
    base_iters = base["iters"]
    base_solve_times = base["solve_times"]
    n_steps = base["n_steps"]
    base_iters_per_site = base["iters_per_site"] ./ n_steps
    
    out_dir = joinpath(plots_dir, "\$(nodes[1])x\$(nodes[2])")
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
    pf_plot = plot(title="Pareto Front: Build Time vs Total Solve Time (\$nodes)",
                   xlabel="Preconditioner Build Time (s)",
                   ylabel="Total Solve Time (s)", legend=:outertopright)
    
    for gdim in gdims
        for mode in ["precond_all", "precond_local"]
            idx = findfirst(x -> gdims_list[x] == gdim && modes[x] == mode, 1:length(gdims_list))
            if !isnothing(idx)
                mode_label = mode == "precond_all" ? "All Sites" : "Local"
                shape = mode == "precond_all" ? :circle : :square
                scatter!(pf_plot, [build_times[idx]], [total_solve_times[idx]],
                    label=latexstring("FAI \\\$\\\\chi_{max}=\$(gdim)\\\$ (\$mode_label)"),
                    marker=shape, markersize=6)
            end
        end
    end
    hline!(pf_plot, [base_time], label="Baseline Total Solve", color=:red, lw=2, ls=:dash)
    plot!(pf_plot, [0, base_time], [base_time, 0], fillrange=0, fillalpha=0.1, fillcolor=:green, linealpha=0, label="Net Speedup Region")
    xlims!(pf_plot, 0, maximum(build_times) * 1.05)
    ylims!(pf_plot, 0, max(maximum(total_solve_times), base_time) * 1.05)
    savefig(pf_plot, joinpath(out_dir, "pareto.png"))
    
    # Combined Iterations and Speedup across all configurations
    p_iters_comb = plot(title="Krylov Iterations vs Time Step (\$nodes)", xlabel="Chorin Time Step", ylabel="Avg Krylov Iterations", legend=:outertopright)
    p_sp_comb = plot(title="Solve Speedup vs Time Step (\$nodes)", xlabel="Chorin Time Step", ylabel="Speedup Ratio", legend=:outertopright)
    
    plot!(p_iters_comb, 1:n_steps, base_iters, label="Baseline", color=:black, lw=3)
    hline!(p_sp_comb, [1.0], color=:black, lw=3, ls=:dash, label="Break-even (1.0x)")
    
    colors = palette(:tab20)
    for (i, gdim) in enumerate(gdims)
        for mode in ["precond_all", "precond_local"]
            idx = findfirst(x -> gdims_list[x] == gdim && modes[x] == mode, 1:length(gdims_list))
            if !isnothing(idx)
                v = grid_data[[k for (k, val) in grid_data if val["mode"] == mode && val["G_maxdim"] == gdim][1]]
                mode_label = mode == "precond_all" ? "All" : "Local"
                shape = mode == "precond_all" ? :circle : :square
                
                plot!(p_iters_comb, 1:n_steps, v["iters"], label=latexstring("FAI \\\$\\\\chi_{max}=\$(gdim)\\\$ (\$mode_label)"), lw=1, marker=shape, markersize=3)
                plot!(p_sp_comb, 1:n_steps, base_solve_times ./ v["solve_times"], label=latexstring("FAI \\\$\\\\chi_{max}=\$(gdim)\\\$ (\$mode_label)"), lw=1, marker=shape, markersize=3)
            end
        end
    end
    savefig(p_iters_comb, joinpath(out_dir, "combined_step_iterations.png"))
    savefig(p_sp_comb, joinpath(out_dir, "combined_step_speedup.png"))
    
    # New plot: Site-by-Site Average Iterations
    for gdim in gdims
        p_site = plot(title="Avg Krylov Iterations per Site (\$nodes)\nPreconditioner Rank = \$gdim",
                      xlabel="MPS Site Index", ylabel="Avg Iterations", legend=:outertopright)
        
        plot!(p_site, 1:length(base_iters_per_site), base_iters_per_site, label="Baseline", color=:black, lw=2)
        
        for mode in ["precond_all", "precond_local"]
            idx = findfirst(x -> gdims_list[x] == gdim && modes[x] == mode, 1:length(gdims_list))
            if !isnothing(idx)
                v = grid_data[[k for (k, val) in grid_data if val["mode"] == mode && val["G_maxdim"] == gdim][1]]
                iters_per_site = v["iters_per_site"] ./ n_steps
                mode_label = mode == "precond_all" ? "All Sites" : "Local"
                c = mode == "precond_all" ? :blue : :red
                shape = mode == "precond_all" ? :circle : :square
                plot!(p_site, 1:length(iters_per_site), iters_per_site, label=mode_label, color=c, lw=2, marker=shape)
            end
        end
        savefig(p_site, joinpath(out_dir, "site_iterations_gdim_\$(gdim).png"))
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
