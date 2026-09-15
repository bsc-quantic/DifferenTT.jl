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
base_folder = "experiments/preconditioner_benchmarks/tt_poisson_global"
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

results_file = joinpath(base_folder, "results", "collected_results.jld2")
@info "Loading dataframe from $results_file"
df = load(results_file, "df")

results_dict = Dict()
for row in eachrow(df)
    nodes = row.N
    if !haskey(results_dict, nodes)
        results_dict[nodes] = Dict()
    end
    k = row.key
    d = Dict(
        "key" => k,
        "method" => row.method,
        "time" => row.solve_time,
        "solve_time" => row.solve_time,
        "total_time" => row.total_time,
        "precond_build_time" => row.precond_build_time,
        "mu" => hasproperty(row, :mu) && !ismissing(row.mu) ? row.mu : NaN,
        "G_maxdim" => hasproperty(row, :G_maxdim) && !ismissing(row.G_maxdim) ? row.G_maxdim : 0,
        "M" => hasproperty(row, :M) && !ismissing(row.M) ? Int(row.M) : 0,
        "global_iters" => hasproperty(row, :global_iters) ? row.global_iters : nothing
    )
    results_dict[nodes][k] = d
end

system_sizes = sort(collect(keys(results_dict)))

# To track max speedups and iterations for the scaling plot
max_total_speedups_fai = Float64[]
max_solve_speedups_fai = Float64[]
max_total_speedups_sinc = Float64[]
max_solve_speedups_sinc = Float64[]
baseline_iters_list = Float64[]
min_iters_fai = Float64[]
min_iters_sinc = Float64[]
grid_labels = String[]

detailed_fai_solve = Dict{Tuple{Int, Float64}, Vector{Float64}}()
detailed_fai_total = Dict{Tuple{Int, Float64}, Vector{Float64}}()
detailed_fai_iters = Dict{Tuple{Int, Float64}, Vector{Float64}}()
detailed_sinc_solve = Dict{Int, Vector{Float64}}()
detailed_sinc_total = Dict{Int, Vector{Float64}}()
detailed_sinc_iters = Dict{Int, Vector{Float64}}()

for (node_idx, nodes) in enumerate(system_sizes)
    @info "Generating plots for system size $nodes"

    grid_data = results_dict[nodes]

    if !haskey(grid_data, "baseline")
        @warn "Skipping $nodes: Missing baseline data"
        continue
    end

    base_time = grid_data["baseline"]["time"]
    base_iters = get(grid_data["baseline"], "global_iters", nothing)

    # FAI grid-search arrays
    mus = Float64[]
    gdims = Int[]
    fai_solve_speedups = Float64[]
    fai_total_speedups = Float64[]
    fai_iters_list = Float64[]
    cond_improvement_factors = Float64[]
    fai_build_times = Float64[]
    fai_solve_times = Float64[]
    fai_keys = String[]

    sinc_Ms = Int[]
    sinc_solve_speedups = Float64[]
    sinc_total_speedups = Float64[]
    sinc_iters_list = Float64[]
    sinc_build_times = Float64[]
    sinc_solve_times = Float64[]
    sinc_keys = String[]

    for (k, v) in grid_data
        v["method"] == "baseline" && continue
        if v["method"] == "sinc"
            m_val = v["M"]
            push!(sinc_keys, k)
            push!(sinc_Ms, m_val)
            push!(sinc_build_times, v["precond_build_time"])
            push!(sinc_solve_times, v["solve_time"])
            s_sp = base_time / v["solve_time"]
            t_sp = base_time / v["total_time"]
            push!(sinc_solve_speedups, s_sp)
            push!(sinc_total_speedups, t_sp)

            s_iter = get(v, "global_iters", nothing)
            s_it_val = isnothing(s_iter) || ismissing(s_iter) ? NaN : Float64(s_iter)
            push!(sinc_iters_list, s_it_val)

            if !haskey(detailed_sinc_solve, m_val)
                detailed_sinc_solve[m_val] = fill(NaN, length(system_sizes))
                detailed_sinc_total[m_val] = fill(NaN, length(system_sizes))
                detailed_sinc_iters[m_val] = fill(NaN, length(system_sizes))
            end
            detailed_sinc_solve[m_val][node_idx] = s_sp
            detailed_sinc_total[m_val][node_idx] = t_sp
            detailed_sinc_iters[m_val][node_idx] = s_it_val
        elseif v["method"] == "fai"
            mu_val = v["mu"]
            gdim_val = v["G_maxdim"]
            push!(fai_keys, k)
            push!(mus, mu_val)
            push!(gdims, gdim_val)
            push!(fai_build_times, v["precond_build_time"])
            push!(fai_solve_times, v["solve_time"])
            s_sp = base_time / v["solve_time"]
            t_sp = base_time / v["total_time"]
            push!(fai_solve_speedups, s_sp)
            push!(fai_total_speedups, t_sp)

            f_iter = get(v, "global_iters", nothing)
            f_it_val = isnothing(f_iter) || ismissing(f_iter) ? NaN : Float64(f_iter)
            push!(fai_iters_list, f_it_val)

            cfg = (gdim_val, mu_val)
            if !haskey(detailed_fai_solve, cfg)
                detailed_fai_solve[cfg] = fill(NaN, length(system_sizes))
                detailed_fai_total[cfg] = fill(NaN, length(system_sizes))
                detailed_fai_iters[cfg] = fill(NaN, length(system_sizes))
            end
            detailed_fai_solve[cfg][node_idx] = s_sp
            detailed_fai_total[cfg][node_idx] = t_sp
            detailed_fai_iters[cfg][node_idx] = f_it_val
        end
    end

    if isempty(mus) && isempty(sinc_Ms)
        @warn "No preconditioned runs found for $nodes in nocond results"
        continue
    end

    # Sort FAI and Sinc keys so bar plots are ordered by chi and M
    sort!(fai_keys, by = k -> (get(grid_data[k], "G_maxdim", 0), get(grid_data[k], "mu", NaN)))
    sort!(sinc_keys, by = k -> get(grid_data[k], "M", 0))

    # Track for scaling plot
    push!(grid_labels, "N_$(nodes)")
    push!(max_solve_speedups_fai, isempty(fai_solve_speedups) ? NaN : maximum(fai_solve_speedups))
    push!(max_total_speedups_fai, isempty(fai_total_speedups) ? NaN : maximum(fai_total_speedups))
    push!(max_solve_speedups_sinc, isempty(sinc_solve_speedups) ? NaN : maximum(sinc_solve_speedups))
    push!(max_total_speedups_sinc, isempty(sinc_total_speedups) ? NaN : maximum(sinc_total_speedups))

    b_it = isnothing(base_iters) || ismissing(base_iters) ? NaN : Float64(base_iters)
    push!(baseline_iters_list, b_it)
    val_fai = filter(!isnan, fai_iters_list)
    push!(min_iters_fai, isempty(val_fai) ? NaN : minimum(val_fai))
    val_sinc = filter(!isnan, sinc_iters_list)
    push!(min_iters_sinc, isempty(val_sinc) ? NaN : minimum(val_sinc))

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
    # 3. Global Iteration Profiles
    # ---------------------------------------------------------
    base_global_iters = get(grid_data["baseline"], "global_iters", nothing)
    if !isnothing(base_global_iters) && !ismissing(base_global_iters)
        labels = ["Baseline"]
        iters = [base_global_iters]
        bar_colors = [:red]

        # Group FAI configurations
        for k in fai_keys
            v = grid_data[k]
            if !isnothing(v["global_iters"]) && !ismissing(v["global_iters"])
                push!(labels, "FAI χ=$(v["G_maxdim"])")
                push!(iters, v["global_iters"])
                push!(bar_colors, :blue)
            end
        end

        # Group Sinc configurations
        for k in sinc_keys
            v = grid_data[k]
            if !isnothing(v["global_iters"]) && !ismissing(v["global_iters"])
                push!(labels, "Sinc M=$(v["M"])")
                push!(iters, v["global_iters"])
                push!(bar_colors, :darkorange)
            end
        end

        p_iters = bar(labels, iters,
            title="Global Iterations (N=$(nodes), d=8)",
            ylabel="Total Krylov Iterations",
            color=bar_colors,
            legend=false,
            yscale=:log10,
            xrotation=45,
            bottom_margin=15Plots.mm)

        savefig(p_iters, joinpath(plots_dir, "global_iterations_N_$(nodes).png"))
    end

    # ---------------------------------------------------------
    # 3b. Solve Time Profiles
    # ---------------------------------------------------------
    base_solve_time = get(grid_data["baseline"], "solve_time", nothing)
    if !isnothing(base_solve_time) && !ismissing(base_solve_time)
        labels_st = ["Baseline"]
        stimes = [base_solve_time]
        bar_colors_st = [:red]

        # Group FAI configurations
        for k in fai_keys
            v = grid_data[k]
            if !isnothing(v["solve_time"]) && !ismissing(v["solve_time"])
                push!(labels_st, "FAI χ=$(v["G_maxdim"])")
                push!(stimes, v["solve_time"])
                push!(bar_colors_st, :blue)
            end
        end

        # Group Sinc configurations
        for k in sinc_keys
            v = grid_data[k]
            if !isnothing(v["solve_time"]) && !ismissing(v["solve_time"])
                push!(labels_st, "Sinc M=$(v["M"])")
                push!(stimes, v["solve_time"])
                push!(bar_colors_st, :darkorange)
            end
        end

        p_times = bar(labels_st, stimes,
            title="Solve Time (N=$(nodes), d=8)",
            ylabel="Time (s)",
            color=bar_colors_st,
            legend=false,
            xrotation=45,
            yscale=:log10,
            bottom_margin=15Plots.mm)

        savefig(p_times, joinpath(plots_dir, "solve_times_N_$(nodes).png"))
    end
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

    p_scale_iters = plot(
        title="Scaling of Iterations",
        xlabel="System Size",
        ylabel="Total Krylov Iterations",
        yscale=:log10,
        xticks=(1:length(grid_labels), grid_labels),
        legend=:topleft,
        framestyle=:box)

    if any(!isnan, baseline_iters_list)
        plot!(p_scale_iters, 1:length(grid_labels), baseline_iters_list,
            label="Baseline", color=:black, lw=3, marker=:circle, markersize=8)
    end
    plot!(p_scale_iters, 1:length(grid_labels), min_iters_fai,
        label="Min FAI Iterations", color=:blue, lw=3, marker=:square, markersize=8)
    plot!(p_scale_iters, 1:length(grid_labels), min_iters_sinc,
        label="Min Sinc Iterations", color=:darkorange, lw=3, marker=:diamond, markersize=8)

    savefig(p_scale_iters, joinpath(plots_dir, "scaling_summary_iters.png"))

    p_scale_det_solve = plot(
        title="Detailed Scaling of Solve Speedups",
        xlabel="System Size",
        ylabel="Speedup Factor",
        xticks=(1:length(grid_labels), grid_labels),
        legend=:outertopright,
        framestyle=:box)

    p_scale_det_total = plot(
        title="Detailed Scaling of Total Speedups",
        xlabel="System Size",
        ylabel="Speedup Factor",
        xticks=(1:length(grid_labels), grid_labels),
        legend=:outertopright,
        framestyle=:box)

    p_scale_det_iters = plot(
        title="Detailed Scaling of Iterations",
        xlabel="System Size",
        ylabel="Total Krylov Iterations",
        yscale=:log10,
        xticks=(1:length(grid_labels), grid_labels),
        legend=:outertopright,
        framestyle=:box)

    if any(!isnan, baseline_iters_list)
        plot!(p_scale_det_iters, 1:length(grid_labels), baseline_iters_list,
              label="Baseline", color=:black, lw=3, marker=:circle, markersize=6)
    end

    fai_cfgs = sort(collect(keys(detailed_fai_solve)))
    sinc_cfgs = sort(collect(keys(detailed_sinc_solve)))
    
    fai_colors = [cgrad(:Blues)[z] for z in range(0.45, 1.0, length=max(2, length(fai_cfgs)))]
    sinc_colors = [cgrad(:Oranges)[z] for z in range(0.45, 1.0, length=max(2, length(sinc_cfgs)))]

    for (i, cfg) in enumerate(fai_cfgs)
        g, m = cfg
        c = fai_colors[mod1(i, length(fai_colors))]
        plot!(p_scale_det_solve, 1:length(grid_labels), detailed_fai_solve[cfg], 
              label="FAI χ=$(g)", lw=2.5, marker=:circle, color=c)
        plot!(p_scale_det_total, 1:length(grid_labels), detailed_fai_total[cfg], 
              label="FAI χ=$(g)", lw=2.5, marker=:square, color=c)
        if haskey(detailed_fai_iters, cfg)
            plot!(p_scale_det_iters, 1:length(grid_labels), detailed_fai_iters[cfg], 
                  label="FAI χ=$(g)", lw=2.5, marker=:circle, color=c)
        end
    end
    
    for (i, m_val) in enumerate(sinc_cfgs)
        c = sinc_colors[mod1(i, length(sinc_colors))]
        plot!(p_scale_det_solve, 1:length(grid_labels), detailed_sinc_solve[m_val], 
              label="Sinc M=$(m_val)", lw=2.5, marker=:diamond, color=c)
        plot!(p_scale_det_total, 1:length(grid_labels), detailed_sinc_total[m_val], 
              label="Sinc M=$(m_val)", lw=2.5, marker=:utriangle, color=c)
        if haskey(detailed_sinc_iters, m_val)
            plot!(p_scale_det_iters, 1:length(grid_labels), detailed_sinc_iters[m_val], 
                  label="Sinc M=$(m_val)", lw=2.5, marker=:diamond, color=c)
        end
    end

    hline!(p_scale_det_solve, [1.0], label="Break-even (1.0x)", color=:red, lw=2, ls=:dash)
    hline!(p_scale_det_total, [1.0], label="Break-even (1.0x)", color=:red, lw=2, ls=:dash)

    savefig(p_scale_det_solve, joinpath(plots_dir, "scaling_summary_detailed_solve.png"))
    savefig(p_scale_det_total, joinpath(plots_dir, "scaling_summary_detailed_total.png"))
    savefig(p_scale_det_iters, joinpath(plots_dir, "scaling_summary_detailed_iters.png"))
end

@info "All plots successfully generated in $plots_dir"