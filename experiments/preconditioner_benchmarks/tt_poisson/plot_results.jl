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

omit_unreliable_data = true # Toggle to omit N=4, N=50 and sinc M=1 data

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

if omit_unreliable_data
    system_sizes = filter(n -> n != 40 && n != 50, system_sizes)
end

# To track max speedups for the scaling plot
max_total_speedups_fai = Float64[]
max_solve_speedups_fai = Float64[]
max_total_speedups_sinc = Float64[]
max_solve_speedups_sinc = Float64[]
grid_labels = String[]

detailed_fai_solve = Dict{Tuple{Int, Float64}, Vector{Float64}}()
detailed_fai_total = Dict{Tuple{Int, Float64}, Vector{Float64}}()
detailed_sinc_solve = Dict{Int, Vector{Float64}}()
detailed_sinc_total = Dict{Int, Vector{Float64}}()

for (node_idx, nodes) in enumerate(system_sizes)
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
            m_val = v["M"]
            if omit_unreliable_data && m_val == 1
                continue
            end
            push!(sinc_keys, k)
            push!(sinc_Ms, m_val)
            push!(sinc_build_times, v["precond_build_time"])
            push!(sinc_solve_times, v["solve_time"])
            s_sp = base_time / v["solve_time"]
            t_sp = base_time / v["total_time"]
            push!(sinc_solve_speedups, s_sp)
            push!(sinc_total_speedups, t_sp)
            
            if !haskey(detailed_sinc_solve, m_val)
                detailed_sinc_solve[m_val] = fill(NaN, length(system_sizes))
                detailed_sinc_total[m_val] = fill(NaN, length(system_sizes))
            end
            detailed_sinc_solve[m_val][node_idx] = s_sp
            detailed_sinc_total[m_val][node_idx] = t_sp
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

            cfg = (gdim_val, mu_val)
            if !haskey(detailed_fai_solve, cfg)
                detailed_fai_solve[cfg] = fill(NaN, length(system_sizes))
                detailed_fai_total[cfg] = fill(NaN, length(system_sizes))
            end
            detailed_fai_solve[cfg][node_idx] = s_sp
            detailed_fai_total[cfg][node_idx] = t_sp

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
    unique_Ms = sort(unique(sinc_Ms))

    # ---------------------------------------------------------
    # 1. Performance Plots and Heatmaps — FAI only
    # ---------------------------------------------------------
    if !isempty(mus)
        Z_cond = fill(NaN, length(unique_gdims), length(unique_mus))

        for i in 1:length(mus)
            μ_idx = findfirst(==(mus[i]), unique_mus)
            g_idx = findfirst(==(gdims[i]), unique_gdims)
            Z_cond[g_idx, μ_idx] = cond_improvement_factors[i]
        end

        p_solve = plot(
            title="Solve Speedup ($nodes)",
            xlabel=L"Bond Dimension ($\chi_{max}$)",
            ylabel="Speedup Factor",
            legend=:outertopright,
            framestyle=:box
        )
        p_total = plot(
            title="Total Speedup ($nodes)",
            xlabel=L"Bond Dimension ($\chi_{max}$)",
            ylabel="Speedup Factor",
            legend=:outertopright,
            framestyle=:box
        )

        for μ in unique_mus
            idx = findall(==(μ), mus)
            p_idx = sortperm(gdims[idx])
            sorted_idx = idx[p_idx]
            
            plot!(p_solve, gdims[sorted_idx], fai_solve_speedups[sorted_idx], label="μ = $μ", marker=:circle, lw=2)
            plot!(p_total, gdims[sorted_idx], fai_total_speedups[sorted_idx], label="μ = $μ", marker=:circle, lw=2)
        end

        for M in unique_Ms
            idx = findfirst(==(M), sinc_Ms)
            if !isnothing(idx)
                hline!(p_solve, [sinc_solve_speedups[idx]], label="Sinc M = $M", lw=2, ls=:dashdot)
                hline!(p_total, [sinc_total_speedups[idx]], label="Sinc M = $M", lw=2, ls=:dashdot)
            end
        end

        hline!(p_solve, [1.0], label="Break-even (1.0x)", color=:black, lw=2, ls=:dash)
        hline!(p_total, [1.0], label="Break-even (1.0x)", color=:black, lw=2, ls=:dash)

        savefig(p_solve, joinpath(plots_dir, "plot_solve_N_$(nodes).png"))
        savefig(p_total, joinpath(plots_dir, "plot_total_N_$(nodes).png"))

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
        @info "Skipping FAI plots for $nodes (no FAI runs)"
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
    # 3. Per-Bond Profiles and Combined Overlays
    # ---------------------------------------------------------
    sites_range_combined = !isnothing(base_iters) ? (1:length(base_iters)) : []
    p_iters_combined = plot(
        title=latexstring("Local Iterations Profile ($nodes)\nBest μ per \$\\chi_{max}\$, plus sinc"),
        xlabel="MPS Site Index", ylabel="Krylov Iterations", legend=:outertopright, yaxis=:log10
    )
    p_conds_combined = plot(
        yaxis=:log, title=latexstring("Local Condition Number Profile ($nodes)\nBest μ per \$\\chi_{max}\$, plus sinc"),
        xlabel="MPS Site Index", ylabel="Condition Number (Log Scale)", legend=:outertopright
    )
    p_cond_imp_combined = plot(
        title=latexstring("Condition Number Improvement ($nodes)\nBest μ per \$\\chi_{max}\$, plus sinc"),
        xlabel="MPS Site Index", ylabel="Improvement Factor", legend=:outertopright
    )

    if !isnothing(base_iters)
        plot!(p_iters_combined, sites_range_combined, base_iters, label="Baseline", color=:black, lw=3, marker=:circle)
    end

    base_conds_clean = (isnothing(base_conds) || !nonempty_numeric_vec(base_conds)) ? nothing : clean_kappa(base_conds)
    if !isnothing(base_conds_clean)
        plot!(p_conds_combined, sites_range_combined, base_conds_clean, label="Baseline", color=:black, lw=3, marker=:circle)
        hline!(p_cond_imp_combined, [1.0], color=:black, lw=3, ls=:dash, label="No improvement")
    end

    colors = palette(:tab20)
    sinc_colors = palette(:Dark2)

    sinc_cfgs = Dict{Int, Dict}()
    for (si, M) in enumerate(unique_Ms)
        best_s_idx = -1
        best_s_sp = -1.0
        for idx in 1:length(sinc_solve_speedups)
            if sinc_Ms[idx] == M && sinc_solve_speedups[idx] > best_s_sp
                best_s_sp = sinc_solve_speedups[idx]
                best_s_idx = idx
            end
        end
        best_s_idx == -1 && continue
        sinc_v = grid_data_nocond[sinc_keys[best_s_idx]]
        sinc_cfgs[M] = sinc_v
        sinc_cond = match_cond(grid_data_cond, sinc_v)
        sinc_kpc = isnothing(sinc_cond) ? nothing : kappa_pc(sinc_cond)
        c_s = sinc_colors[mod1(si, length(sinc_colors))]

        if !isnothing(base_iters) && !isnothing(sinc_v["iters_per_site"])
            plot!(p_iters_combined, sites_range_combined, sinc_v["iters_per_site"],
                label="Sinc M=$(M)", color=c_s, lw=2, marker=:diamond, ls=:dash)
        end
        if !isnothing(base_conds_clean) && !isnothing(sinc_kpc)
            plot!(p_conds_combined, sites_range_combined, sinc_kpc,
                label="Sinc M=$(M)", color=c_s, lw=2, marker=:diamond, ls=:dash)
            plot!(p_cond_imp_combined, sites_range_combined, base_conds_clean ./ sinc_kpc,
                label="Sinc M=$(M)", color=c_s, lw=2, marker=:diamond, ls=:dash)
        end
    end

    for (i, gdim) in enumerate(unique_gdims)
        best_idx = -1
        best_speedup = -1.0

        for idx in 1:length(fai_solve_speedups)
            gdims[idx] != gdim && continue
            if fai_solve_speedups[idx] > best_speedup
                best_speedup = fai_solve_speedups[idx]
                best_idx = idx
            end
        end

        if best_idx == -1 || isnothing(base_iters)
            @warn "Skipping site-by-site plots for $nodes and gdim=$gdim: No suitable FAI data found"
            continue
        end

        best_v_nocond = grid_data_nocond[fai_keys[best_idx]]
        best_v_cond = match_cond(grid_data_cond, best_v_nocond)
        fai_kpc = isnothing(best_v_cond) ? nothing : kappa_pc(best_v_cond)

        sites_range = 1:length(base_iters)
        out_dir = joinpath(plots_dir, "N_$(nodes)", "chi_$gdim")
        mkpath(out_dir)

        p_iters = plot(sites_range, base_iters,
            label="Baseline", color=:red, lw=2, marker=:circle,
            title=latexstring("Local Iterations Profile ($nodes)\nBest FAI for \$\\chi_{max}=$(gdim)\$: μ=$(best_v_nocond["mu"])"),
            xlabel="MPS Site Index",
            ylabel="Krylov Iterations",
            legend=:outertopright)

        plot!(p_iters, sites_range, best_v_nocond["iters_per_site"],
            label="FAI", color=:blue, lw=2, marker=:square)

        plot!(p_iters, sites_range, base_iters, fillrange=best_v_nocond["iters_per_site"],
            fillalpha=0.12, fillcolor=:blue, label="", linealpha=0)

        for (si, M) in enumerate(unique_Ms)
            haskey(sinc_cfgs, M) || continue
            sinc_v = sinc_cfgs[M]
            isnothing(sinc_v["iters_per_site"]) && continue
            plot!(p_iters, sites_range, sinc_v["iters_per_site"],
                label="Sinc M=$(M)", color=sinc_colors[mod1(si, length(sinc_colors))], lw=2, marker=:diamond, ls=:dash)
        end

        savefig(p_iters, joinpath(out_dir, "site_iterations.png"))

        c_color = colors[mod1(i, length(colors))]
        plot!(p_iters_combined, sites_range, best_v_nocond["iters_per_site"],
            label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ (μ=$(best_v_nocond["mu"]))"), color=c_color, lw=2, marker=:square)

        p_conds = plot(sites_range, something(base_conds_clean, Float64[]),
            label="Baseline", color=:red, lw=2, marker=:circle, yaxis=:log,
            title="Local Condition Number Profile ($nodes)",
            xlabel="MPS Site Index",
            ylabel="Condition Number (Log Scale)",
            legend=:outertopright)

        plotted_any_pc = false
        if !isnothing(base_conds_clean) && !isnothing(fai_kpc)
            plot!(p_conds, sites_range, fai_kpc, label="FAI", color=:blue, lw=2, marker=:square)
            plotted_any_pc = true
            plot!(p_conds_combined, sites_range, fai_kpc,
                label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ (μ=$(best_v_nocond["mu"]))"), color=c_color, lw=2, marker=:square)
            plot!(p_cond_imp_combined, sites_range, base_conds_clean ./ fai_kpc,
                label=latexstring("FAI \$\\chi_{max}=$(gdim)\$ (μ=$(best_v_nocond["mu"]))"), color=c_color, lw=2, marker=:square)
        end

        p_cond_imp = plot(
            title="Condition Number Improvement ($nodes)\nRatio: unpreconditioned / preconditioned",
            xlabel="MPS Site Index",
            ylabel="Improvement Factor",
            legend=:outertopright)
        hline!(p_cond_imp, [1.0], color=:red, lw=2, ls=:dash, label="No improvement")

        if !isnothing(base_conds_clean) && !isnothing(fai_kpc)
            plot!(p_cond_imp, sites_range, base_conds_clean ./ fai_kpc,
                label="FAI", color=:blue, lw=2, marker=:square)
        end

        for (si, M) in enumerate(unique_Ms)
            haskey(sinc_cfgs, M) || continue
            sinc_cond = match_cond(grid_data_cond, sinc_cfgs[M])
            sinc_kpc = isnothing(sinc_cond) ? nothing : kappa_pc(sinc_cond)
            isnothing(sinc_kpc) && continue
            c_s = sinc_colors[mod1(si, length(sinc_colors))]
            if !isnothing(base_conds_clean)
                plot!(p_conds, sites_range, sinc_kpc,
                    label="Sinc M=$(M)", color=c_s, lw=2, marker=:diamond, ls=:dash)
                plot!(p_cond_imp, sites_range, base_conds_clean ./ sinc_kpc,
                    label="Sinc M=$(M)", color=c_s, lw=2, marker=:diamond, ls=:dash)
                plotted_any_pc = true
            end
        end

        if !isnothing(base_conds_clean)
            savefig(p_conds, joinpath(out_dir, "site_condition.png"))
            savefig(p_cond_imp, joinpath(out_dir, "site_cond_improvement.png"))
        end
    end

    out_dir_combined = joinpath(plots_dir, "N_$(nodes)")
    mkpath(out_dir_combined)

    if !isnothing(base_iters)
        savefig(p_iters_combined, joinpath(out_dir_combined, "combined_site_iterations.png"))
    end
    if !isnothing(base_conds_clean)
        savefig(p_conds_combined, joinpath(out_dir_combined, "combined_site_condition.png"))
        savefig(p_cond_imp_combined, joinpath(out_dir_combined, "combined_site_cond_improvement.png"))
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

    fai_cfgs = sort(collect(keys(detailed_fai_solve)))
    sinc_cfgs = sort(collect(keys(detailed_sinc_solve)))
    
    fai_colors = [cgrad(:Blues)[z] for z in range(0.45, 1.0, length=max(2, length(fai_cfgs)))]
    sinc_colors = [cgrad(:Oranges)[z] for z in range(0.45, 1.0, length=max(2, length(sinc_cfgs)))]

    for (i, cfg) in enumerate(fai_cfgs)
        g, m = cfg
        c = fai_colors[mod1(i, length(fai_colors))]
        plot!(p_scale_det_solve, 1:length(grid_labels), detailed_fai_solve[cfg], 
              label="FAI χ=$(g), μ=$(m)", lw=2.5, marker=:circle, color=c)
        plot!(p_scale_det_total, 1:length(grid_labels), detailed_fai_total[cfg], 
              label="FAI χ=$(g), μ=$(m)", lw=2.5, marker=:square, color=c)
    end
    
    for (i, m_val) in enumerate(sinc_cfgs)
        c = sinc_colors[mod1(i, length(sinc_colors))]
        plot!(p_scale_det_solve, 1:length(grid_labels), detailed_sinc_solve[m_val], 
              label="Sinc M=$(m_val)", lw=2.5, marker=:diamond, color=c)
        plot!(p_scale_det_total, 1:length(grid_labels), detailed_sinc_total[m_val], 
              label="Sinc M=$(m_val)", lw=2.5, marker=:utriangle, color=c)
    end

    hline!(p_scale_det_solve, [1.0], label="Break-even (1.0x)", color=:red, lw=2, ls=:dash)
    hline!(p_scale_det_total, [1.0], label="Break-even (1.0x)", color=:red, lw=2, ls=:dash)

    savefig(p_scale_det_solve, joinpath(plots_dir, "scaling_summary_detailed_solve.png"))
    savefig(p_scale_det_total, joinpath(plots_dir, "scaling_summary_detailed_total.png"))
end

@info "All plots successfully generated in $plots_dir"