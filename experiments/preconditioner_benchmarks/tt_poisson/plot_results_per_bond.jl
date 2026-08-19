"""
    FAI / Sinc Preconditioner Benchmark: Plotting Script per Bond Dimension

    Reads the extended grid search JLD2 results and generates comprehensive
    visualizations using Plots.jl, focusing on site-by-site Iteration and
    Condition Number profiles for the best μ at EACH bond dimension, with
    sinc quadrature overlays.

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
        "mu" => ismissing(row.mu) ? NaN : row.mu,
        "G_maxdim" => ismissing(row.G_maxdim) ? 0 : row.G_maxdim,
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
            "mu" => ismissing(row.mu) ? NaN : row.mu,
            "G_maxdim" => ismissing(row.G_maxdim) ? 0 : row.G_maxdim,
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
    base_conds_clean = nonempty_numeric_vec(base_conds) ? clean_kappa(base_conds) : nothing

    mus = Float64[]
    gdims = Int[]
    fai_solve_speedups = Float64[]
    fai_keys = String[]

    sinc_Ms = Int[]
    sinc_solve_speedups = Float64[]
    sinc_keys = String[]

    for (k, v) in grid_data_nocond
        v["method"] == "baseline" && continue
        if v["method"] == "sinc"
            push!(sinc_keys, k)
            push!(sinc_Ms, v["M"])
            push!(sinc_solve_speedups, base_time / v["solve_time"])
        elseif v["method"] == "fai"
            push!(fai_keys, k)
            push!(mus, v["mu"])
            push!(gdims, v["G_maxdim"])
            push!(fai_solve_speedups, base_time / v["solve_time"])
        end
    end

    if isempty(mus) && isempty(sinc_Ms)
        @warn "No preconditioned runs found for $nodes in nocond results"
        continue
    end

    unique_gdims = sort(unique(gdims))
    unique_Ms = sort(unique(sinc_Ms))

    sites_range_combined = !isnothing(base_iters) ? (1:length(base_iters)) : []
    p_iters_combined = plot(
        title=latexstring("Local Iterations Profile ($nodes)\nBest μ per \$\\chi_{max}\$, plus sinc"),
        xlabel="MPS Site Index", ylabel="Krylov Iterations", legend=:outertopright
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

@info "All per-bond plots successfully generated in $plots_dir"