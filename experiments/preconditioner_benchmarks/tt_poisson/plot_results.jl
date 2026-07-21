using JLD2
using DataFrames
using CairoMakie
using Statistics

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/tt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/tt_poisson/results"
input_file = joinpath(base_folder, "collected_results.jld2")

df = load(input_file, "df")
plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

# 1. Scaling Plot (Total Time vs N)
fig_scale = Figure(size=(800, 600))
ax_scale = Axis(fig_scale[1, 1], xlabel = "Dimensions (N)", ylabel = "Total Solve Time (s)", title = "N-Dimensional TT Poisson Scaling", yscale = log10)

df_base = df[df.method .== "baseline", :]
sort!(df_base, :N)
if nrow(df_base) > 0
    lines!(ax_scale, Float64.(df_base.N), df_base.total_time, label="Baseline", linewidth=3, color=:black)
    scatter!(ax_scale, Float64.(df_base.N), df_base.total_time, color=:black, markersize=10)
end

df_sinc = df[df.method .== "sinc", :]
for M in sort(unique(df_sinc.M))
    df_M = df_sinc[df_sinc.M .== M, :]
    sort!(df_M, :N)
    lines!(ax_scale, Float64.(df_M.N), df_M.total_time, label="Sinc (M=$M)", linewidth=2)
    scatter!(ax_scale, Float64.(df_M.N), df_M.total_time, markersize=8)
end

df_fai = df[df.method .== "fai", :]
for μ in sort(unique(df_fai.mu))
    for gdim in sort(unique(df_fai.G_maxdim))
        df_mg = df_fai[(df_fai.mu .== μ) .& (df_fai.G_maxdim .== gdim), :]
        if nrow(df_mg) > 0
            sort!(df_mg, :N)
            lines!(ax_scale, Float64.(df_mg.N), df_mg.total_time, label="FAI (μ=$μ, χ=$gdim)", linewidth=2, linestyle=:dash)
            scatter!(ax_scale, Float64.(df_mg.N), df_mg.total_time, markersize=8)
        end
    end
end
axislegend(ax_scale, position=:lt, nbanks=2)
save(joinpath(plots_folder, "scaling_vs_N.png"), fig_scale)

# 2. Pareto Front (Build Time vs Solve Time) for a fixed N (e.g. N=8)
target_N = 8
df_N = df[df.N .== target_N, :]
if nrow(df_N) > 0
    fig_pareto = Figure(size=(800, 600))
    ax_pareto = Axis(fig_pareto[1, 1], xlabel = "Preconditioner Build Time (s)", ylabel = "Solve Time (s)", title = "Pareto Front (N=$target_N)")
    
    base_solve = df_N[df_N.method .== "baseline", :solve_time]
    base_t = length(base_solve) > 0 ? base_solve[1] : 0.0
    hlines!(ax_pareto, [base_t], color=:red, linestyle=:dash, label="Baseline Solve Time")
    
    df_f = df_N[df_N.method .== "fai", :]
    if nrow(df_f) > 0
        scatter!(ax_pareto, df_f.precond_build_time, df_f.solve_time, color=:blue, markersize=12, label="FAI Preconditioners")
    end
    
    df_s = df_N[df_N.method .== "sinc", :]
    if nrow(df_s) > 0
        scatter!(ax_pareto, df_s.precond_build_time, df_s.solve_time, color=:green, marker=:diamond, markersize=12, label="Sinc Preconditioners")
    end
    
    axislegend(ax_pareto)
    save(joinpath(plots_folder, "pareto_N$(target_N).png"), fig_pareto)
end

# 3. Site-by-site Iteration Profile (N=8)
if nrow(df_N) > 0 && !use_nocond
    fig_iter = Figure(size=(900, 500))
    ax_iter = Axis(fig_iter[1, 1], xlabel = "MPS Site", ylabel = "Average Krylov Iterations", title = "Site-by-Site Iteration Profile (N=$target_N)")
    
    df_b = df_N[df_N.method .== "baseline", :]
    if nrow(df_b) > 0 && hasproperty(df_b, :iters_per_site)
        iters_b = df_b.iters_per_site[1]
        x_sites = 1:length(iters_b)
        lines!(ax_iter, x_sites, iters_b, label="Baseline", color=:black, linewidth=3)
    end
    
    # Best FAI based on solve time
    if nrow(df_f) > 0 && hasproperty(df_f, :iters_per_site)
        best_fai = df_f[argmin(df_f.solve_time), :]
        iters_f = best_fai.iters_per_site
        lines!(ax_iter, 1:length(iters_f), iters_f, label="Best FAI (μ=$(best_fai.mu), χ=$(best_fai.G_maxdim))", color=:blue, linewidth=3)
        if nrow(df_b) > 0
            band!(ax_iter, 1:length(iters_f), iters_f, df_b.iters_per_site[1], color=(:blue, 0.2))
        end
    end
    axislegend(ax_iter)
    save(joinpath(plots_folder, "iters_profile_N$(target_N).png"), fig_iter)
end
