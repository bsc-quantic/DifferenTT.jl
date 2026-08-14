using JLD2
using DataFrames
using Plots
using Statistics

gr()
default(fmt=:png, dpi=300, size=(900, 500), fontfamily="Computer Modern", margin=5Plots.mm)

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/tt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/tt_poisson/results"
input_file = joinpath(base_folder, "collected_results.jld2")

df = load(input_file, "df")

if ncol(df) == 0 || nrow(df) == 0
    println("Warning: Dataframe is empty or has no columns in $(input_file). Exiting gracefully.")
    exit(0)
end

plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

# 1. Scaling Plot (Total Time vs N)
p_scale = plot(
    xlabel = "Dimensions (N)",
    ylabel = "Total Solve Time (s)",
    title = "N-Dimensional TT Poisson Scaling",
    yaxis = :log10,
    legend = :outertopright,
    framestyle = :box
)

df_base = df[df.method .== "baseline", :]
sort!(df_base, :N)
if nrow(df_base) > 0
    plot!(p_scale, Float64.(df_base.N), df_base.total_time, label="Baseline", lw=3, color=:black, marker=:circle)
end

df_sinc = df[df.method .== "sinc", :]
for M in sort(unique(df_sinc.M))
    df_M = df_sinc[df_sinc.M .== M, :]
    sort!(df_M, :N)
    plot!(p_scale, Float64.(df_M.N), df_M.total_time, label="Sinc (M=$M)", lw=2, marker=:circle)
end

df_fai = df[df.method .== "fai", :]
for μ in sort(unique(df_fai.mu))
    for gdim in sort(unique(df_fai.G_maxdim))
        df_mg = df_fai[(df_fai.mu .== μ) .& (df_fai.G_maxdim .== gdim), :]
        if nrow(df_mg) > 0
            sort!(df_mg, :N)
            plot!(p_scale, Float64.(df_mg.N), df_mg.total_time, label="FAI (μ=$μ, χ=$gdim)", lw=2, ls=:dash, marker=:square)
        end
    end
end
savefig(p_scale, joinpath(plots_folder, "scaling_vs_N.png"))


# 2. Pareto Front (Build Time vs Solve Time) for a fixed N (e.g. N=8)
target_N = 8
df_N = df[df.N .== target_N, :]
if nrow(df_N) > 0
    p_pareto = plot(
        xlabel = "Preconditioner Build Time (s)",
        ylabel = "Solve Time (s)",
        title = "Pareto Front (N=$target_N)",
        legend = :outertopright,
        framestyle = :box
    )
    
    base_solve = df_N[df_N.method .== "baseline", :solve_time]
    base_t = length(base_solve) > 0 ? base_solve[1] : 0.0
    hline!(p_pareto, [base_t], color=:red, ls=:dash, label="Baseline Solve Time")
    
    df_f = df_N[df_N.method .== "fai", :]
    if nrow(df_f) > 0
        gdims = sort(unique(df_f.G_maxdim))
        colors = palette(:tab20)
        for (i, gdim) in enumerate(gdims)
            df_g = df_f[df_f.G_maxdim .== gdim, :]
            scatter!(p_pareto, df_g.precond_build_time, df_g.solve_time, color=colors[mod1(i, length(colors))], markersize=8, label="FAI χ=$gdim")
        end
    end
    
    df_s = df_N[df_N.method .== "sinc", :]
    if nrow(df_s) > 0
        scatter!(p_pareto, df_s.precond_build_time, df_s.solve_time, color=:green, marker=:diamond, markersize=8, label="Sinc Preconditioners")
    end
    
    savefig(p_pareto, joinpath(plots_folder, "pareto_N$(target_N).png"))
end

# 3. Site-by-site Iteration Profile (N=8)
if nrow(df_N) > 0 && !use_nocond
    p_iter = plot(
        xlabel = "MPS Site",
        ylabel = "Total Krylov Iterations",
        title = "Site-by-Site Iteration Profile (N=$target_N)",
        legend = :outertopright,
        framestyle = :box
    )
    
    df_b = df_N[df_N.method .== "baseline", :]
    if nrow(df_b) > 0 && hasproperty(df_b, :iters_per_site)
        iters_b = df_b.iters_per_site[1]
        x_sites = 1:length(iters_b)
        plot!(p_iter, x_sites, iters_b, label="Baseline", color=:black, lw=3)
    end
    
    # Best FAI based on solve time
    if nrow(df_f) > 0 && hasproperty(df_f, :iters_per_site)
        best_fai = df_f[argmin(df_f.solve_time), :]
        iters_f = best_fai.iters_per_site
        plot!(p_iter, 1:length(iters_f), iters_f, label="Best FAI (μ=$(best_fai.mu), χ=$(best_fai.G_maxdim))", color=:blue, lw=3)
        if nrow(df_b) > 0
            plot!(p_iter, 1:length(iters_f), df_b.iters_per_site[1], fillrange=iters_f, fillalpha=0.2, fillcolor=:green, label="Iterations Saved", linealpha=0)
        end
    end
    
    savefig(p_iter, joinpath(plots_folder, "iters_profile_N$(target_N).png"))
end
