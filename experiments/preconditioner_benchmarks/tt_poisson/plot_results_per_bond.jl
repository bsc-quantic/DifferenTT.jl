using JLD2
using DataFrames
using CairoMakie

use_nocond = length(ARGS) > 0 && ARGS[1] == "nocond"
base_folder = use_nocond ? "experiments/preconditioner_benchmarks/tt_poisson/results_nocond" : "experiments/preconditioner_benchmarks/tt_poisson/results"
input_file = joinpath(base_folder, "collected_results.jld2")

df = load(input_file, "df")

plots_folder = joinpath(base_folder, "plots")
mkpath(plots_folder)

system_dims = sort(unique(df.N))

for N in system_dims
    df_N = df[df.N .== N, :]
    
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel = "μ",
        ylabel = "Total Time (s)",
        title = "N=$N TT Poisson Preconditioning",
        xscale = log10,
        yscale = log10
    )
    
    # Baseline
    df_base = df_N[df_N.method .== "baseline", :]
    if nrow(df_base) > 0
        hlines!(ax, [df_base.solve_time[1]], color=:black, linestyle=:dash, label="Baseline")
    end
    
    # Sinc best (we don't plot against mu, but we can plot as horizontal lines)
    df_sinc = df_N[df_N.method .== "sinc", :]
    if nrow(df_sinc) > 0
        best_sinc = minimum(df_sinc.solve_time .+ df_sinc.precond_build_time)
        hlines!(ax, [best_sinc], color=:green, linestyle=:dash, label="Best Sinc")
    end
    
    # FAI
    df_fai = df_N[df_N.method .== "fai", :]
    gdims = sort(unique(df_fai.G_maxdim))
    
    for gdim in gdims
        df_gdim = df_fai[df_fai.G_maxdim .== gdim, :]
        sort!(df_gdim, :mu)
        
        lines!(ax, Float64.(df_gdim.mu), df_gdim.solve_time .+ df_gdim.precond_build_time, label="FAI χ=$(gdim)", linewidth=2, marker=:rect)
    end
    
    axislegend(ax, position=:lt)
    
    save(joinpath(plots_folder, "fai_per_bond_N_$(N).png"), fig)
end
