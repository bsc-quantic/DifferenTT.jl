using JLD2
using DataFrames
using Plots

gr()
default(fmt=:png, dpi=300, size=(800, 600), fontfamily="Computer Modern", margin=5Plots.mm)

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

system_dims = sort(unique(df.N))

for N in system_dims
    df_N = df[df.N .== N, :]
    
    p = plot(
        xlabel = "μ",
        ylabel = "Total Time (s)",
        title = "N=$N TT Poisson Preconditioning",
        xaxis = :log10,
        yaxis = :log10,
        legend = :outertopright,
        framestyle = :box
    )
    
    # Baseline
    df_base = df_N[df_N.method .== "baseline", :]
    if nrow(df_base) > 0
        hline!(p, [df_base.solve_time[1]], color=:black, ls=:dash, label="Baseline")
    end
    
    # Sinc best (we don't plot against mu, but we can plot as horizontal lines)
    df_sinc = df_N[df_N.method .== "sinc", :]
    if nrow(df_sinc) > 0
        best_sinc = minimum(df_sinc.solve_time .+ df_sinc.precond_build_time)
        hline!(p, [best_sinc], color=:green, ls=:dash, label="Best Sinc")
    end
    
    # FAI
    df_fai = df_N[df_N.method .== "fai", :]
    gdims = sort(unique(df_fai.G_maxdim))
    colors = palette(:tab20)
    
    for (i, gdim) in enumerate(gdims)
        df_gdim = df_fai[df_fai.G_maxdim .== gdim, :]
        sort!(df_gdim, :mu)
        
        plot!(p, Float64.(df_gdim.mu), df_gdim.solve_time .+ df_gdim.precond_build_time, label="FAI χ=$(gdim)", lw=2, marker=:rect, color=colors[mod1(i, length(colors))])
    end
    
    savefig(p, joinpath(plots_folder, "fai_per_bond_N_$(N).png"))
end
