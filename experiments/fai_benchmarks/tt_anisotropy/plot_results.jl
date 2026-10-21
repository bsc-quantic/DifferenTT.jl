"""
    TT anisotropy benchmark (local ResidualALS): plotting script

We draw here the figures that compare the preconditioners of this benchmark from the collected results: speedup and
iteration counts against the baseline, and how they depend on the system size or on the preconditioner parameters.

# Figures

1. Solve / total speedup vs baseline
2. Total local Krylov iterations
3. Local condition numbers κ(A) and κ(P⁻¹A), with the analytic and discrete floors as reference
4. Sinc plateau: local κ(P⁻¹A) (or iterations) vs M, one line per F
5. Preconditioner build time and solution bond dimension
6. Final relative residual (convergence check)

We take the timings, iterations and residuals from the `nocond` results and the local condition numbers from the results
that record them. We plot, per dimension N, the floor F on the x axis, and, in addition, the file `crossover_summary.md`
and the figure `crossover_ratio.png` across N. `avg_pc_cond` is the average over sites of the local κ(P⁻¹A) of the
projected problems, not the global κ(P L), so the floor lines are a reference, not a bound.

# Inputs and outputs

Reads `results_nocond/collected_results.jld2` and, if it exists, `results/collected_results.jld2` of this folder, plus
the discrete floor of `paper/anisotropy_floor/discrete_floor.jld2` when that file exists, and writes PNG figures and
`crossover_summary.md` to `plots/`.

In smoke mode (`DIFFERENTT_SMOKE=1`) it reads the folder `DIFFERENTT_SMOKE_DIR` that the smoke worker wrote, never the
committed results; `experiments/smoke.jl` sets it.

# How to run

From the root of the repository, after `collect.jl`:

    julia --project=experiments experiments/fai_benchmarks/tt_anisotropy/plot_results.jl

# Resources

There is no job script: we run it locally on the collected data.

# Paper

None.
"""

include(joinpath(@__DIR__, "..", "..", "common", "smoke.jl"))
# Smoke mode reads (and writes) the folder DIFFERENTT_SMOKE_DIR instead of this folder.
const OUT_ROOT = output_root(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "common", "anisotropy_plot_utils.jl"))

base_folder = OUT_ROOT
nocond_folder = joinpath(base_folder, "results_nocond")
cond_folder = joinpath(base_folder, "results")
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

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

discrete_floor = load_discrete_floor()

for N in sort(unique(df_nocond.N))
    @info "Generating plots for N = $N"
    dfN = df_nocond[df_nocond.N .== N, :]
    dfN_cond = isempty(df_cond) ? DataFrame() : df_cond[df_cond.N .== N, :]

    plot_speedup_vs_F(dfN, N, :solve_time, plots_dir)
    plot_speedup_vs_F(dfN, N, :total_time, plots_dir)

    plot_metric_vs_F(
        dfN,
        :total_iters;
        title="Total local Krylov iterations (N=$N)",
        ylabel="Iterations",
        file=joinpath(plots_dir, "iters_vs_F_N_$(N).png"),
    )

    plot_metric_vs_F(
        dfN,
        :relative_residual;
        title="Final relative residual (N=$N)",
        ylabel=L"\|Ax-b\| / \|b\|",
        file=joinpath(plots_dir, "residual_vs_F_N_$(N).png"),
    )

    plot_cost_vs_F(dfN, N, plots_dir)

    if !isempty(dfN_cond) && hasproperty(dfN_cond, :avg_pc_cond)
        Fs = sort(unique(Float64.(dfN_cond.F)))
        # Baseline has no avg_pc_cond: use its unpreconditioned local κ(A) instead
        dfN_cond = transform(
            dfN_cond,
            AsTable(:) => ByRow(r -> r.method == "baseline" ? numval(r, :avg_cond) : numval(r, :avg_pc_cond)) => :kappa,
        )
        plot_metric_vs_F(
            dfN_cond,
            :kappa;
            title="Average local condition number (N=$N)",
            ylabel=L"\kappa(A)\ \mathrm{(baseline)},\ \kappa(P^{-1}A)",
            file=joinpath(plots_dir, "cond_vs_F_N_$(N).png"),
            refs=p -> add_floor_refs!(p, Fs, N, discrete_floor),
        )
        plot_sinc_plateau(dfN_cond, N, :avg_pc_cond, L"\kappa(P^{-1}A)", joinpath(plots_dir, "sinc_plateau_N_$(N).png"))
    else
        plot_sinc_plateau(
            dfN, N, :total_iters, "Total local Krylov iterations", joinpath(plots_dir, "sinc_plateau_N_$(N).png")
        )
    end
end

crossover_summary(df_nocond, plots_dir; iters_col=:total_iters)
@info "Plots saved to $plots_dir"
