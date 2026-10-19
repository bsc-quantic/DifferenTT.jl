"""
    TT Anisotropy Benchmark (local ResidualALS): Plotting Script

Uses nocond results for timings, iterations and residuals, and cond results (if present) for the
local condition numbers. Generates, per dimension N, with the floor F on the x-axis:
    1. Solve / total speedup vs baseline
    2. Total local Krylov iterations
    3. Local condition numbers κ(A) and κ(P⁻¹A), with the analytic and discrete floors as reference
    4. Sinc plateau: local κ(P⁻¹A) (or iterations) vs M, one line per F
    5. Preconditioner build time and solution bond dimension
    6. Final relative residual (convergence check)
plus crossover_summary.md / crossover_ratio.png across N.

Note: `avg_pc_cond` is the average over sites of the local κ(P⁻¹A) of the projected problems, not
the global κ(P L), so the floor lines are a reference, not a bound.
"""

include(joinpath(@__DIR__, "..", "anisotropy_plot_utils.jl"))

base_folder = "experiments/preconditioner_benchmarks/tt_anisotropy"
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
