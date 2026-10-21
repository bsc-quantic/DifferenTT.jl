"""
    TT anisotropy benchmark (global TT-CG): plotting script

We draw here the figures that compare the preconditioners of this benchmark from the collected results: speedup and
iteration counts against the baseline, and how they depend on the system size or on the preconditioner parameters.

# Figures

1. Solve / total speedup vs baseline
2. Global CG iterations
3. Sinc plateau: global CG iterations vs M, one line per F
4. Preconditioner build time and solution bond dimension
5. Final relative residual (convergence check; runs hitting maxiter are listed)

We plot, per dimension N, the floor F on the x axis, and, in addition, the file `crossover_summary.md` and the figure
`crossover_ratio.png` across N. Runs that hit `maxiter` are listed.

# Inputs and outputs

Reads `results/collected_results.jld2` of this folder, plus the discrete floor of
`paper/anisotropy_floor/discrete_floor.jld2` when that file exists, and writes PNG figures and `crossover_summary.md` to
`plots/`.

In smoke mode (`DIFFERENTT_SMOKE=1`) it reads the folder `DIFFERENTT_SMOKE_DIR` that the smoke worker wrote, never the
committed results; `experiments/smoke.jl` sets it.

# How to run

From the root of the repository, after `collect.jl`:

    julia --project=experiments experiments/fai_benchmarks/tt_anisotropy_global/plot_results.jl

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
results_folder = joinpath(base_folder, "results")
plots_dir = joinpath(base_folder, "plots")
mkpath(plots_dir)

results_file = joinpath(results_folder, "collected_results.jld2")
@info "Loading dataframe from $results_file"
df = load(results_file, "df")

if hasproperty(df, :converged)
    not_converged = df[coalesce.(df.converged, true) .== false, :]
    for r in eachrow(not_converged)
        @warn "Not converged (hit maxiter)" N=r.N F=r.F key=r.key global_iters=r.global_iters
    end
end

for N in sort(unique(df.N))
    @info "Generating plots for N = $N"
    dfN = df[df.N .== N, :]

    plot_speedup_vs_F(dfN, N, :solve_time, plots_dir)
    plot_speedup_vs_F(dfN, N, :total_time, plots_dir)

    plot_metric_vs_F(
        dfN,
        :global_iters;
        title="Global CG iterations (N=$N)",
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

    plot_sinc_plateau(dfN, N, :global_iters, "Global CG iterations", joinpath(plots_dir, "sinc_plateau_N_$(N).png"))
end

crossover_summary(df, plots_dir; iters_col=:global_iters)
@info "Plots saved to $plots_dir"
