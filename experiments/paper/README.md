# Scripts of the paper

This folder holds the scripts that produce the figures and tables of the paper "Factorized approximate inverse
preconditioners in the tensor train format", from the problem to the figure. We ported them from the paper repository (the
workers and `plot_paper.jl`) and from the cluster mirror (`collect.jl` and the shared helpers) to the current library; the
setups, that is the problems, parameters, sizes, seeds, solver and preconditioner settings and task lists, are those of the
paper repository and we did not change them. The benchmarks that the paper does not use are in `../fai_benchmarks/`.

## Which script draws which figure

| Paper (label) | Folder | Worker | Collector | Plotter |
|---|---|---|---|---|
| Fig. 3: negative eigenvalues of the truncated approximate inverses (`fig:spd_nneg`) | `linsolve_spd/` | `worker.jl` | – (one results file) | `plot_paper.jl` (`plot_results.jl` is a quick look at the same data) |
| Table 1: anisotropy floor and solution ranks (`tab:anisotropy_floor`) | `anisotropy_floor/` | `discrete_floor.jl` (computes the table) | – | – |
| Table 2 and Figs. 9-10: anisotropic diffusion, MALS parameters, iterations and iterations per site (`tab:anisotropy_params`, `fig:mals_iters`, `fig:mals_site_iters`) | `tt_anisotropy_stats/` | `worker.jl` | `collect.jl` | `plot_paper.jl` |
| Table 3 and Figs. 11-12: Chorin pipeline, MALS parameters, iterations and cumulative speedup (`tab:chorin_mals_params`, `fig:chorin_mals_iterations`, `fig:chorin_mals_cumulative_speedup`) | `chorin_pipeline_stats/` | `worker.jl` | `collect.jl` | `plot_paper.jl` |

The statistics benchmarks run every task `BENCH_N_REPEATS` times and write one file per task and repeat; the collector
turns them into one table and the plotter draws the spread over the repeats. Their shared helpers are in
`../common/`: `stats_harness.jl` (the repeats and the layout of the jobs), `stats_utils.jl` (collection, paired speedups and
the sanity checks of the repeats). The Chorin worker
and `linsolve_spd` read the masks of `../fai_benchmarks/chorin_pipeline/masks.jld2`, and the anisotropy worker the problem
of `../common/anisotropy_problem.jl`.

## How to run

All the commands start from the root of the repository, after instantiating the environment of `experiments/`. Smoke mode
(`DIFFERENTT_SMOKE=1`) shrinks every script to a tiny problem and writes to a temporary folder; `experiments/smoke.jl` runs
the whole sequence of a folder:

```bash
julia --project=experiments experiments/smoke.jl tt_anisotropy_stats chorin_pipeline_stats linsolve_spd anisotropy_floor
```

The runs of the paper are submitted to SLURM; the job scripts read `SLURM_ACCOUNT`, `SLURM_QOS` and `JULIA_MODULE` (see
`experiments/README.md`), and we run the collector and the plotter locally:

| Folder | Submit | Collect and plot | Resources requested by the job script |
|---|---|---|---|
| `linsolve_spd/` | `bash experiments/paper/linsolve_spd/submit.sh` | `julia --project=experiments experiments/paper/linsolve_spd/plot_paper.jl` | one job of 20 CPUs for up to 36 h |
| `anisotropy_floor/` | – (we run it locally: `julia --project=experiments experiments/paper/anisotropy_floor/discrete_floor.jl`) | – | – |
| `tt_anisotropy_stats/` | `bash experiments/paper/tt_anisotropy_stats/submit.sh nocond` | `julia --project=experiments experiments/paper/tt_anisotropy_stats/collect.jl nocond`, then `plot_paper.jl` | an array of 12 × `BENCH_N_REPEATS` blocks (120 for 10 repeats), 20 CPUs and up to 2 days each; a collector job of 4 CPUs and 30 minutes |
| `chorin_pipeline_stats/` | `bash experiments/paper/chorin_pipeline_stats/submit.sh nocond` | `julia --project=experiments experiments/paper/chorin_pipeline_stats/collect.jl nocond`, then `plot_paper.jl` | for 10 repeats, 10 block jobs for the 8×8 grid (up to 2 days) and 100 unit jobs for the 10×10 grid (up to 1 day), 20 CPUs each; a collector job of 4 CPUs and 30 minutes |

The comments of each `submit.sh` give the time of each block or unit that we measured in the run of October 2026, and how
to add repeats afterwards (`BENCH_REPEAT_OFFSET`). The plotters read the `nocond` runs of the default μ rule
(`results_nocond_mugersh/`, see below); the runs with local condition numbers are much slower and are not used by the
paper's figures.

## The penalty μ of the FAI build (`FAI_MU_RULE`)

The FAI solves Eq. (21) with the operator P_L (I ⊗ A) P_L + μ (I - P_L), whose condition number is
max(μ, λmax(A)) / min(μ, λmin(A)), so μ only needs to sit at the top of the spectrum of A. The paper sets μ = ‖A‖_∞, the
Gershgorin bound of λmax(A) (Eqs. (32) and (37)). The earlier drafts used a fixed μ = 10⁷, which is close to λmax(A) on
the 2¹⁰ × 2¹⁰ Chorin grid but 130 to 1300 times above it in the anisotropy benchmark. Both statistics workers
(`tt_anisotropy_stats/`, `chorin_pipeline_stats/`) take two environment variables:

| Variable | Values | Effect |
|---|---|---|
| `FAI_MU_RULE` | `gershgorin` (default) or `fixed` | `gershgorin`: μ = ‖A‖_∞, the Gershgorin bound of λmax(A), as in the paper, computed per problem from the stencil coefficients of the operator the FAI is built on, Σ_k \|c_k\| per axis times h⁻² (or h⁻¹ for the mixed terms), which no boundary closure can exceed (`../common/mu_rules.jl`). `fixed`: the μ = 10⁷ of the task list of the worker, as in the earlier drafts |
| `FAI_MU_SCALE` | a positive number, default 1.0 | multiplies the value of the rule (not allowed with `fixed`) |

With the rule, μ = N (4 + ρ (N - 1)) / h² for the FAI on L of the anisotropy benchmark (4N / h² for the FAI on L₀; 7840 to
7.8 × 10⁴ at the paper sizes) and 6.5016 (h_x⁻² + h_y⁻²) for the Chorin pressure operator (5.3 × 10⁵ on 2⁸ × 2⁸, 8.5 × 10⁶
on 2¹⁰ × 2¹⁰). `../common/mu_rules_check.jl` checks that Σ_k |c_k| bounds the 1D operators with every closure, and
λmax(A) ≤ ‖A‖_∞ ≤ μ ≤ 1.5 λmax(A) on small assembled operators (`experiments/smoke.jl` runs
it as `mu_rules_check`). A rule-based run writes under `results_mugersh/` or `results_nocond_mugersh/`
(`_mugersh_x<scale>` with a scale), never into the fixed-μ folders; its keys name the rule (`fai_mu_gersh_gdim_χ`,
`fai_L0_mu_gersh_gdim_χ`, `mu_gersh_gdim_χ`) and the result files store the numeric μ (`"mu"`), the rule (`"mu_rule"`) and
the scale (`"mu_scale"`). The collectors and the plotters read the same variables, so by default they read that folder,
and the plotters write `plots_paper_mugersh/` and label the rule μ = ‖A‖_∞. To run, collect and plot with the fixed μ,
which writes under `results/` or `results_nocond/` and `plots_paper/`, set the variable for every step:

```bash
FAI_MU_RULE=fixed bash experiments/paper/tt_anisotropy_stats/submit.sh nocond
FAI_MU_RULE=fixed julia --project=experiments experiments/paper/tt_anisotropy_stats/collect.jl nocond
FAI_MU_RULE=fixed julia --project=experiments experiments/paper/tt_anisotropy_stats/plot_paper.jl
```

The job scripts pass the folder to the workers and the collector (`TT_ANISOTROPY_STATS_RESULTS_DIR`,
`CHORIN_STATS_RESULTS_DIR`). Every FAI result also stores the iterations of the local solves of the build
(`precond_build_total_iters`, `precond_build_iters_per_site`, summed over the sweeps, and `precond_build_nsweeps`), where a
change of the conditioning of the build shows up; the outer solve is not expected to change much.

## The data of the current figures, and what changed since

The figures and tables of the paper are drawn from the rerun of the scripts of this folder at commit `3945c32` (tag
`paper-ttfai-v0.1-data`), on MareNostrum 5 in October 2026, with `FAI_MU_RULE=gershgorin` (now the default) for the two
statistics runs. Following the data policy of `experiments/README.md`, those data are archived on Zenodo, not in this
repository: [doi:10.5281/zenodo.23663326](https://doi.org/10.5281/zenodo.23663326). The archive holds `linsolve_spd.jld2`, the
`collected_results.jld2` of `tt_anisotropy_stats` and `chorin_pipeline_stats`, `discrete_floor.{jld2,md}`, the logs, the
`Project.toml` and `Manifest.toml` of `experiments/`, and a README that says where to copy each file to redraw the figures.

Earlier drafts of the paper used the data of the tag `paper-ttfai-data` (commit `6850540`), produced with the scripts of
the paper repository. **New runs of the scripts of this folder are not comparable with those data.**
Three fixes of the library change the numbers (all in `NEWS.md`): the FAI used to solve the transposed FSAI problem (and
`linsolve_preconditioner` returned the transpose of the inverse for a nonsymmetric operator); the Frobenius floor of the
residual-based truncation of `ResidualALS` now uses `cutoff`, where it used `cutoff²`, so that bond dimensions and
solutions at `cutoff > 0` differ (`tt_anisotropy_stats` uses `cutoff = 1e-13`; the Chorin worker uses `cutoff = 0`, which
the change does not affect); and the Chorin and `linsolve_spd` workers now build the grid with `periodic = periodic_axes(bcs)`,
so that the period in y is the length of the box (see `NEWS.md`), which changes the numbers.
Besides, the call of the FAI is now `fai_preconditioner` and the workers no longer pass `compute_cond`; both give the same
preconditioner (we checked that the matrices agree exactly on a small case). Do not mix collected results of the two
versions in one collect or plot.

The old data stay in the paper repository. Of the data of the paper, this repository keeps only:

- `anisotropy_floor/discrete_floor.jld2` and `.md`: the data of Table 1 (the discrete floor and the ranks of the exact
  solution), regenerated in the rerun (the same files as in the Zenodo archive). We commit them because the plotters of
  `../fai_benchmarks/tt_anisotropy` read them.
- `linsolve_spd/plots/summary.md`: the summary of the final run of the D1 era.

## What the maintainer has to do before a run (TODO)

1. **Masks.** The masks of `../fai_benchmarks/chorin_pipeline/masks.jld2` were regenerated on the periodic grid (commit
   `6386bd9`) and carry the grid metadata, so the Chorin worker and `linsolve_spd` accept them. If you change the box or the
   grid, regenerate them with the generator, check the printed `ranks_mask` and `error_mask` (tolerance 1e-10, bond dimension
   at most 50), and commit the two files:

   ```bash
   julia --project=experiments experiments/fai_benchmarks/chorin_pipeline/generate_masks.jl
   ```

   The masks of `chorin_pipeline_globalcg` are a separate file, not read by `paper/`. A missing mask file is not an error:
   the Chorin worker then warns and runs without the obstacle, so check the log for "mask not found".
2. **Data.** A new run of the paper is a new version of the data: archive it as a new version of the Zenodo record above,
   and record here the commit and the tag that produced it and its DOI (the data policy is in `experiments/README.md`).
