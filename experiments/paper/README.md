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
| Figs. 9-11: MALS iterations, solve time and iterations per site, anisotropic diffusion (`fig:mals_iters`, `fig:mals_solve`, `fig:mals_site_iters`) | `tt_anisotropy_stats/` | `worker.jl` | `collect.jl` | `plot_paper.jl` |
| Table 2 and Figs. 12-13: Chorin pipeline, MALS parameters, iterations and cumulative speedup (`tab:chorin_mals_params`, `fig:chorin_mals_iterations`, `fig:chorin_mals_cumulative_speedup`) | `chorin_pipeline_stats/` | `worker.jl` | `collect.jl` | `plot_paper.jl` |

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
to add repeats afterwards (`BENCH_REPEAT_OFFSET`). The plotters read the `nocond` runs (`results_nocond/`); the runs
with local condition numbers are much slower and are not used by the paper's figures.

## The data of the current figures, and what changed since

The figures of the current version of the paper were produced by the code of the tag `paper-ttfai-data` (commit `6850540`)
with the scripts of the paper repository. **New runs of the scripts of this folder are not comparable with those data.**
Three fixes of the library change the numbers (all in `NEWS.md`): the FAI used to solve the transposed FSAI problem (and
`linsolve_preconditioner` returned the transpose of the inverse for a nonsymmetric operator); the Frobenius floor of the
residual-based truncation of `ResidualALS` now uses `cutoff`, where it used `cutoff²`, so that bond dimensions and
solutions at `cutoff > 0` differ (`tt_anisotropy_stats` uses `cutoff = 1e-13`; the Chorin worker uses `cutoff = 0`, which
the change does not affect); and the Chorin and `linsolve_spd` workers now build the grid with `periodic = periodic_axes(bcs)`,
so that the period in y is the length of the box and not the length plus one cell (the old data used the plain `Grid`).
Besides, the call of the FAI is now `fai_preconditioner` and the workers no longer pass `compute_cond`; both give the same
preconditioner (we checked that the matrices agree exactly on a small case). Do not mix collected results of the two
versions in one collect or plot.

The old data stay in the paper repository. Three files in this repository are the exception:

- `anisotropy_floor/discrete_floor.jld2` and `.md`: the data of Table 1 (the discrete floor and the ranks of the exact
  solution), computed in the D1 era. We commit them because the plotters of `../fai_benchmarks/tt_anisotropy` read them.
  The floor and the ranks depend on the discretization of the operator only, not on the FAI or on `ResidualALS`, but we
  regenerate them in session R with the others.
- `linsolve_spd/plots/summary.md`: the summary of the final run of the D1 era.

## What the maintainer has to do before a run (TODO)

1. **Regenerate the masks.** The committed `../fai_benchmarks/chorin_pipeline/masks.jld2` and `maskplots/` were sampled on
   the plain grid and carry no grid metadata, so the Chorin worker and `linsolve_spd` stop at once with an `ArgumentError`
   ("does not belong to the grid of this run"). That is intended: use the generator, check the printed `ranks_mask` and
   `error_mask` (tolerance 1e-10, bond dimension at most 50), and commit the two files:

   ```bash
   julia --project=experiments experiments/fai_benchmarks/chorin_pipeline/generate_masks.jl
   ```

   The generator for `chorin_pipeline_globalcg` is handled together with that benchmark. A missing mask file is not an
   error: the Chorin worker then warns and runs without the obstacle, so check the log for "mask not found".
2. TODO(session R): after the rerun on the release candidate, record here the commit that produced the new data and where
   they are kept (the data policy is in `experiments/README.md`), and regenerate Table 1 with the others.
