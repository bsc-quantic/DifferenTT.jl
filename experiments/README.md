# Experiments

This folder holds the numerical experiments behind the paper "Factorized approximate inverse preconditioners in the
tensor train format" and the benchmarks and validation runs built on the same library. We organize it by purpose, so that
a reader who wants to reproduce a figure, benchmark a solver or check a physical setup knows where to start. The didactic
introductions to the library are not here: they live in `examples/` at the root of the repository.

## Environment

All scripts run in the environment of this folder, which pins every package in `Manifest.toml` and takes DifferenTT from
the checkout through `[sources]`. We instantiate it once from the root of the repository:

```bash
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
```

Scripts are launched from the root of the repository, with `--project=experiments`. Plots is a dependency of the
environment, so that the plotting extension of DifferenTT (`visualize_*`) loads. The environment contains neither
`Revise`, `MKL` nor any GPU package: the two exploratory scripts of `validation/` that load the first two are listed in
the README of that folder.

## Layout

| Folder | What it computes | Paper | Resources of the job scripts | Outputs |
|---|---|---|---|---|
| `common/` | Helpers shared by the scripts: the anisotropic-diffusion problem, plotting utilities and the GC ballast | – | – | – |
| `paper/` | The scripts that produce the figures and tables of the paper. Currently `anisotropy_floor` and `linsolve_spd`; the other sets are added by the port of the paper scripts | Table 1 (`tab:anisotropy_floor`), Fig. 3 (`fig:spd_nneg`); more to come | `linsolve_spd`: 20 CPUs, up to 36 h | `discrete_floor.{jld2,md}`; `linsolve_spd/results/` |
| `fai_benchmarks/` | Benchmarks of the FAI, Sinc and unpreconditioned solvers on Poisson-type systems: anisotropic diffusion, N-dimensional Poisson, the pressure systems of the Chorin projection and of the lid-driven cavity. Each has a local (`ResidualALS`) and a global (`GlobalCG`) variant | Sec. 5 uses the stats variants of these benchmarks (see `paper/`) | Job array of 10–132 tasks, 20 CPUs and up to 36 h each; a collector job of 4 CPUs and 10 minutes | `results*/tasks/*.jld2`, `results*/collected_results.jld2`, `plots/` |
| `validation/` | Checks of the Navier–Stokes solvers against references and exploratory flows past obstacles | Sec. 5.2 describes the physical setup of `tunnel2d` | `liddriven_cavity`: 16 CPUs, up to 36 h; `tunnel2d`: 8 CPUs, 6 h | `results*/`, `circle_data/` |
| `archive/` | Early demonstrations, superseded by the tutorials in `examples/`. Unmaintained | – | – | – |

`tt_poisson/poissonNd.jl` is the N-dimensional Poisson demonstration from which `examples/03` derives. It stays at this
level until the update of its FAI call has been made, and then moves to `archive/`.

## How to run

Every benchmark folder follows the same sequence: a `worker.jl` solves one task of a sweep, selected by its task number;
`submit.sh` submits the sweep as a SLURM job array followed by a collector job; `collect.jl` gathers the task files into
`collected_results.jld2`; and `plot_results.jl` draws the figures from it. A task is one combination of problem size,
preconditioner and parameters. Where a folder supports it (the local benchmarks), passing `nocond` to the submit script, the worker and the collector selects the run that does not
estimate local condition numbers: it is the one whose timings we use, because the estimates are expensive (the plotter
reads both runs).

```bash
bash experiments/fai_benchmarks/tt_poisson/submit.sh nocond                       # SLURM (cluster)
julia --project=experiments experiments/fai_benchmarks/tt_poisson/worker.jl 3 nocond   # one task, locally
```

Smoke mode, which runs every script on tiny sizes in a temporary folder, does not exist yet: `TODO(5a-2)`.

The SLURM scripts set the job geometry and read the GC ballast (`GC_BALLAST_GB`, see `common/gc_ballast.jl`) from the
environment. The account, quality of service and Julia module are still fixed in the scripts: `TODO(5a-2)`.

## Data

The `collected_results.jld2` files that are tracked in this repository were produced with the library before the fixes of
the FAI orientation, the `ResidualALS` cutoff convention and the period of periodic grids (see `NEWS.md`). They are not
comparable with new runs, and must not be mixed with them in one collection or plot. The data policy (what is committed and
where large data goes) is `TODO(5a-4)`.
