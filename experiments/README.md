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

We launch the scripts with `--project=experiments`; they build every path from their own location, so the working directory
does not matter. The job scripts (`submit.sh`, `.slurm`), in contrast, are submitted from the root of the repository. Plots is a dependency of the
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

## Smoke mode

A sweep that fails after hours is expensive, and a script that no longer runs after a change of the library is only found when it
is submitted. `DIFFERENTT_SMOKE=1` therefore makes a worker, a collector or a plotter run on a tiny problem (a few tasks on a small
grid) and write to a temporary folder (`DIFFERENTT_SMOKE_DIR`, or a new one if it is unset), so that it takes minutes and never
touches the committed results. `smoke.jl` runs every non-archived script this way, one `julia --project=experiments` process at a
time with a timeout of 5 minutes: for each benchmark the generator of its inputs (if any), the worker on the baseline and on one FAI
task, the collector and the plotter. It prints one line per step and, at the end, a table with the result, the time and the last
error line of every step.

```bash
julia --project=experiments experiments/smoke.jl                    # every benchmark
julia --project=experiments experiments/smoke.jl tt_poisson         # the named benchmarks only
DIFFERENTT_SMOKE=1 julia --project=experiments experiments/fai_benchmarks/tt_poisson/worker.jl   # one script
```

Smoke mode only checks that a script runs: its numbers say nothing about the problem. A script that still uses a name that the
library has renamed or removed fails here, and the table shows which one. The GitHub workflow `ExperimentsSmoke.yml` runs
`smoke.jl` every week and on demand, with one parallel job per benchmark; no other check depends on it.

The SLURM scripts set the job geometry, and read the cluster settings and the GC ballast from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `SLURM_ACCOUNT` | `bsc21` | Account of the jobs |
| `SLURM_QOS` | `ngp_bsccase` | Quality of service |
| `JULIA_MODULE` | `julia/1.12.0` | Module that provides Julia |
| `GC_BALLAST_GB` | `7` | GC ballast of the workers (see `common/gc_ballast.jl`); 0 disables it |

```bash
SLURM_ACCOUNT=<project> SLURM_QOS=<qos> bash experiments/fai_benchmarks/tt_poisson/submit.sh nocond
```

The static `#SBATCH` lines of the `.slurm` files of `validation/tunnel2d*` cannot expand variables: we override them on the
command line (`sbatch --account=<project> --qos=<qos> experiments/validation/tunnel2d/tunnel2d.slurm`), and they read `JULIA_MODULE`.

## Data

The `collected_results.jld2` files that are tracked in this repository were produced with the library before the fixes of
the FAI orientation, the `ResidualALS` cutoff convention and the period of periodic grids (see `NEWS.md`). They are not
comparable with new runs, and must not be mixed with them in one collection or plot. The data policy (what is committed and
where large data goes) is `TODO(5a-4)`.
