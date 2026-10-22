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
`Revise`, `MKL` nor any GPU package. The two exploratory scripts of `validation/tunnel2d*` load `MKL` if it is installed in
your own environment (a faster BLAS on Intel CPUs) and never load `Revise`.

## Layout

| Folder | What it computes | Paper | Resources of the job scripts | Outputs |
|---|---|---|---|---|
| `common/` | Helpers shared by the scripts: the anisotropic-diffusion problem, plotting utilities, the GC ballast, the repeat harness and statistics of the paper's `*_stats` runs, and the check that a Chorin mask belongs to the grid of the run | – | – | – |
| `paper/` | The scripts that produce the figures and tables of the paper: `anisotropy_floor`, `linsolve_spd`, `tt_anisotropy_stats` and `chorin_pipeline_stats`. The README of the folder maps every label of the paper to its scripts | Table 1 (`tab:anisotropy_floor`), Fig. 3 (`fig:spd_nneg`), Figs. 9-11, Table 2 and Figs. 12-13 | `linsolve_spd`: 20 CPUs, up to 36 h; the `*_stats` job arrays: 20 CPUs and up to 2 days per job | `discrete_floor.{jld2,md}`; `linsolve_spd/results/`; `*_stats/results_nocond/`, `*_stats/plots_paper/` |
| `fai_benchmarks/` | Benchmarks of the FAI, Sinc and unpreconditioned solvers on Poisson-type systems: anisotropic diffusion, N-dimensional Poisson, the pressure systems of the Chorin projection and of the lid-driven cavity. Each has a local (`ResidualALS`) and a global (`GlobalCG`) variant | Sec. 5 uses the statistics variants of these benchmarks, which are in `paper/` | Job array of 10–132 tasks, 20 CPUs and up to 36 h each; a collector job of 4 CPUs and 10 minutes | `results*/tasks/*.jld2`, `results*/collected_results.jld2`, `plots/` |
| `validation/` | Checks of the Navier–Stokes solvers against references and exploratory flows past obstacles | Sec. 5.2 describes the physical setup of `tunnel2d` | `liddriven_cavity`: 16 CPUs, up to 36 h; `tunnel2d`: 8 CPUs, 6 h | `results*/`, `circle_data/` |
| `archive/` | Early demonstrations, superseded by the tutorials in `examples/`. Unmaintained | – | – | – |

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

### Data from before this update are not comparable

The `collected_results.jld2` files that are tracked in this repository, and the Chorin masks, were produced with the library
before the fixes of the FAI orientation (the FAI solved the transposed FSAI problem), the `ResidualALS` cutoff convention (its
Frobenius floor used `cutoff²`, now `cutoff` as in `ALS` and `AMEn`) and the period of periodic grids (the plain `Grid` has the
period `L + h`; the Chorin and tunnel scripts now use `Grid(arch, L; periodic=periodic_axes(bcs))`, whose period is `L`); see
`NEWS.md`. They are not comparable with new runs and must not be mixed with them in one collection or plot. The READMEs of the
folders that show results say so.

The Chorin workers stop with an `ArgumentError` if their mask was sampled on another grid (`common/mask_grid.jl`). The masks
of `fai_benchmarks/chorin_pipeline` are regenerated, and `fai_benchmarks/chorin_pipeline_globalcg` carries a copy of them (its
generator is the same script, so the files are identical). To regenerate either set, which also rewrites its `maskplots/`, run,
from the root of the repository (here for the second folder):

```bash
julia --project=experiments experiments/fai_benchmarks/chorin_pipeline_globalcg/generate_masks.jl
```

and check the printed `ranks_mask` and `error_mask` (tolerance 1e-10, bond dimension at most 50) before committing both files.

### What we commit and what we do not

| Committed | Not committed |
|---|---|
| `collected_results.jld2` of a benchmark (a few hundred kB; about 1 MB at most) | The task files `tasks/task_NNN.jld2` and the logs (`.gitignore`) |
| The input masks `masks.jld2` (about 0.8 MB each) and the pictures `maskplots/*.png` | Any other `.jld2`, `.png`, `.svg` or `.pdf` that a script writes (`.gitignore`), such as `circle_data/` of `tunnel2d` and the `test_*.png` of `tunnel2dsteps` |
| Small tables that a script prints to Markdown (`summary.md`, `ghia_comparison.txt`, `discrete_floor.md`) and `paper/anisotropy_floor/discrete_floor.jld2` (Table 1, D1 era) | SLURM output (`output.out`, `error.err`, `slurm-*.out`) and snapshots of long runs (`finaldata/`) |

A file above about 1 MB does not go into the repository: it is kept outside it, on the cluster or in an archive, and the
README of the benchmark that produced it records where and which commit made it. The data of the rerun of the paper
experiments (session R) is the first case: `paper/README.md` carries a `TODO(session R)` for the commit and the location
of that data, which is decided when the rerun has been done.
