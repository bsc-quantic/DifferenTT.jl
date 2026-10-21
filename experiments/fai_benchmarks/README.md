# FAI benchmarks

We benchmark here the factorized approximate inverse (FAI) preconditioner against an unpreconditioned baseline and,
where it applies, against the Sinc preconditioner of Khoromskij (2009), on elliptic systems whose discretizations are
symmetric positive definite (SPD). Each benchmark solves the same sweep of tasks, one per combination of system size,
preconditioner and parameters, and records the build time of the preconditioner, the solve time, the number of
iterations and the bond dimensions. The FAI is built with μ = 10⁷ in every worker, and its bond dimension χ_G sweeps from 2 to 10 in the Chorin and cavity benchmarks. The benchmarks come in pairs: the
`ResidualALS` variant preconditions the local problems of each site (Sec. 4 of the paper), and the `_globalcg` or
`_global` variant preconditions a global TT-CG solve of the whole system.

| Folder | System | Sweep |
|---|---|---|
| `tt_anisotropy`, `tt_anisotropy_global` | Anisotropic diffusion −∇·(K∇u) = f on (0,1)ᴺ with K = (1−ρ)I + ρ𝟏𝟏ᵀ, in the non-quantized TT format (`common/anisotropy_problem.jl`) | N ∈ {5, 10}, anisotropy floor F ∈ {1, 2, 4, 10, 30, 100}, FAI and Sinc configurations, 15 nodes per dimension |
| `tt_poisson`, `tt_poisson_global` | Poisson equation on (0,1)ᴺ, in the non-quantized TT format | N ∈ {5, 10, 15, 20, 25}, 15 nodes per dimension, FAI and Sinc configurations |
| `chorin_pipeline`, `chorin_pipeline_globalcg` | Pressure Poisson systems that appear along the 2D Chorin projection of a flow past a cylinder in a channel [0,2]×[0,1] at Re = 3000, in the QTT format | 2⁸×2⁸ grid (100 steps) and 2¹⁰×2¹⁰ grid (50 steps), FAI with a bond dimension χ_G from 2 to 10 |
| `liddriven_pipeline`, `liddriven_pipeline_globalcg` | Stream function systems −Δψ = ω along the vorticity–streamfunction evolution of the lid-driven cavity, in the QTT format | 2¹¹×2¹¹ grid for the local variant (`CAVITY_LIDS=2` selects the doubly-driven cavity), 2⁸×2⁸ for the global one |

The problem sizes above are the ones set in each `worker.jl`; they are the starting point, not a recommendation. The
Chorin benchmarks read the cylinder mask from `masks.jld2` in their own folder, which `generate_masks.jl` writes with
Tensor Cross Interpolation (`maskplots/` holds the pictures of the masks). The masks and the collected data tracked here
were produced before the library fixes listed in `experiments/README.md`, so they are not comparable with new runs.

## How to run

Every folder contains `worker.jl`, `submit.sh`, `collect.jl` and `plot_results.jl` (see the sequence in
`experiments/README.md`). From the root of the repository:

```bash
bash experiments/fai_benchmarks/chorin_pipeline/submit.sh nocond   # job array + collector, on the cluster
```

The job array of each folder has 10 to 132 tasks of 20 CPUs and up to 36 h (the Chorin and cavity benchmarks request
exclusive nodes); the collector takes 4 CPUs (1 for the `_globalcg` variants of the Chorin and cavity benchmarks) and 10 minutes. The workers raise Julia's
GC thresholds with a memory ballast, controlled by `GC_BALLAST_GB` (`common/gc_ballast.jl`). Smoke mode: `TODO(5a-2)`.

## Outputs

Each worker writes `tasks/task_NNN.jld2` under `results/` or, for the `nocond` run, `results_nocond/`. The collector
merges them into `collected_results.jld2` in the same folder, and `plot_results.jl` writes its figures to `plots/`
(heatmaps of speedup, Pareto fronts of build against solve time, per-step and per-site profiles of iterations and
condition numbers, and scaling with the system size).
