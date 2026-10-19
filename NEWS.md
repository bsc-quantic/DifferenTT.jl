# DifferenTT.jl release notes

## v0.1.0 (unreleased)

The first public release. Version 0.0.1 is the state of the library that produced the data of the FAI paper
(tag `paper-ttfai-data`, commit `6850540`); the changes below are relative to it.

### Added

- `rng::AbstractRNG = Random.default_rng()` keyword on every randomized routine: `tt_mul` (MPS-MPS), `_rsi_tt_mul`,
  `elementwise_function`, `sketch`, `positive_sketch`, `suffix_sketches`, `fai_preconditioner`, `fai_dg`,
  `linsolve_preconditioner`, `TensorCrossInterpolation.optfirstpivot` and `_tci_random_pivot`;
  `RandomSPSDexpansion(; rng)` and `RandomLTexpansion(; rng)` store it. Without `rng` the results are unchanged.
- Keyword form `fai_preconditioner(A; μ, solver, totalmaxdim=5, D_cutoff=1e-15, D_maxdim=typemax(Int),
  D_oversample=10, preconditioning=false, residual_tol=nothing, compute_residual=false, rng, kwargs...)`, with the
  defaults the positional form used. `fai_dg` stays internal.
- `fai_dg(...; compute_residual=false)`: see Changed.
- `GlobalUpdaterArgs(; converged_store)`: a `Ref{Bool}` that receives whether the global Krylov solve converged.
- `enable_debug_timings!()` and `disable_debug_timings!()` (they were exported but undefined).
- `public` declarations for the documented non-exported API (tensor-train utilities, operator and physics building
  blocks, Krylov algorithm types and statistics, the local updaters and the `ExpandedALS` expansions
  (`AbstractUpdater`, `AbstractALSExpansion`, `NullExpansion`, `RandomSPSDexpansion`, `RandomLTexpansion`,
  `ResidualExpansion`), the observer helpers `bind_observer` and `default_differenTT_checkdone`, and the renamed
  functions below).
- `THIRD_PARTY_NOTICES.md`, with the KrylovKit.jl and TT-Toolbox notices and the provenance of the RSI code.
- Plotting moved to the package extension `DifferenTTPlotsExt`: `visualize_scalar`, `visualize_vector` and
  `visualize_vorticity` need `using Plots`, and print a hint otherwise.

### Changed

- Since the paper data were produced: `spd_tol` in `LocalUpdaterArgs` without `ishermitian && isposdef` throws
  `ArgumentError`; a non-SPD local CG block no longer warns (only `spd_tol`, which throws `NotSPDError`, and
  `spectrum_store` detect it); `preconditioner_sites` outside `1:length(b)` throws `ArgumentError`.
- User input is validated with `ArgumentError` or `DimensionMismatch` instead of `@assert`, so the exception types of
  invalid architectures, array sets, `tt_mul` operands, initial pivots, sinc and exact inverse-Laplacian inputs and
  field constructors change from `AssertionError`.
- The library prints nothing by default. `DirichletBC` no longer logs at construction (the note moved to its
  docstring). The `ChorinCache` regularization flag, `linsolve_preconditioner` and the `ResidualObserver` early stop are
  `@debug`. The CFL and Kolmogorov checks and the `vmax = 1` fallback of `INSVorticityExperiment` keep their
  `@info`/`@warn` messages, since they tell how an experiment will behave for its parameters.
- `fai_dg`, and through it `fai_preconditioner`, compute and log the residual of the projected FAI system only with
  `compute_residual = true` (the default is `false`; it used to be always computed and logged).
- The global solvers (`GlobalCG`, `GlobalGMRES`) warn when they stop without converging, and throw `ArgumentError`
  for a `numiter_store` or `converged_store` that does not support `store[] = value`.
- `ResidualALS.residual_tol` and `residual_damp` are `Float64`.
- `tt_mul_fallbacks` counts with a `Threads.Atomic{Int}`.
- `elementwise_function` builds its suffix sketches once per call, as `tt_mul` does (one sketch index and one
  right-to-left pass, `suffix_sketches(...; positive)`), instead of re-sketching every suffix at every bond. It is
  faster, and with a fixed seed its results (including the `D` of `fai_preconditioner`) differ from v0.0.1 at the
  sketch-noise level.
- `Project.toml`: version `0.1.0-DEV`; `DataFrames`, `LaTeXStrings` and `Statistics` removed; `Plots` is a weak
  dependency.

### Breaking

- Since the paper data were produced: `LocalUpdaterArgs` no longer accepts `compute_cond`; attach `cond_store` or
  `pc_cond_store` (or `spd_tol`) to get the local condition numbers.
- **Type piracy removed.** DifferenTT no longer adds methods to functions of Base, Random, ITensors, ITensorMPS or
  KrylovKit for argument types it does not own. The affected functions are renamed:

  | Old (pirated) | New (owned) |
  |---|---|
  | `siteinds(dims::Vector{Int}; kw...)` | `DifferenTT.site_indices(dims; kw...)` |
  | `linkinds(dims::Vector{Int}; kw...)` | `DifferenTT.link_indices(dims; kw...)` |
  | `ITensorMPS.replace_siteinds!(::MPO, sites)`, `DifferenTT.replace_siteinds!(::MPO, sites)`, `DifferenTT.replace_siteinds(::MPO, sites)` | `DifferenTT.replace_mpo_siteinds!(mpo, sites)` (returns the MPO), `DifferenTT.replace_mpo_siteinds(mpo, sites)` |
  | `transpose(::MPO)` | `DifferenTT.transpose_mpo(mpo)` |
  | `reverse(::MPO)` | `DifferenTT.reverse_mpo(mpo)` |
  | `Random.rand!([rng,] ::ITensor)` | `DifferenTT.uniform_itensor([rng,] inds...)` |
  | `Index(pivs::Vector{Vector{<:Pair}}; tags)` (index space `Vector{Vector{<:Pair}}`) | `DifferenTT.pivot_index(pivs; tags)` (index space `DifferenTT.PivotSpace`; read the pivots with `DifferenTT.index_pivots(i)`) |
  | `ITensors.swapprime(::OneITensor, …)` | internal `DifferenTT._swapprime` |
  | `ITensors.contract(P::AbstractProjMPO)` (one argument) | `DifferenTT.dense_projected_operator(P)` |
  | `KrylovKit.linsolve(op, b, x₀, M⁻¹, [alg::CG/GMRES], a₀, a₁; kw...)` | `DifferenTT.preconditioned_linsolve(op, b, x₀, M⁻¹, [alg], a₀, a₁; kw...)` |

  `siteinds(::Architecture)` and the `KrylovKit.linsolve` methods for `TTCG` and `TTGMRES` are unchanged.
- `initialize_reducedprecondlinearproblem(::MPO, ::Vector{MPS}, …)` (two methods whose body was
  `error("Not implemented yet")`) and the unused `onedim_neumann_boundary_correction` are removed.
- `VectorField(components)` needs a non-empty tuple of one concrete `ScalarField{N}` type, with `N` components; a
  tuple of the wrong length throws `DimensionMismatch`.

### Fixed

- TT-CG on shifted systems `(a₀I + a₁A)x = b` with `a₀ ≠ 0`: the step denominator used `a₁⟨p,p⟩` instead of `a₀⟨p,p⟩`,
  so the iteration diverged. `GlobalCG`, which solves the unshifted system, was not affected.
- Preconditioned `GlobalGMRES` threw a site-index mismatch in the Arnoldi step.
- Preconditioned GMRES (`preconditioned_linsolve` with `GMRES`, and TT-GMRES with `M⁻¹`) that stopped without
  converging returned the norm of the preconditioned residual `M⁻¹r` as `normres`; the dense version also returned a
  residual `r` that was not updated after the last restart. Both now return the residual of the unpreconditioned
  system and its norm.
- `ResidualExpansion` throws `ArgumentError` instead of `ErrorException` when the sites of `A` and `b` do not match.
- `DifferentialOpsScalar(grid, field, n_eyepad)` threw on a missing grid field. The padded operator is scaled by the
  spacing of the coarse grid of `2^(n - p)` nodes over the same box in QTT, and by the original spacing in TT.
- The exports `sited_dim` (now `site_dim`) and `enable_debug_timings!` named functions that did not exist;
  `enable_debug_timings!` and `disable_debug_timings!` are now defined.
- Since the paper data were produced: the forward-Euler CFL bound no longer carries a dimension factor.
- `disable_debug_timings!` had no effect when called in the same world as `enable_debug_timings!`.

### Deprecated

- The positional `fai_preconditioner(A, μ, solver, D_kwargs; kwargs...)`: use the keyword form, with the `cutoff`,
  `maxdim` and `oversample` of `D_kwargs` passed as `D_cutoff`, `D_maxdim` and `D_oversample`. The positional form
  forwards to it with a deprecation warning, and gives the same result.

### Known limitations

- Chorin projection: the mean removal of the regularized pressure right-hand side subtracts `Σrhs / Π(nᵢ - 1)`
  instead of `Σrhs / Π nᵢ` (inactive when the pressure has a Dirichlet face), and the projection `u = u* - dt ∇p` also
  changes the Dirichlet wall rows of the velocity. Both are recorded as `@test_broken` and are to be fixed in v0.2.
