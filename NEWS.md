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
- Invalid arguments that used to throw an `ErrorException` throw `ArgumentError` or `DimensionMismatch`:
  `toeplitz_mpo` with too few sites for its stencil, `column_pick_mpo`, `row_pick_mpo` and `entry_mpo` with an index
  outside `1:2^N` (an index above `2^N` threw a `DimensionMismatch` from `arrays2mpo`), `triang_projector_mpo` with a
  number of combiners different from the number of sites, `TensorCrossInterpolation.optfirstpivot` when every value it
  tries is zero, `_rsi_tt_mul` and `elementwise_function` with `nsite < 2` (`tt_mul` still falls back to the direct
  product), `backsolveL`/`backsolveU` on a non-square matrix, and `interpolative` with prescribed `cols` for the
  `"prrldu"` algorithm.
- The library prints nothing by default. `DirichletBC` no longer logs at construction (the note moved to its
  docstring). The `ChorinCache` regularization flag, `linsolve_preconditioner` and the `ResidualObserver` early stop are
  `@debug`. The CFL and Kolmogorov checks and the `vmax = 1` fallback of `INSVorticityExperiment` keep their
  `@info`/`@warn` messages, since they tell how an experiment will behave for its parameters.
- `fai_dg`, and through it `fai_preconditioner`, compute and log the residual of the projected FAI system only with
  `compute_residual = true` (the default is `false`; it used to be always computed and logged).
- The global solvers (`GlobalCG`, `GlobalGMRES`) warn when they stop without converging, and throw `ArgumentError`
  for a `numiter_store` or `converged_store` that does not support `store[] = value`.
- `ResidualALS.residual_tol` and `residual_damp` are `Float64`.
- `ResidualALS` follows the ITensors cutoff convention in the Frobenius floor of its residual-based truncation: the
  discarded relative weight `Σσ²` is at most `cutoff`, as for `ALS` and `AMEn` (it was `cutoff^2`). With `cutoff > 0`
  the floor keeps fewer singular values than before, so the bond dimensions and solutions of `ResidualALS` differ from
  v0.0.1 and from the paper data, which used `cutoff = 1e-13` (Sec. 5.1) or `cutoff = 0` (Sec. 5.2, unaffected).
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
- `vec2tensor(v, leg_dims)` and `mat2tensor(m, leg_dims)` take the leg dimensions in site order (most significant
  first, as `siteinds` and `contract_mps`/`contract_mpo`), and return a `d₁ × … × d_N` array (for `mat2tensor`, the row
  axes followed by the column axes). They used to take them least significant first and return the axes reversed:
  `vec2tensor(collect(1:6), [2, 3])` is now `[1 2 3; 4 5 6]` instead of the 3 × 2 `[1 2; 3 4; 5 6]`. Results change
  only when the dimensions are not a palindrome, so QTT (all sites of dimension 2) is unaffected.

### Fixed

- TT-CG on shifted systems `(a₀I + a₁A)x = b` with `a₀ ≠ 0`: the step denominator used `a₁⟨p,p⟩` instead of `a₀⟨p,p⟩`,
  so the iteration diverged. `GlobalCG`, which solves the unshifted system, was not affected.
- Preconditioned `GlobalGMRES` threw a site-index mismatch in the Arnoldi step.
- Preconditioned GMRES (`preconditioned_linsolve` with `GMRES`, and TT-GMRES with `M⁻¹`) that stopped without
  converging returned the norm of the preconditioned residual `M⁻¹r` as `normres`; the dense version also returned a
  residual `r` that was not updated after the last restart. Both now return the residual of the unpreconditioned
  system and its norm.
- `ResidualExpansion` throws `ArgumentError` instead of `ErrorException` when the sites of `A` and `b` do not match.
- **FAI construction.** `fai_dg`, and through it `fai_preconditioner`, applied `A` to the row index of the vectorized
  unknown, so it solved `(A G̃_L)_ij = δ_ij` for `i ≥ j` (the transposed FSAI) instead of Eq. (10) of the paper,
  `(G̃_L A)_ij = δ_ij`. `G_L = D G̃_L` was then not the inverse Cholesky factor `L_A⁻¹`: even for an exactly solved
  `G̃_L`, `κ(G_LᵀG_L A)` was 3.3 on an 8 × 8 Laplacian (κ(A) = 32) and 79 on a 2⁶-point QTT Laplacian (κ(A) = 1712),
  instead of 1. The preconditioner was positive definite, but every FAI result differs from v0.0.1 and from the paper
  data, which were produced with the transposed construction.
- `linsolve_preconditioner` had the same orientation error and returned `A⁻ᵀ` instead of `A⁻¹` for a nonsymmetric
  operator (both with and without `symmetrize`). Results for symmetric operators are unchanged.
- `inverse_laplacian_preconditioner` throws `ArgumentError` instead of `ErrorException` for unsupported boundary
  conditions.
- `fai_preconditioner(...; preconditioning=true)` preconditions the complement block `μ(I - P_L)` of Eq. (22) with
  `(I - P_L)/μ` instead of `μ(I - P_L)`. The build needed about 22 times more local iterations for `μ ≥ 1e3`; results
  for `μ = 1` are unchanged.
- `DifferentialOpsScalar(grid, field, n_eyepad)` threw on a missing grid field. The padded operator is scaled by the
  spacing of the coarse grid of `2^(n - p)` nodes over the same box in QTT, and by the original spacing in TT.
- The exports `sited_dim` (now `site_dim`) and `enable_debug_timings!` named functions that did not exist;
  `enable_debug_timings!` and `disable_debug_timings!` are now defined.
- Since the paper data were produced: the forward-Euler CFL bound no longer carries a dimension factor.
- `disable_debug_timings!` had no effect when called in the same world as `enable_debug_timings!`.
- `tt_mul` (MPS-MPS) and `elementwise_function` returned their (first) input unchanged, with no error, for an MPS with
  fewer sites than `nsite` (e.g. a single site); the product or map is now computed on the full tensor. Any
  `nsite > 2` threw.
- `site_dims(::TTArchitecture)` returned the dimensions in variable order instead of site order, so for a `var_map`
  that permutes variables of different sizes, `siteinds(arch)` built sites of the wrong dimension (and the TCI
  wrappers sampled with the wrong local dimensions).
- `ndim_operator_mpo` on a `TTArchitecture` returned the identity, with no error, for an `axis` outside `1:N`; it
  throws `ArgumentError`.
- `ndim_operator_mpo` on a `QTTArchitecture` threw a `BoundsError` for a variable with a single site.
- `zeros_padding` transposed the operator it pads when its tensors store the site indices as `(s, s')`. Its callers
  (`corner_pick_mpo` and the QTT Neumann corrections) store `(s', s)`, so their operators are unchanged.
- `upper_triang_mpo` and `lower_triang_mpo` threw a `BoundsError` on a single site, and so did `arrays2mps` and
  `arrays2mpo`.
- `toeplitz_mpo` accepted two sites with three coefficients but threw a `DimensionMismatch`; it now returns the
  4 × 4 Toeplitz (or circulant) matrix.
- `vec2mps` and `mat2mpo` scrambled the entries of the vector or matrix whenever the site dimensions were not a
  palindrome (e.g. sites of dimensions `[2, 3]`), with no error. QTT sites all have dimension 2, so their QTT uses were
  correct.
- `ndim_operator_mpo` and `ndim_operator_sum_mpo` on a `TTArchitecture` rebuilt the matrices with the wrapper type of
  the (first) operator, with no error: an `Adjoint` or `Transpose` operator was transposed back, and a `Symmetric`,
  `Diagonal` or triangular first operator symmetrized, diagonalized or truncated the other operators of the sum (a
  view threw). `backend_type` of an array now looks through such wrappers to the array that stores the data. The
  library's own callers pass dense matrices, so their operators are unchanged.
- `ones_mps` and `ones_mpo` on a single site ignored `array_type`.
- Array backends (found by code review; there is no GPU test yet): the interpolative decompositions of the RSI crosses
  called `prrldu` on the device matrix, which has no method for it, so on a GPU `elementwise_function` threw and
  `tt_mul` always fell back to the direct product; they now run on the CPU and move their factors back. The sketches
  of `sketch`, `positive_sketch` and `suffix_sketches`, the delta tensors of `diagmpo`, `diagmps` and
  `triang_projector_mpo`, and the link tensors of `zeros_padding` were created on the CPU; they now follow the backend
  of their input. `sample` contracts on the CPU. Results on the CPU are unchanged.
- `push!` and `setindex!` on a `VectorField` appended or replaced the states of the first components before a site
  mismatch in a later component threw, which left the components with different numbers of stored states; they now
  check every component first. `push!` and `setindex!` on a `ScalarField` or `VectorField` return the field (they
  returned its internal vector of states, or `nothing`).
- The QTT Neumann corrections of the differential operators (`left_neumann_boundary_correction`,
  `right_neumann_boundary_correction`) ignored `array_type` and built their corner block on the CPU, so a Neumann side
  could not be used on a GPU (found by code review). Results on the CPU are unchanged.
- `chorin_evolve!` and `vorticity_evolve!` with `save_every > 1` also stored the state of step `nsteps - 1` whenever
  `nsteps - 1` was not a multiple of `save_every`, and the `k`-th entry of the logged `linsolve_errorlog` was the
  residual of a different step than the `k`-th stored pressure or stream function. The fields now store the initial
  state and the steps `save_every`, `2 save_every`, … and `nsteps`, with one residual per stored step; `save_every < 1`
  throws `ArgumentError`. The paper's workers call `chorin_step` and are not affected.

### Deprecated

- The positional `fai_preconditioner(A, μ, solver, D_kwargs; kwargs...)`: use the keyword form, with the `cutoff`,
  `maxdim` and `oversample` of `D_kwargs` passed as `D_cutoff`, `D_maxdim` and `D_oversample`. The positional form
  forwards to it with a deprecation warning, and gives the same result.

### Known limitations

- Chorin projection: the mean removal of the regularized pressure right-hand side subtracts `Σrhs / Π(nᵢ - 1)`
  instead of `Σrhs / Π nᵢ` (inactive when the pressure has a Dirichlet face), and the projection `u = u* - dt ∇p` also
  changes the Dirichlet wall rows of the velocity. Both are recorded as `@test_broken` and are to be fixed in v0.2.
