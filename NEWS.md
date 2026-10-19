# DifferenTT.jl release notes

## v0.1.0 (unreleased)

### Breaking

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
- `VectorField(components)` needs a non-empty tuple of one concrete `ScalarField{N}` type, with `N` components; a
  tuple of the wrong length throws `DimensionMismatch`.

### Fixed

- TT-CG on shifted systems `(a₀I + a₁A)x = b` with `a₀ ≠ 0`: the step denominator used `a₁⟨p,p⟩` instead of `a₀⟨p,p⟩`,
  so the iteration diverged. `GlobalCG`, which solves the unshifted system, was not affected.
- Preconditioned `GlobalGMRES` threw a site-index mismatch in the Arnoldi step.
- `DifferentialOpsScalar(grid, field, n_eyepad)` threw on a missing grid field. The padded operator is scaled by the
  spacing of the coarse grid of `2^(n - p)` nodes over the same box in QTT, and by the original spacing in TT.
- The exports `sited_dim` (now `site_dim`) and `enable_debug_timings!` named functions that did not exist;
  `enable_debug_timings!` and `disable_debug_timings!` are now defined.

### Known limitations

- Chorin projection: the mean removal of the regularized pressure right-hand side subtracts `Σrhs / Π(nᵢ - 1)`
  instead of `Σrhs / Π nᵢ` (inactive when the pressure has a Dirichlet face), and the projection `u = u* - dt ∇p` also
  changes the Dirichlet wall rows of the velocity. Both are recorded as `@test_broken` and are to be fixed in v0.2.
