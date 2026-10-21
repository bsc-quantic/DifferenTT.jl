```@meta
CurrentModule = DifferenTT
```

# [Linear solvers](@id manual_linear_solvers)

The discretization of an elliptic PDE in the TT format gives a linear system ``A \mathbf{x} = \mathbf{b}`` in which the
operator ``A`` is an MPO, the right-hand side ``\mathbf{b}`` an MPS, and the unknown ``\mathbf{x}`` must again be an MPS
of low bond dimension. We cannot apply a classical Krylov method directly, because its iterates would have growing
bond dimensions. DifferenTT therefore provides two classes of solvers, as in Sec. 2.1 of the paper, and a single entry
point, [`linearsolve`](@ref), that runs any of them. The signatures are in the
[API reference](@ref api_linearsolvers).

```julia
x = linearsolve(solver, A, b, x0)            # x0 = nothing starts from the zero MPS
x = linearsolve(solver, A, b, M⁻¹, x0)       # preconditioned
```

## DMRG-type solvers

These solvers sweep along the chain. At each step they project the system onto one or two sites, solve the local problem
with a Krylov method, and recover the TT format by an SVD that also sets the bond dimension. Since the local solve is
exact within the current subspace, its convergence is not affected by truncation, which makes these solvers robust.

| Solver | Purpose |
|---|---|
| [`ALS`](@ref) | Alternating linear scheme on one or two sites (Holtz, Rohwedder & Schneider, 2012). Two sites (MALS) lets the bond dimension change, with the rank chosen by singular values alone. |
| [`ResidualALS`](@ref) | Two-site ALS whose SVD truncation chooses the rank from the local residual (Oseledets & Dolgov, 2012). This is the MALS solver of the paper's experiments. |
| [`AMEn`](@ref) | One-site solver that adapts the rank by enriching each core with directions from the residual (Dolgov & Savostyanov, 2014). |
| [`ExpandedALS`](@ref) | One-site ALS whose basis is expanded between sweeps by an [`AbstractALSExpansion`](@ref), such as [`ResidualExpansion`](@ref). |

Each solver stores three pieces of configuration: [`SweepingArgs`](@ref) (sweeps, truncation, observers),
[`LocalUpdaterArgs`](@ref) (the local Krylov solver and diagnostics) and the local updater. For an SPD system, set both
`ishermitian=true` and `isposdef=true` in the `LocalUpdaterArgs`, since KrylovKit selects CG only then. The keywords of
`linearsolve` override the stored configuration for that call.

!!! note "Stopping the sweeps"
    A [`ResidualObserver`](@ref) stops the sweeps when the relative residual of the whole system falls below `rtol`. The
    residual is evaluated by an expansion whose own rounding error is about ``10^{-8}`` relative to ``\|\mathbf{b}\|``,
    so a tolerance below that value cannot be reached by this observer.

## Global solvers

[`GlobalCG`](@ref) and [`GlobalGMRES`](@ref) run a Krylov method on the whole tensor train, with the iterates truncated
after every operation. They need no sweeps, but the truncation perturbs the Krylov recurrences, so convergence is
guaranteed only when the iterates are compressible. CG needs an SPD operator, and GMRES does not. They are configured by
[`GlobalUpdaterArgs`](@ref), and the underlying algorithms are [`TTCG`](@ref) and [`TTGMRES`](@ref).

## Preconditioning and diagnostics

A preconditioner goes between the right-hand side and the initial guess of `linearsolve`: the DMRG-type solvers apply it
to the local problems, in the projected form of Sec. 4 of the paper, and the global ones to the whole residual.
[Preconditioners](@ref) describes how to build one. The keyword `preconditioner_sites` restricts the preconditioning to
some local problems.

The local solves can be instrumented. [`KrylovSweepStats`](@ref) records the number of Krylov iterations per site and
sweep, [`KrylovCondStats`](@ref) the condition number of the local operator, and [`KrylovSpectrumStats`](@ref) its
spectrum, which [`spectrum_table`](@ref) summarizes. Each one is passed to the `LocalUpdaterArgs`, costs nothing when it
is not requested, and the dense local spectrum, of cost ``\mathcal{O}(n_{\mathrm{loc}}^3)``, is computed only when a
diagnostic asks for it. A local block that is not SPD raises no warning unless `spd_tol` is set, in which case a
[`NotSPDError`](@ref) stops the solve.
