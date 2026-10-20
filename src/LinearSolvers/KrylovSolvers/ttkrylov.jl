"""
    _assign_ref_store!(store, value, field::Symbol)

Set `store[] = value`, unless `store` is `nothing`. Throws an `ArgumentError` naming the field
`GlobalUpdaterArgs.<field>` if `store` does not support `setindex!` without an index.
"""
function _assign_ref_store!(store::Any, value, field::Symbol)::Nothing
    if isnothing(store)
        return nothing
    end
    try
        store[] = value
    catch e
        e isa ArgumentError && rethrow()
        throw(
            ArgumentError(
                "GlobalUpdaterArgs.$(field) must be `nothing` or support `store[] = value` (e.g. a `Ref`). Got $(typeof(store)).",
            ),
        )
    end
    return nothing
end

"""
    record_global_krylov_info!(numiter_store, converged_store, info)

Store the iteration count and the convergence flag of the KrylovKit `info` of a global solve in the optional stores
of [`GlobalUpdaterArgs`](@ref), and warn if the solve did not converge.
"""
function record_global_krylov_info!(numiter_store::Any, converged_store::Any, info)::Nothing
    _assign_ref_store!(numiter_store, Int(info.numiter), :numiter_store)
    converged = info.converged != 0
    _assign_ref_store!(converged_store, converged, :converged_store)
    if !converged
        @warn "Global linear solver stopped without converging after $(info.numiter) iterations (normres = $(info.normres))"
    end
    return nothing
end
"""
    TTCG(; maxiter=100, tol=1e-8, verbosity=0, maxdim=typemax(Int), cutoff=1e-8, orth_maxdim=typemax(Int),
         orth_cutoff=1e-12) <: KrylovKit.KrylovAlgorithm

Conjugate gradient (CG) algorithm on tensor trains, run by `KrylovKit.linsolve(A, b, x₀, [M⁻¹,] alg::TTCG, a₀, a₁)`
and by [`GlobalCG`](@ref).

Its iterates are MPS whose bond dimensions are truncated after each operation, so that the cost stays bounded. We
therefore evaluate each step length with the exact sandwich ``\\langle \\mathbf{p}, A \\mathbf{p} \\rangle`` rather than
with the truncated product ``A\\mathbf{p}``, and recompute the residual from ``\\mathbf{x}`` when the recurrence
reports convergence, restarting CG if the true residual is still above the tolerance. Note that truncation still
perturbs the conjugacy of the search directions, so the convergence of CG is not guaranteed when the truncation is
severe (Sec. 2.1 of Torrente et al.).

# Fields

  - `maxiter::Int=100`: maximum number of iterations.
  - `tol::Float64=1e-8`: tolerance relative to ``\\|\\mathbf{b}\\|``: the solve stops at
    ``\\|\\mathbf{r}\\| \\le \\texttt{tol}\\, \\|\\mathbf{b}\\|``.
  - `verbosity::Int=0`: KrylovKit verbosity (1: warn on no convergence, 2: start and end, 3: every iteration).
  - `maxdim::Int=typemax(Int)`, `cutoff::Float64=1e-8`: truncation of the applications of `A` and `M⁻¹` and of the
    updates of ``\\mathbf{x}`` and ``\\mathbf{r}``.
  - `orth_maxdim::Int=typemax(Int)`, `orth_cutoff::Float64=1e-12`: truncation of the new search direction
    ``\\mathbf{p}``.
"""
Base.@kwdef struct TTCG <: KrylovKit.KrylovAlgorithm
    maxiter::Int = 100
    tol::Float64 = 1e-8
    verbosity::Int = 0
    maxdim::Int = typemax(Int)
    cutoff::Float64 = 1e-8
    orth_maxdim::Int = typemax(Int)
    orth_cutoff::Float64 = 1e-12
end

"""
    GlobalCG(; updater_kwargs=GlobalUpdaterArgs(isposdef=true, ishermitian=true)) <: AbstractGlobalSolver
    GlobalCG(updater_kwargs::GlobalUpdaterArgs)

Global CG solver for an SPD ``A \\mathbf{x} = \\mathbf{b}`` in the TT format: [`linearsolve`](@ref) runs a
[`TTCG`](@ref) built from `updater_kwargs` on the whole tensor train.

A global solver needs no sweeps and no local problems, but every iteration increases the bond dimensions of its
iterates, which we have to truncate, so its cost and its convergence both depend on how compressible the Krylov
vectors are (see [`AbstractGlobalSolver`](@ref)). The global solvers are not used in the experiments of Torrente et
al. With a preconditioner, `M⁻¹` is applied to the whole residual rather than to local problems.

# Fields

  - `updater_kwargs::GlobalUpdaterArgs`: tolerance, iteration limit, truncation and stores.

# Throws

  - `ArgumentError`: if `updater_kwargs.isposdef` or `updater_kwargs.ishermitian` is `false`.

# Examples

```julia
numiter = Ref(0)
solver = GlobalCG(;
    updater_kwargs=GlobalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-8, maxdim=40, numiter_store=numiter)
)
x = linearsolve(solver, A, b, nothing)
```

See also [`GlobalGMRES`](@ref), [`TTCG`](@ref).
"""
Base.@kwdef struct GlobalCG <: AbstractGlobalSolver
    updater_kwargs::GlobalUpdaterArgs = GlobalUpdaterArgs(isposdef=true, ishermitian=true)

    function GlobalCG(updater_kwargs)
        if !updater_kwargs.isposdef || !updater_kwargs.ishermitian
            throw(ArgumentError("GlobalCG requires updater_kwargs.isposdef and updater_kwargs.ishermitian to be true."))
        end
        return new(updater_kwargs)
    end
end

"""
    linearsolve(solver::GlobalCG, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...) -> MPS

Solve the SPD system ``A \\mathbf{x} = \\mathbf{b}`` by TT-CG from `x0`. The keywords override the fields of
`solver.updater_kwargs`; other keywords are ignored.
"""
function linearsolve(solver::GlobalCG, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0

    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))

    alg = TTCG(;
        maxiter=merged_kwargs.maxiter,
        tol=merged_kwargs.rtol,
        verbosity=merged_kwargs.verbosity,
        maxdim=get(merged_kwargs, :maxdim, typemax(Int)),
        cutoff=get(merged_kwargs, :cutoff, 1e-8),
        orth_maxdim=get(merged_kwargs, :orth_maxdim, typemax(Int)),
        orth_cutoff=get(merged_kwargs, :orth_cutoff, 1e-12),
    )

    x, info = KrylovKit.linsolve(A, b, init, alg)
    record_global_krylov_info!(merged_kwargs.numiter_store, merged_kwargs.converged_store, info)
    return x
end

"""
    linearsolve(solver::GlobalCG, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing}; kwargs...) -> MPS

Solve the SPD system ``A \\mathbf{x} = \\mathbf{b}`` by preconditioned TT-CG from `x0`. The preconditioner must be
SPD; a [`FactorizedPreconditioner`](@ref) ``G_L^\\top G_L`` is positive semidefinite by construction. The keywords
override the fields of `solver.updater_kwargs`; other keywords, including `preconditioner_sites`, are ignored.
"""
function linearsolve(solver::GlobalCG, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0

    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))

    alg = TTCG(;
        maxiter=merged_kwargs.maxiter,
        tol=merged_kwargs.rtol,
        verbosity=merged_kwargs.verbosity,
        maxdim=get(merged_kwargs, :maxdim, typemax(Int)),
        cutoff=get(merged_kwargs, :cutoff, 1e-8),
        orth_maxdim=get(merged_kwargs, :orth_maxdim, typemax(Int)),
        orth_cutoff=get(merged_kwargs, :orth_cutoff, 1e-12),
    )

    x, info = KrylovKit.linsolve(A, b, init, M⁻¹, alg)
    record_global_krylov_info!(merged_kwargs.numiter_store, merged_kwargs.converged_store, info)
    return x
end

"""
    TTGMRES(; krylovdim=30, maxiter=100, tol=1e-8, verbosity=0, maxdim=typemax(Int), cutoff=1e-8,
            orth_maxdim=typemax(Int), orth_cutoff=1e-12) <: KrylovKit.KrylovAlgorithm

Restarted GMRES algorithm on tensor trains (Dolgov, 2013), run by
`KrylovKit.linsolve(A, b, x₀, [M⁻¹,] alg::TTGMRES, a₀, a₁)` and by [`GlobalGMRES`](@ref).

Each cycle builds an Arnoldi basis of at most `krylovdim` MPS by classical Gram–Schmidt: every entry
``H_{ik} = \\langle \\mathbf{v}_i, A \\mathbf{v}_k \\rangle`` of the Hessenberg matrix is an exact sandwich with the
untruncated ``\\mathbf{v}_k``, while the orthogonalized vector is truncated with `orth_maxdim` and `orth_cutoff`.
Since the basis is truncated and never reorthogonalized, it is only approximately orthonormal, and the least-squares
residual of the cycle is only an estimate. We therefore recompute the residual from ``\\mathbf{x}`` at the end of
every cycle, at the cost of one more application of `A`, and stop only when it is below the tolerance.

# Fields

  - `krylovdim::Int=30`: maximum dimension of the Krylov basis before a restart.
  - `maxiter::Int=100`: maximum number of restart cycles; `info.numiter` counts the cycles.
  - `tol::Float64=1e-8`: tolerance relative to ``\\|\\mathbf{b}\\|``.
  - `verbosity::Int=0`: KrylovKit verbosity (1: warn on no convergence, 2: start and end, 3: every step).
  - `maxdim::Int=typemax(Int)`, `cutoff::Float64=1e-8`: truncation of the applications of `A` and `M⁻¹` and of the
    updates of ``\\mathbf{x}`` and ``\\mathbf{r}``.
  - `orth_maxdim::Int=typemax(Int)`, `orth_cutoff::Float64=1e-12`: truncation of the Gram–Schmidt step.
"""
Base.@kwdef struct TTGMRES <: KrylovKit.KrylovAlgorithm
    krylovdim::Int = 30
    maxiter::Int = 100
    tol::Float64 = 1e-8
    verbosity::Int = 0
    maxdim::Int = typemax(Int)
    cutoff::Float64 = 1e-8
    orth_maxdim::Int = typemax(Int)
    orth_cutoff::Float64 = 1e-12
end

"""
    GlobalGMRES(; updater_kwargs=GlobalUpdaterArgs()) <: AbstractGlobalSolver
    GlobalGMRES(updater_kwargs::GlobalUpdaterArgs)

Global restarted GMRES solver for ``A \\mathbf{x} = \\mathbf{b}`` in the TT format, for operators that are not SPD:
[`linearsolve`](@ref) runs a [`TTGMRES`](@ref) built from `updater_kwargs` on the whole tensor train. As for
[`GlobalCG`](@ref), the truncation of the iterates sets both the cost and the attainable accuracy.

# Fields

  - `updater_kwargs::GlobalUpdaterArgs`: Krylov dimension, tolerance, number of restarts, truncation and stores.

# Examples

```julia
solver = GlobalGMRES(; updater_kwargs=GlobalUpdaterArgs(; krylovdim=20, maxiter=10, rtol=1e-8, maxdim=40))
x = linearsolve(solver, A, b, nothing)
```

See also [`GlobalCG`](@ref), [`TTGMRES`](@ref).
"""
Base.@kwdef struct GlobalGMRES <: AbstractGlobalSolver
    updater_kwargs::GlobalUpdaterArgs = GlobalUpdaterArgs()
end

"""
    linearsolve(solver::GlobalGMRES, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` by restarted TT-GMRES from `x0`. The keywords override the fields of
`solver.updater_kwargs`; other keywords are ignored.
"""
function linearsolve(solver::GlobalGMRES, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0

    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))

    alg = TTGMRES(;
        krylovdim=merged_kwargs.krylovdim,
        maxiter=merged_kwargs.maxiter,
        tol=merged_kwargs.rtol,
        verbosity=merged_kwargs.verbosity,
        maxdim=get(merged_kwargs, :maxdim, typemax(Int)),
        cutoff=get(merged_kwargs, :cutoff, 1e-8),
        orth_maxdim=get(merged_kwargs, :orth_maxdim, typemax(Int)),
        orth_cutoff=get(merged_kwargs, :orth_cutoff, 1e-12),
    )

    x, info = KrylovKit.linsolve(A, b, init, alg)
    record_global_krylov_info!(merged_kwargs.numiter_store, merged_kwargs.converged_store, info)
    return x
end

"""
    linearsolve(solver::GlobalGMRES, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing};
                kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` by left-preconditioned restarted TT-GMRES from `x0`. The keywords override the
fields of `solver.updater_kwargs`; other keywords, including `preconditioner_sites`, are ignored.
"""
function linearsolve(
    solver::GlobalGMRES, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing}; kwargs...
)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0

    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))

    alg = TTGMRES(;
        krylovdim=merged_kwargs.krylovdim,
        maxiter=merged_kwargs.maxiter,
        tol=merged_kwargs.rtol,
        verbosity=merged_kwargs.verbosity,
        maxdim=get(merged_kwargs, :maxdim, typemax(Int)),
        cutoff=get(merged_kwargs, :cutoff, 1e-8),
        orth_maxdim=get(merged_kwargs, :orth_maxdim, typemax(Int)),
        orth_cutoff=get(merged_kwargs, :orth_cutoff, 1e-12),
    )

    x, info = KrylovKit.linsolve(A, b, init, M⁻¹, alg)
    record_global_krylov_info!(merged_kwargs.numiter_store, merged_kwargs.converged_store, info)
    return x
end

"""
    apply(P::Preconditioner, v::MPS; kwargs...) -> MPS
    apply(P::FactorizedPreconditioner, v::MPS; kwargs...) -> MPS

Return ``M^{-1}\\mathbf{v}``, computed by `ITensorMPS.apply` with the truncation `kwargs`. For a
[`FactorizedPreconditioner`](@ref) we apply the two factors in turn, ``G_L^\\top (G_L \\mathbf{v})``, truncating after
each, so that the product ``G_L^\\top G_L`` is never formed.
"""
function apply(P::Preconditioner, v::MPS; kwargs...)
    return apply(P.operator, v; kwargs...)
end

function apply(P::FactorizedPreconditioner, v::MPS; kwargs...)
    v_out = apply(P.operator, v; kwargs...)
    return apply(transpose_mpo(P.operator), v_out; kwargs...)
end

"""
    inner(v::MPS, P::Preconditioner, w::MPS)
    inner(v::MPS, P::FactorizedPreconditioner, w::MPS)

Return ``\\langle \\mathbf{v}, M^{-1} \\mathbf{w} \\rangle`` without truncation. For a
[`FactorizedPreconditioner`](@ref) we evaluate ``\\langle G_L \\mathbf{v}, G_L \\mathbf{w} \\rangle``, which is
non-negative for ``\\mathbf{v} = \\mathbf{w}``. Unlike `ITensorMPS.inner(v', A, w)`, `v` is passed unprimed. Methods are
defined for both `ITensors.inner` and `KrylovKit.inner`.
"""
function ITensors.inner(v::MPS, P::Preconditioner, w::MPS)
    return ITensorMPS.inner(v', P.operator, w)
end

function KrylovKit.inner(v::MPS, P::Preconditioner, w::MPS)
    return ITensorMPS.inner(v', P.operator, w)
end

function ITensors.inner(v::MPS, P::FactorizedPreconditioner, w::MPS; kwargs...)
    return ITensorMPS.inner(P.operator, v, P.operator, w)
end

function KrylovKit.inner(v::MPS, P::FactorizedPreconditioner, w::MPS; kwargs...)
    return ITensorMPS.inner(P.operator, v, P.operator, w)
end
