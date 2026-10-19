"""
    ALS(; sweep_kwargs=SweepingArgs(), updater_kwargs=LocalUpdaterArgs(), updater=KrylovUpdater()) <: AbstractDMRGSolver

Alternating Linear Scheme (ALS) for ``A \\mathbf{x} = \\mathbf{b}`` in the TT format (Holtz, Rohwedder & Schneider,
2012), built on the sweeps of `ITensorMPS.alternating_update`.

At each step, we project the system onto `sweep_kwargs.nsite` adjacent sites, solve the local problem with `updater`,
and split the new local tensor back into the tensor train by an SVD truncated to `sweep_kwargs.maxdim` and
`sweep_kwargs.cutoff`. With `nsite = 2`, the default, this is the two-site variant also known as Modified ALS (MALS),
whose SVD can increase or decrease the bond dimension of the solution. With `nsite = 1` the bond dimension of `x0` is
kept, so the 1-site variant only converges if `x0` already has enough rank; [`AMEn`](@ref) and
[`ExpandedALS`](@ref) are the 1-site solvers that enrich the basis. For an SPD `A`, set `ishermitian=true` and
`isposdef=true` in `updater_kwargs` so that the local problems, which are SPD as well, are solved by CG.

The truncation of `ALS` is controlled by the singular values alone. [`ResidualALS`](@ref) instead chooses the rank
by its effect on the local residual, which is the variant used in the paper's experiments.

# Fields

  - `sweep_kwargs::SweepingArgs`: sweeps, truncation and observers.
  - `updater_kwargs::LocalUpdaterArgs`: local Krylov solver and diagnostics.
  - `updater::AbstractUpdater`: local solver, [`KrylovUpdater`](@ref) or [`QRUpdater`](@ref).

# Examples

```julia
solver = ALS(;
    sweep_kwargs=SweepingArgs(; nsweeps=4, maxdim=16),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true),
)
x = linearsolve(solver, A, b, nothing)
```

See also [`linearsolve`](@ref), [`ResidualALS`](@ref).
"""
Base.@kwdef struct ALS <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    updater::AbstractUpdater = KrylovUpdater()
end

# TODO(release): support the shifted system (a₀ + a₁A)x = b; the updaters already accept `coefficients`, but
# `linearsolve` always passes (false, true).

"""
    linearsolve(solver::ALS, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` by ALS sweeps from `x0`. The keywords override the fields of
`solver.sweep_kwargs` (see [`linearsolve`](@ref)).
"""
function linearsolve(solver::ALS, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_problem = ReducedLinearProblem(A, b)
    updater_kwargs = (; coefficients=(false, true), NamedTuple(solver.updater_kwargs)...)
    sweep_kwargs = linearsolve_sweep_kwargs(solver, A, b, kwargs)
    return alternating_update(reduced_problem, init; solver.updater, updater_kwargs, sweep_kwargs...)
end

"""
    linearsolve(solver::ALS, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing};
                preconditioner_sites=nothing, kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` by ALS sweeps from `x0`, preconditioning each local problem with the projection
of `M⁻¹` (Sec. 4 of Torrente et al.). Only the local problems whose first site is in `preconditioner_sites` are
preconditioned, when it is given. The other keywords override the fields of `solver.sweep_kwargs` (see
[`linearsolve`](@ref)).
"""
function linearsolve(solver::ALS, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_precond_problem = initialize_reducedprecondlinearproblem(A, b, M⁻¹)
    updater_kwargs, sweep_kwargs = precond_linearsolve_kwargs(solver, A, b, kwargs)
    return alternating_update(reduced_precond_problem, init; solver.updater, updater_kwargs, sweep_kwargs...)
end
