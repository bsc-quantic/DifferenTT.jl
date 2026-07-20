using ITensorMPS: ReducedLinearProblem

"""
    ALS <: AbstractDMRGSolver

Standard Alternating Linear Scheme (DMRG) for solving linear systems.
"""
Base.@kwdef struct ALS <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    updater::AbstractUpdater = KrylovUpdater()
end

# TODO Support different coefficients

function linearsolve(solver::ALS, A::MPO, b::MPS, x0::Union{MPS, Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_problem = ReducedLinearProblem(A, b)
    updater_kwargs = (; coefficients = (false, true), NamedTuple(solver.updater_kwargs)...)
    return ITensorMPS.alternating_update(reduced_problem, init; solver.updater, updater_kwargs, NamedTuple(solver.sweep_kwargs)...)
end

function linearsolve(solver::ALS,  A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS, Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_precond_problem = initialize_reducedprecondlinearproblem(A, b, M⁻¹)
    updater_kwargs = (; coefficients = (false, true), NamedTuple(solver.updater_kwargs)...)
    return ITensorMPS.alternating_update(reduced_precond_problem, init; solver.updater, updater_kwargs, NamedTuple(solver.sweep_kwargs)...)
end

