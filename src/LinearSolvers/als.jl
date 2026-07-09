"""
    ALS <: AbstractDMRGSolver

Standard Alternating Linear Scheme (DMRG) for solving linear systems.
"""
Base.@kwdef struct ALS <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    updater::Any = nothing
end

function linearsolve(solver::ALS, A, b, x0; kwargs...)
    # TODO: Implement Standard Alternating Linear Scheme
    error("ALS solver not yet implemented")
end
