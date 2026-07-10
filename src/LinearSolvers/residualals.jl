"""
    ResidualALS <: AbstractDMRGSolver

Standard Alternating Linear Scheme (DMRG) for solving linear systems.
"""
Base.@kwdef struct ResidualALS <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    updater::Any = nothing
end

function linearsolve(solver::ResidualALS, A, b, x0; kwargs...)
    # TODO: Implement Standard Alternating Linear Scheme based on residual truncation
    error("Residual truncation ALS solver not yet implemented")
end

function linearsolve(solver::ResidualALS, A, b, M⁻¹, x0; kwargs...)
    # TODO: Implement Preconditioned Alternating Linear Scheme based on residual truncation
    error("Residual truncation ALS solver not yet implemented")
end

