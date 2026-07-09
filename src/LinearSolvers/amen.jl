"""
    AMEn <: AbstractDMRGSolver

Standard Alternating Minimal Energy (AMEn) for solving linear systems.
"""
Base.@kwdef struct AMEn <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    updater::Any = nothing
end

function linearsolve(solver::AMEn, A, b, x0; kwargs...)
    # TODO: Implement Standard Alternating Minimal Energy method
    error("AMEn solver not yet implemented")
end