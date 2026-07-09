"""
    ExpandedALS <: AbstractDMRGSolver

ALS(t + z) for solving linear systrems. It executes a 1-site DMRG and at the end of the sweep, expands the current
tensor train with another one given by the expander func.
"""
Base.@kwdef struct ExpandedALS <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    expander_func::Function = nothing
    updater::Any = nothing

    function ExpandedALS(updater_kwargs)
        if updater_kwargs.nsite != 1
            throw(ArgumentError("ALS(t+z) currently only supports nsite = 1."))
        end
        return new(updater_kwargs)
    end
end

function linearsolve(solver::ExpandedALS, A, b, x0; kwargs...)
    # TODO: Implement Standard Alternating Linear Scheme
    error("ALS(t + z) solver not yet implemented")
end