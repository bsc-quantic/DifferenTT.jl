abstract type AbstractLinearSolver end

# DMRG-based linear solvers
abstract type AbstractDMRGSolver <: AbstractLinearSolver end

# TT-manifold global solvers
abstract type AbstractGlobalSolver <: AbstractLinearSolver end

"""
    linearsolve(solver::AbstractLinearSolver, A, b, x0; kwargs...)

Solve the linear system A x = b with initial guess x0 using the specified `solver`.
Returns the solution `x` and a `NamedTuple` with solver statistics (e.g., info).
"""
function linearsolve end

"""
    LocalUpdaterArgs

Configuration for the local subspace updater used in DMRG-based linear solvers.
"""
Base.@kwdef struct LocalUpdaterArgs
    ishermitian::Bool = false
    isposdef::Bool = false
    compute_cond::Bool = false
    rtol::Float64 = 1e-8
    krylovdim::Int = 30
    maxiter::Int = 100
    verbosity::Int = 0
    numiter_store::Any = nothing
    cond_store::Any = nothing
    pc_cond_store::Any = nothing
end

"""
    LocalUpdaterArgs

Configuration for the sweeps in DMRG-based linear solvers.
"""
Base.@kwdef struct SweepingArgs
    sweeps::Int = 5
    maxdim::Int = 30
    cutoff::Float64 = 1e-10
    nsite::Int = 2
    outputlevel::Int = 0
    observer!::Any = ITensorMPS.default_observer()
    sweep_observer!::Any = ITensorMPS.default_sweep_observer()
    checkdone::Any = ITensorMPS.default_checkdone()
end