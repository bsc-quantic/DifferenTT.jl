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

"""
    Base.NamedTuple(args::LocalUpdaterArgs)

Convert `LocalUpdaterArgs` to a `NamedTuple`.
"""
function Base.NamedTuple(args::LocalUpdaterArgs)
    return (;
        ishermitian = args.ishermitian,
        isposdef = args.isposdef,
        compute_cond = args.compute_cond,
        rtol = args.rtol,
        krylovdim = args.krylovdim,
        maxiter = args.maxiter,
        verbosity = args.verbosity,
        numiter_store = args.numiter_store,
        cond_store = args.cond_store,
        pc_cond_store = args.pc_cond_store
    )
end

"""
    Base.NamedTuple(args::SweepingArgs)

Convert `SweepingArgs` to a `NamedTuple`.
"""
function Base.NamedTuple(args::SweepingArgs)
    return (;
        sweeps = args.sweeps,
        maxdim = args.maxdim,
        cutoff = args.cutoff,
        nsite = args.nsite,
        outputlevel = args.outputlevel,
        observer! = args.observer!,
        sweep_observer! = args.sweep_observer!,
        checkdone = args.checkdone
    )
end