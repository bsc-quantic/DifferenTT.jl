abstract type AbstractLinearSolver end

# DMRG-based linear solvers
abstract type AbstractDMRGSolver <: AbstractLinearSolver end

# TT-manifold global solvers
abstract type AbstractGlobalSolver <: AbstractLinearSolver end

"""
    linearsolve(solver::AbstractLinearSolver, A, b, x0; kwargs...)
    linearsolve(solver::AbstractLinearSolver, A, b, M⁻¹, x0; kwargs...)

Solve the linear system `A x = b` using the specified `solver`.

# Returns
- `x`: the solved state.
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
    GlobalUpdaterArgs

Configuration for the global truncated updaters runnin on the TT-Manifold
"""
Base.@kwdef struct GlobalUpdaterArgs
    ishermitian::Bool = false
    isposdef::Bool = false
    krylovdim::Int = 30
    maxiter::Int = 100
    rtol::Float64 = 1e-8
    verbosity::Int = 0
    maxdim::Int = typemax(Int)
    cutoff::Float64 = 1e-8
    orth_maxdim::Int = typemax(Int)
    orth_cutoff::Float64 = 1e-12
end

"""
    SweepingArgs

Configuration for the sweeps in DMRG-based linear solvers.
"""
Base.@kwdef struct SweepingArgs
    nsweeps::Int = 5
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
    Base.NamedTuple(args::GlobalUpdaterArgs)

Convert `GlobalUpdaterArgs` to a `NamedTuple`.
"""
function Base.NamedTuple(args::GlobalUpdaterArgs)
    return (;
        ishermitian = args.ishermitian,
        isposdef = args.isposdef,
        rtol = args.rtol,
        krylovdim = args.krylovdim,
        maxiter = args.maxiter,
        verbosity = args.verbosity,
        maxdim = args.maxdim,
        cutoff = args.cutoff,
        orth_maxdim = args.orth_maxdim,
        orth_cutoff = args.orth_cutoff,
    )
end

"""
    Base.NamedTuple(args::SweepingArgs)

Convert `SweepingArgs` to a `NamedTuple`.
"""
function Base.NamedTuple(args::SweepingArgs)
    return (;
        nsweeps = args.nsweeps,
        maxdim = args.maxdim,
        cutoff = args.cutoff,
        nsite = args.nsite,
        outputlevel = args.outputlevel,
        observer! = args.observer!,
        sweep_observer! = args.sweep_observer!,
        checkdone = args.checkdone
    )
end

"""
    default_linearsolve_init(const_term::MPS)

Generate a default initial guess of zeros for the linear system solver.
"""
function default_linearsolve_init(const_term::MPS)
    return zeros_mps(siteinds(const_term), backend_type(const_term))
end

"""
    AbstractUpdater

Abstract type for local updaters used in alternating linear schemes.
"""
abstract type AbstractUpdater end