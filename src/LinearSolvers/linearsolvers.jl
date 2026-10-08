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
    spectrum_store::Any = nothing
    spd_tol::Union{Nothing, Float64} = nothing

    function LocalUpdaterArgs(ishermitian, isposdef, compute_cond, rtol, krylovdim, maxiter, verbosity, numiter_store, cond_store, pc_cond_store, spectrum_store, spd_tol)
        if compute_cond && (!ishermitian || !isposdef)
            @warn "compute_cond=true is typically only meaningful for Conjugate Gradient (CG) solvers, which require ishermitian=true and isposdef=true."
        end
        new(ishermitian, isposdef, compute_cond, rtol, krylovdim, maxiter, verbosity, numiter_store, cond_store, pc_cond_store, spectrum_store, spd_tol)
    end
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
    numiter_store::Any = nothing
end

# Bridge beween DifferenTT and ITensorMPS to avoid crashes in the observer pipeline
function default_differenTT_checkdone(; kwargs...)
    obs = get(kwargs, :observer, get(kwargs, :observer!, ITensorMPS.default_observer()))
    sweep_obs = get(kwargs, :sweep_observer, get(kwargs, :sweep_observer!, ITensorMPS.default_sweep_observer()))
    
    clean_kwargs = filter(kv -> kv.first ∉ (:observer, :observer!, :sweep_observer, :sweep_observer!), kwargs)
    
    obs_done = (typeof(obs) === typeof(ITensorMPS.default_observer())) ? false : ITensorMPS.checkdone!(obs; clean_kwargs...)
    sweep_obs_done = (typeof(sweep_obs) === typeof(ITensorMPS.default_sweep_observer())) ? false : ITensorMPS.checkdone!(sweep_obs; clean_kwargs...)
    
    return obs_done || sweep_obs_done
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
    checkdone::Any = default_differenTT_checkdone
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
        pc_cond_store = args.pc_cond_store,
        spectrum_store = args.spectrum_store,
        spd_tol = args.spd_tol
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
        numiter_store = args.numiter_store
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


"""
    ResidualObserver <: AbstractObserver

An observer that computes the relative global residual norm `||A * x - b||/||b||` 
and terminates the DMRG/ALS sweeps early if the residual falls below a given tolerance.

`A` and `b` can be left as `nothing`: the sweep-based `linearsolve` methods then bind them to the
operator and right-hand side of that call (always the unpreconditioned ones, never `M⁻¹`), so one
observer stored in `SweepingArgs` can be reused across solves with different right-hand sides.
Global solvers (`GlobalCG`, `GlobalGMRES`) do not use observers; they stop on their own `rtol`.
"""
Base.@kwdef struct ResidualObserver <: AbstractObserver
    A::Any = nothing
    b::Union{MPS, Nothing} = nothing
    rtol::Float64 = 1e-8
end

"""
    bind_observer(observer, A, b)

Return `observer` with any unset operator / right-hand side filled in from `A` and `b`. Observers
without such fields are returned unchanged. The input observer is never mutated.
"""
bind_observer(observer, A, b) = observer

function bind_observer(o::ResidualObserver, A, b)
    return ResidualObserver(isnothing(o.A) ? A : o.A, isnothing(o.b) ? b : o.b, o.rtol)
end

"""
    linearsolve_sweep_kwargs(solver, A, b, kwargs)

Merge the solver's `SweepingArgs` with the call-time `kwargs` and bind the observers to the
system `A x = b` of this call.
"""
function linearsolve_sweep_kwargs(solver, A, b, kwargs)
    sweep_kwargs = merge(NamedTuple(solver.sweep_kwargs), values(kwargs))
    return merge(sweep_kwargs, (;
        observer! = bind_observer(sweep_kwargs.observer!, A, b),
        sweep_observer! = bind_observer(sweep_kwargs.sweep_observer!, A, b)
    ))
end

"""
    precond_linearsolve_kwargs(solver, A, b, kwargs)

Split the call-time `kwargs` of a preconditioned sweep-based `linearsolve` into the local updater and
sweep keyword arguments.

The optional `preconditioner_sites` (default `nothing`: precondition every block) goes to the local
updater, which only applies `M⁻¹` to the blocks whose first site is in `preconditioner_sites`; the other
blocks are solved unpreconditioned. The remaining `kwargs` are merged with the solver's `SweepingArgs`
by `linearsolve_sweep_kwargs`.

# Returns
- `(updater_kwargs, sweep_kwargs)`
"""
function precond_linearsolve_kwargs(solver, A, b, kwargs)
    preconditioner_sites = get(kwargs, :preconditioner_sites, nothing)
    updater_kwargs = (; coefficients = (false, true), preconditioner_sites = preconditioner_sites, NamedTuple(solver.updater_kwargs)...)
    kw = Dict(kwargs...)
    delete!(kw, :preconditioner_sites)
    sweep_kwargs = linearsolve_sweep_kwargs(solver, A, b, pairs((; kw...)))
    return updater_kwargs, sweep_kwargs
end

function ITensorMPS.checkdone!(o::ResidualObserver; state, kwargs...)
    (isnothing(o.A) || isnothing(o.b)) && throw(ArgumentError("ResidualObserver has no operator or right-hand side bound; pass A and b explicitly or use it through a sweep-based `linearsolve`."))

    term1 = inner(o.A, state, o.A, state)
    term2 = -2.0 * real(inner(o.b', o.A, state))
    term3 = inner(o.b, o.b)

    res = sqrt(max(0.0, term1 + term2 + term3))
    rel_res = term3 > 0.0 ? res / sqrt(term3) : res

    if rel_res < o.rtol
        @info "Residual early truncation condition met: ||Ax-b||/||b|| = $(rel_res) < $(o.rtol)"
        return true
    end
    return false
end
