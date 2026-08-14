import ITensorMPS: apply

"""
    TTCG <: KrylovKit.KrylovAlgorithm

Tensor Train Conjugate Gradient (CG) algorithm for linear systems.
This is a helper struct in order for KrylovKit to know which algo + parameters to run. Includes truncation parameters 
to keep bond dimensions bounded.
"""
Base.@kwdef struct TTCG <: KrylovKit.KrylovAlgorithm
    maxiter::Int = 100
    tol::Float64 = 1e-8
    verbosity::Int = 0
    maxdim::Int = typemax(Int)
    cutoff::Float64 = 1e-8
    orth_maxdim::Int = typemax(Int)
    orth_cutoff::Float64 = 1e-12
end

"""
    GlobalCG <: AbstractGlobalSolver

Global Conjugate Gradient (CG) solver for linear systems on the TT-manifold.
"""
Base.@kwdef struct GlobalCG <: AbstractGlobalSolver
    updater_kwargs::GlobalUpdaterArgs = GlobalUpdaterArgs(isposdef=true, ishermitian=true)

    function GlobalCG(updater_kwargs)
        if !updater_kwargs.isposdef || !updater_kwargs.ishermitian
            throw(ArgumentError("GlobalCG requires updater_kwargs.isposdef and updater_kwargs.ishermitian to be true."))
        end
        return new(updater_kwargs)
    end
end

function linearsolve(solver::GlobalCG, A::MPO, b::MPS, x0::Union{MPS, Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0

    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))
    
    alg = TTCG(; maxiter = merged_kwargs.maxiter,
                 tol = merged_kwargs.rtol,
                 verbosity = merged_kwargs.verbosity,
                 maxdim = get(merged_kwargs, :maxdim, typemax(Int)),
                 cutoff = get(merged_kwargs, :cutoff, 1e-8),
                 orth_maxdim = get(merged_kwargs, :orth_maxdim, typemax(Int)),
                 orth_cutoff = get(merged_kwargs, :orth_cutoff, 1e-12)
    )
                     
    x, info = KrylovKit.linsolve(A, b, init, alg)
    if !isnothing(merged_kwargs.numiter_store)
        merged_kwargs.numiter_store[] = info.numiter
    end
    return x
end

function linearsolve(solver::GlobalCG, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS, Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0

    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))
    
    alg = TTCG(; maxiter = merged_kwargs.maxiter,
                 tol = merged_kwargs.rtol,
                 verbosity = merged_kwargs.verbosity,
                 maxdim = get(merged_kwargs, :maxdim, typemax(Int)),
                 cutoff = get(merged_kwargs, :cutoff, 1e-8),
                 orth_maxdim = get(merged_kwargs, :orth_maxdim, typemax(Int)),
                 orth_cutoff = get(merged_kwargs, :orth_cutoff, 1e-12)
)
                     
    x, info = KrylovKit.linsolve(A, b, init, M⁻¹, alg)
    if !isnothing(merged_kwargs.numiter_store)
        merged_kwargs.numiter_store[] = info.numiter
    end
    return x
end


"""
    TTGMRES <: KrylovKit.KrylovAlgorithm

Tensor Train Generalized Minimal Residual Method (GMRES) algorithm for linear systems.
This is a helper struct in order for KrylovKit to know which algo + parameters to run. Includes truncation parameters 
for application and orthogonalization steps.
"""
Base.@kwdef struct TTGMRES <: KrylovKit.KrylovAlgorithm
    krylovdim::Int = 30
    maxiter::Int = 100
    tol::Float64 = 1e-8
    verbosity::Int = 0
    maxdim::Int = typemax(Int)
    cutoff::Float64 = 1e-8
    orth_maxdim::Int = typemax(Int)
    orth_cutoff::Float64 = 1e-12
end

"""
    GlobalGMRES <: AbstractGlobalSolver

Global Generalized Minimal Residual Method (GMRES) solver for linear systems on the TT-manifold.
"""
Base.@kwdef struct GlobalGMRES <: AbstractGlobalSolver
    updater_kwargs::GlobalUpdaterArgs = GlobalUpdaterArgs()
end

function linearsolve(solver::GlobalGMRES, A::MPO, b::MPS, x0::Union{MPS, Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0

    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))
    
    alg = TTGMRES(; krylovdim = merged_kwargs.krylovdim,
                    maxiter = merged_kwargs.maxiter,
                    tol = merged_kwargs.rtol,
                    verbosity = merged_kwargs.verbosity,
                    maxdim = get(merged_kwargs, :maxdim, typemax(Int)),
                    cutoff = get(merged_kwargs, :cutoff, 1e-8),
                    orth_maxdim = get(merged_kwargs, :orth_maxdim, typemax(Int)),
                    orth_cutoff = get(merged_kwargs, :orth_cutoff, 1e-12))
                    
    x, info = KrylovKit.linsolve(A, b, init, alg)
    if !isnothing(merged_kwargs.numiter_store)
        merged_kwargs.numiter_store[] = info.numiter
    end
    return x
end

function linearsolve(solver::GlobalGMRES, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS, Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0

    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))
    
    alg = TTGMRES(; krylovdim = merged_kwargs.krylovdim,
                    maxiter = merged_kwargs.maxiter,
                    tol = merged_kwargs.rtol,
                    verbosity = merged_kwargs.verbosity,
                    maxdim = get(merged_kwargs, :maxdim, typemax(Int)),
                    cutoff = get(merged_kwargs, :cutoff, 1e-8),
                    orth_maxdim = get(merged_kwargs, :orth_maxdim, typemax(Int)),
                    orth_cutoff = get(merged_kwargs, :orth_cutoff, 1e-12))
                        
    x, info = KrylovKit.linsolve(A, b, init, M⁻¹, alg)
    if !isnothing(merged_kwargs.numiter_store)
        merged_kwargs.numiter_store[] = info.numiter
    end
    return x
end

#Helper functions in order to handle preconditioners in the TTCG, TTGMRES pipeline efficiently
function apply(P::Preconditioner, v::MPS; kwargs...)
    return apply(P.operator, v; kwargs...)
end

function apply(P::FactorizedPreconditioner, v::MPS; kwargs...)
    v_out = apply(P.operator, v; kwargs...)
    return apply(transpose(P.operator), v_out; kwargs...)
end

function ITensors.inner(v::MPS, P::Preconditioner, w::MPS)
    return ITensorMPS.inner(v', P.operator, w)
end

function KrylovKit.inner(v::MPS, P::Preconditioner, w::MPS)
    return ITensorMPS.inner(v', P.operator, w)
end

function ITensors.inner(v::MPS, P::FactorizedPreconditioner, w::MPS; kwargs...)
    return ITensorMPS.inner(P.operator, v, P.operator, w)
end

function KrylovKit.inner(v::MPS, P::FactorizedPreconditioner, w::MPS; kwargs...)
    return ITensorMPS.inner(P.operator, v, P.operator, w)
end
