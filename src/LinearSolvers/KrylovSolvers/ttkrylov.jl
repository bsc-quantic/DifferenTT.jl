using KrylovKit

"""
    TTCG <: KrylovKit.KrylovAlgorithm

Tensor Train Conjugate Gradient (CG) solver for linear systems.
Includes truncation parameters to keep bond dimensions bounded.
"""
Base.@kwdef struct TTCG <: KrylovKit.KrylovAlgorithm
    maxiter::Int = 100
    tol::Float64 = 1e-8
    verbosity::Int = 0
    maxdim::Int = typemax(Int)
    cutoff::Float64 = 1e-8
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

function linearsolve(solver::GlobalCG, A, b, x0; kwargs...)
    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))
    
    alg = TTCG(; maxiter = merged_kwargs.maxiter,
                 tol = merged_kwargs.rtol,
                 verbosity = merged_kwargs.verbosity,
                 maxdim = get(merged_kwargs, :maxdim, typemax(Int)),
                 cutoff = get(merged_kwargs, :cutoff, 1e-8)
    )
                 
    M_inv = get(merged_kwargs, :preconditioner, v -> v)
    
    x, info = KrylovKit.linsolve(A, b, x0, M_inv, alg)
    return x, (; info, residual=info.residual)
end


"""
    TTGMRES <: KrylovKit.KrylovAlgorithm

Tensor Train Generalized Minimal Residual Method (GMRES) solver for linear systems.
Includes truncation parameters for application and orthogonalization steps.
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

function linearsolve(solver::GlobalGMRES, A, b, x0; kwargs...)
    merged_kwargs = merge(NamedTuple(solver.updater_kwargs), values(kwargs))
    
    alg = TTGMRES(; krylovdim = merged_kwargs.krylovdim,
                    maxiter = merged_kwargs.maxiter,
                    tol = merged_kwargs.rtol,
                    verbosity = merged_kwargs.verbosity,
                    maxdim = get(merged_kwargs, :maxdim, typemax(Int)),
                    cutoff = get(merged_kwargs, :cutoff, 1e-8),
                    orth_maxdim = get(merged_kwargs, :orth_maxdim, typemax(Int)),
                    orth_cutoff = get(merged_kwargs, :orth_cutoff, 1e-12))
                    
    M_inv = get(merged_kwargs, :preconditioner, v -> v)
    
    x, info = KrylovKit.linsolve(A, b, x0, M_inv, alg)
    return x, (; info, residual=info.residual)
end
