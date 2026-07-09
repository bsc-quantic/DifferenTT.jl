"""
    GlobalCG <: AbstractGlobalSolver

Global Conjugate Gradient (CG) solver for linear systems on the TT-manifold.
"""
Base.@kwdef struct GlobalCG <: AbstractGlobalSolver
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs(isposdef=true, ishermitian=true)

    function GlobalCG(updater_kwargs)
        if !updater_kwargs.isposdef || !updater_kwargs.ishermitian
            throw(ArgumentError("GlobalCG requires updater_kwargs.isposdef and updater_kwargs.ishermitian to be true."))
        end
        return new(updater_kwargs)
    end
end

function linearsolve(solver::GlobalCG, A, b, x0; kwargs...)
    # TODO: Implement TT-CG
    error("GlobalCG solver not yet implemented")
end

"""
    GlobalGMRES <: AbstractGlobalSolver

Global Generalized Minimal Residual Method (GMRES) solver for linear systems on the TT-manifold.
"""
Base.@kwdef struct GlobalGMRES <: AbstractGlobalSolver
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
end

function linearsolve(solver::GlobalGMRES, A, b, x0; kwargs...)
    # TODO: Implement TT-GMRES
    error("GlobalGMRES solver not yet implemented")
end
