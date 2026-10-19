using KrylovKit: apply_scalartype, linselector

"""
    KrylovKit.linsolve(f, b, x₀, M⁻¹, a₀::Number=0, a₁::Number=1; kwargs...)

Preconditioned dispatch for KrylovKit's `linsolve`. Selects the appropriate Krylov algorithm
and, if CG is chosen, passes the preconditioner `M⁻¹` to the preconditioned CG implementation.
For non-CG algorithms, emits a warning and falls back to unpreconditioned `linsolve`.
"""
function KrylovKit.linsolve(f, b, x₀, M⁻¹, a₀::Number=0, a₁::Number=1; kwargs...)
    T = apply_scalartype(f, x₀, a₀, a₁)
    alg = linselector(f, b, T; kwargs...)
    if haskey(kwargs, :alg_rrule)
        alg_rrule = kwargs[:alg_rrule]
    else
        alg_rrule = alg
    end
    if alg isa CG || alg isa GMRES
        return linsolve(f, b, x₀, M⁻¹, alg, a₀, a₁; alg_rrule=alg_rrule)
    end
    @warn "Preconditioning is currently only supported for CG and GMRES. Skipping usage of M⁻¹"
    return linsolve(f, b, x₀, alg, a₀, a₁; alg_rrule=alg_rrule)
end
