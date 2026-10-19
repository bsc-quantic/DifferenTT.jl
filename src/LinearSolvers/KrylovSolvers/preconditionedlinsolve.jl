"""
    preconditioned_linsolve(f, b, x₀, M⁻¹, a₀::Number=0, a₁::Number=1; kwargs...)

Solve ``(a_0 + a_1 f) x = b`` with the preconditioner `M⁻¹`. We select the Krylov algorithm with KrylovKit's own
`linselector` and the keywords `kwargs` (`ishermitian`, `isposdef`, `tol`, …), as `KrylovKit.linsolve` does. If it
selects CG or GMRES, we run the preconditioned version of that algorithm; otherwise we warn and fall back to the
unpreconditioned `KrylovKit.linsolve`, ignoring `M⁻¹`.

This is a separate function rather than a method of `KrylovKit.linsolve`, because DifferenTT owns none of the
argument types and a method there would be type piracy.
"""
function preconditioned_linsolve(f, b, x₀, M⁻¹, a₀::Number=0, a₁::Number=1; kwargs...)
    T = apply_scalartype(f, x₀, a₀, a₁)
    alg = linselector(f, b, T; kwargs...)
    if haskey(kwargs, :alg_rrule)
        alg_rrule = kwargs[:alg_rrule]
    else
        alg_rrule = alg
    end
    if alg isa CG || alg isa GMRES
        return preconditioned_linsolve(f, b, x₀, M⁻¹, alg, a₀, a₁; alg_rrule=alg_rrule)
    end
    @warn "Preconditioning is currently only supported for CG and GMRES. Skipping usage of M⁻¹"
    return linsolve(f, b, x₀, alg, a₀, a₁; alg_rrule=alg_rrule)
end
