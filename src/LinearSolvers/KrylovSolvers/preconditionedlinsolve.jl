"""
    preconditioned_linsolve(f, b, x₀, M⁻¹, a₀::Number=0, a₁::Number=1; kwargs...) -> (x, info)

Solve ``(a_0 + a_1 f)\\mathbf{x} = \\mathbf{b}`` from `x₀` with the preconditioner `M⁻¹`, the preconditioned counterpart
of `KrylovKit.linsolve(f, b, x₀, a₀, a₁; kwargs...)`.

We select the Krylov algorithm with KrylovKit's own `linselector` and the keywords `kwargs` (`ishermitian`, `isposdef`,
`rtol`, `atol`, `krylovdim`, `maxiter`, `verbosity`), as `KrylovKit.linsolve` does: CG if `ishermitian && isposdef`,
restarted GMRES otherwise, with the tolerance ``\\max(\\texttt{atol}, \\texttt{rtol}\\, \\|\\mathbf{b}\\|)``. We then
run the preconditioned version of that algorithm, PCG or left-preconditioned GMRES (see the methods with an explicit
`alg`). If KrylovKit selects another algorithm, we warn and fall back to the unpreconditioned `KrylovKit.linsolve`,
ignoring `M⁻¹`. This is what [`KrylovUpdater`](@ref) calls for a preconditioned local problem.

This is a function of its own rather than a method of `KrylovKit.linsolve`, because DifferenTT owns none of the
argument types and such a method would be type piracy.

Returns the solution and a `KrylovKit.ConvergenceInfo`.
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
