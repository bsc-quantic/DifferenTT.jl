"""
    QRUpdater() <: AbstractUpdater

Local solver that solves each local problem exactly, by a dense QR factorization.

We contract the projected operator into a dense ``n_{\\text{loc}} \\times n_{\\text{loc}}`` matrix, where
``n_{\\text{loc}}`` is the dimension of the local tensor, and solve
``(a_0 + a_1 A_{\\text{loc}}) \\mathbf{x}_{\\text{loc}} = \\mathbf{b}_{\\text{loc}}`` by QR, at a cost of
``\\mathcal{O}(n_{\\text{loc}}^3)``. Since ``n_{\\text{loc}} = \\chi^2 d^2`` for a 2-site problem with bond dimension
``\\chi`` and site dimension ``d``, this is only affordable for small bond dimensions, where it serves as an exact
reference for the Krylov solves. It works on unpreconditioned local problems only, and reads `verbosity`,
`numiter_store` (which records one iteration per solve) and `cond_store` from [`LocalUpdaterArgs`](@ref).
"""
struct QRUpdater <: AbstractUpdater end

"""
    (up::QRUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients,
                    kwargs...) -> (x, (; info, residual))

Solve the local problem `problem` by QR, with `coefficients = (a₀, a₁)`. Return the local tensor and a `NamedTuple`
whose `info` has `numiter = 1` and whose `residual` is the relative residual of the dense solve.
"""
function (up::QRUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    verbosity = get(kwargs, :verbosity, 0)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)

    op = dense_projected_operator(operator(problem))
    b = constant_term(problem)

    rowinds = commoninds(op, b)
    colinds = uniqueinds(op, b)

    rowdim = prod(dim.(rowinds))
    coldim = prod(dim.(colinds))

    bvec = reshape(array(b, rowinds...), rowdim)
    Amat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)

    shifted_Amat = coefficients[1] * I + coefficients[2] * Amat # we are solving (α₁ + α₂A)x = b

    decomp_Amat = qr(shifted_Amat)
    x = decomp_Amat \ bvec
    rel_residual = norm(shifted_Amat * x - bvec) / norm(bvec)
    verbosity > 0 && @info "Qr updater got relative residual norm = $(rel_residual)"

    if !isnothing(cond_store)
        # Same metric as KrylovUpdater: κ_λ = max|λ| / min|λ| of the dense local operator
        cond_val, _ = _eig_condition(Amat)
        verbosity > 0 && @info "local cond" lpos = problem.reduced_operator.lpos cond_val
        _record_cond!(cond_store, problem, cond_val)
    end

    if !isnothing(numiter_store)
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), 1)
    end

    info = (; numiter=1, residual=rel_residual)
    return noprime(ITensor(x, colinds...)), (; info, residual=rel_residual)
end
