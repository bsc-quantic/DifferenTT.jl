using ITensorMPS: operator, constant_term, contract, ReducedLinearProblem

"""
    qr_updater(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)

Local updater for `alternating_update` that solves the reduced linear system via QR decomposition.

Contracts the projected operator and constant term into dense matrices, forms the shifted system
`(a₀ I + a₁ A) x = b` using `coefficients = (a₀, a₁)`, and solves it via QR factorization.

# Returns
- `(x::ITensor, (; residual))`: The solution tensor and a named tuple with residual info.
"""
function qr_updater(
    problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...
)
    verbosity = get(kwargs, :verbosity, 0)
    compute_cond = get(kwargs, :compute_cond, false)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)

    op = contract(operator(problem))
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

    if compute_cond || !isnothing(cond_store)
        eigs_op = eigvals(Amat)
        minimum_eigval = minimum(real.(eigs_op))
        maximum_eigval = maximum(real.(eigs_op))

        if minimum_eigval < 0
            @error "OPERATOR IS NOT SPD!" minimum_eigval pos = problem.reduced_operator.lpos
        end
        if compute_cond
            @show problem.reduced_operator.lpos, minimum_eigval, maximum_eigval
        end
        if !isnothing(cond_store)
            cond_val = abs(maximum_eigval) / max(abs(minimum_eigval), 1e-16)
            record_krylov_cond!(cond_store, _krylov_lpos(problem), cond_val)
        end
    end

    if !isnothing(numiter_store)
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), 1)
    end

    info = (; numiter=1, residual=rel_residual)
    return noprime(ITensor(x, colinds...)), (; info, residual=rel_residual)
end