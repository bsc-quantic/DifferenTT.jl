using ITensorMPS: operator, constant_term, contract, ReducedLinearProblem

"""
    (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)

Local updater for `alternating_update` that solves the reduced linear system using KrylovKit's `linsolve`.

Dispatches to KrylovKit's algorithm selector (CG for Hermitian positive-definite, GMRES otherwise).
Optionally computes the condition number of the reduced operator when `compute_cond=true` is passed
via `kwargs`.

# Returns
- `(x::ITensor, (; info))`: The solution tensor and a named tuple containing KrylovKit convergence info.
"""

struct KrylovUpdater <: AbstractUpdater end

function (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    compute_cond = get(kwargs, :compute_cond, false)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)
    kwargs = filter(p -> first(p) != :compute_cond && first(p) != :numiter_store && first(p) != :cond_store && first(p) != :pc_cond_store, kwargs)

    x, info = linsolve(
        operator(problem),
        constant_term(problem),
        init,
        coefficients[1],
        coefficients[2];
        kwargs...,
    )

    if compute_cond || !isnothing(cond_store)
        op = contract(operator(problem))
        b_tensor = constant_term(problem)

        rowinds = commoninds(op, b_tensor)
        colinds = uniqueinds(op, b_tensor)
        rowdim = prod(dim.(rowinds))
        coldim = prod(dim.(colinds))

        op_mat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)
        eigs_op = eigvals(op_mat)

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
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), info.numiter)
    end

    return x, (; info)
end

"""
    (up::KrylovUpdater)(problem::ReducedPrecondLinearProblem, init; internal_kwargs, coefficients, kwargs...)

Local updater for preconditioned problems. Passes the preconditioner `ProjMPO` to KrylovKit's
`linsolve`, enabling preconditioned CG within the sweep.

# Returns
- `(x::ITensor, (; info, residual))`: The solution tensor, KrylovKit info, and residual norm.
"""
function (up::KrylovUpdater)(problem::Union{ReducedFAIPrecondLinearProblem,ReducedPrecondLinearProblem}, init; internal_kwargs, coefficients, kwargs...)
    compute_cond = get(kwargs, :compute_cond, false)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)
    pc_cond_store = get(kwargs, :pc_cond_store, nothing)
    kwargs = filter(p -> first(p) != :compute_cond && first(p) != :numiter_store && first(p) != :cond_store && first(p) != :pc_cond_store, kwargs)

    x, info = linsolve(
        operator(problem.linear_problem),
        constant_term(problem.linear_problem),
        init,
        problem.preconditioner,
        coefficients[1],
        coefficients[2];
        kwargs...,
    )

    if compute_cond || (!isnothing(cond_store) && !isnothing(pc_cond_store))
        op = contract(operator(problem.linear_problem))
        b_tensor = constant_term(problem.linear_problem)

        rowinds = commoninds(op, b_tensor)
        colinds = uniqueinds(op, b_tensor)
        rowdim = prod(dim.(rowinds))
        coldim = prod(dim.(colinds))

        op_mat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)
        eigs_op = eigvals(op_mat)

        minimum_eigval_op = minimum(real.(eigs_op))
        maximum_eigval_op = maximum(real.(eigs_op))

        pc = contract(problem.preconditioner)
        pc_mat = reshape(array(pc, rowinds..., colinds...), rowdim, coldim)
        eigs_pc = eigvals(pc_mat)

        minimum_eigval_pc = minimum(real.(eigs_pc))
        maximum_eigval_pc = maximum(real.(eigs_pc))

        product_mat = pc_mat * op_mat
        eigs_prod = eigvals(product_mat)

        minimum_eigval_prod = minimum(real.(eigs_prod))
        maximum_eigval_prod = maximum(real.(eigs_prod))

        if minimum_eigval_op < 0
            @error "OPERATOR IS NOT SPD!" minimum_eigval_op pos = problem.linear_problem.reduced_operator.lpos
        end

        if minimum_eigval_pc < 0
            @error "Preconditioner IS NOT SPD!" minimum_eigval_pc pos = problem.linear_problem.reduced_operator.lpos
        end
        if compute_cond
            @show problem.linear_problem.reduced_operator.lpos, (maximum_eigval_op / minimum_eigval_op),  (maximum_eigval_prod / minimum_eigval_prod)
        end
        if !isnothing(cond_store)
            unprecond_cond = abs(maximum_eigval_op) / max(abs(minimum_eigval_op), 1e-16)
            precond_cond = abs(maximum_eigval_prod) / max(abs(minimum_eigval_prod), 1e-16)
            cond_improvement = unprecond_cond / precond_cond
            record_krylov_cond!(cond_store, _krylov_lpos(problem), unprecond_cond)
        end
        if !isnothing(pc_cond_store)
            unprecond_cond = abs(maximum_eigval_op) / max(abs(minimum_eigval_op), 1e-16)
            precond_cond = abs(maximum_eigval_prod) / max(abs(minimum_eigval_prod), 1e-16)
            cond_improvement = unprecond_cond / precond_cond
            record_krylov_cond!(pc_cond_store, _krylov_lpos(problem), precond_cond)
        end
    end

    if !isnothing(numiter_store)
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), info.numiter)
    end

    return x, (; info, residual=info.residual)
end