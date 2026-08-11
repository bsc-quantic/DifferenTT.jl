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

# Floor for κ = σ_max / max(σ_min, floor) and relative tolerance for SPD checks
# (truncation / roundoff can produce eigenvalues like -1e-136 that are not meaningful).
const _COND_EIG_FLOOR = 1e-16
const _SPD_EIG_RTOL = 1e-12

"""
    _matrix_condition(mat) -> Float64

2-norm condition number via singular values: κ₂ = σ_max / max(σ_min, ε).
More robust than λ_max/λ_min for non-Hermitian products (e.g. P⁻¹A).
"""
function _matrix_condition(mat::AbstractMatrix)::Float64
    σ = svdvals(mat)
    isempty(σ) && return NaN
    return Float64(σ[1] / max(σ[end], _COND_EIG_FLOOR))
end

"""
    _is_significantly_not_spd(λ_min, λ_max; rtol) -> Bool

True only if the most negative real eigenvalue is larger in magnitude than
`rtol * max(|λ_max|, |λ_min|, 1)`, filtering TT truncation noise.
"""
function _is_significantly_not_spd(λ_min::Real, λ_max::Real; rtol::Float64 = _SPD_EIG_RTOL)::Bool
    scale = max(abs(λ_max), abs(λ_min), 1.0)
    return λ_min < -rtol * scale
end

function _real_eig_extremes(mat::AbstractMatrix)
    eigs = real.(eigvals(mat))
    return minimum(eigs), maximum(eigs)
end

function (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    compute_cond = get(kwargs, :compute_cond, false)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)
    verbosity = get(kwargs, :verbosity, 0)
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
        minimum_eigval, maximum_eigval = _real_eig_extremes(op_mat)

        if _is_significantly_not_spd(minimum_eigval, maximum_eigval)
            @error "OPERATOR IS NOT SPD!" minimum_eigval maximum_eigval pos = problem.reduced_operator.lpos
        end
        cond_val = _matrix_condition(op_mat)
        if compute_cond && verbosity > 0
            @info "local cond" lpos = problem.reduced_operator.lpos minimum_eigval maximum_eigval cond = cond_val
        end
        if !isnothing(cond_store)
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
    verbosity = get(kwargs, :verbosity, 0)
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

    if compute_cond || !isnothing(cond_store) || !isnothing(pc_cond_store)
        op = contract(operator(problem.linear_problem))
        b_tensor = constant_term(problem.linear_problem)

        rowinds = commoninds(op, b_tensor)
        colinds = uniqueinds(op, b_tensor)
        rowdim = prod(dim.(rowinds))
        coldim = prod(dim.(colinds))

        op_mat = reshape(array(op, rowinds..., colinds...), rowdim, coldim)
        minimum_eigval_op, maximum_eigval_op = _real_eig_extremes(op_mat)

        pc = contract(problem.preconditioner)
        pc_mat = reshape(array(pc, rowinds..., colinds...), rowdim, coldim)
        minimum_eigval_pc, maximum_eigval_pc = _real_eig_extremes(pc_mat)

        product_mat = pc_mat * op_mat

        if _is_significantly_not_spd(minimum_eigval_op, maximum_eigval_op)
            @error "OPERATOR IS NOT SPD!" minimum_eigval_op maximum_eigval_op pos = problem.linear_problem.reduced_operator.lpos
        end

        if _is_significantly_not_spd(minimum_eigval_pc, maximum_eigval_pc)
            @error "Preconditioner IS NOT SPD!" minimum_eigval_pc maximum_eigval_pc pos = problem.linear_problem.reduced_operator.lpos
        end

        unprecond_cond = _matrix_condition(op_mat)
        precond_cond = _matrix_condition(product_mat)
        if compute_cond && verbosity > 0
            @info "local precond cond" lpos = problem.linear_problem.reduced_operator.lpos unprecond_cond precond_cond
        end
        if !isnothing(cond_store)
            record_krylov_cond!(cond_store, _krylov_lpos(problem), unprecond_cond)
        end
        if !isnothing(pc_cond_store)
            record_krylov_cond!(pc_cond_store, _krylov_lpos(problem), precond_cond)
        end
    end

    if !isnothing(numiter_store)
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), info.numiter)
    end

    return x, (; info, residual=info.residual)
end
