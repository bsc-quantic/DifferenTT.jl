using ITensorMPS: operator, constant_term, contract, ReducedLinearProblem

"""
    (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)

Local updater for `alternating_update` that solves the reduced linear system using KrylovKit's `linsolve`.

Dispatches to KrylovKit's algorithm selector (CG for Hermitian positive-definite, GMRES otherwise).
Optionally computes the condition number κ_λ = max|λ| / min|λ| of the reduced operator from its
dense local spectrum when `compute_cond=true` is passed via `kwargs`.

# Returns
- `(x::ITensor, (; info))`: The solution tensor and a named tuple containing KrylovKit convergence info.
"""
struct KrylovUpdater <: AbstractUpdater end

# ---------------------------------------------------------------------------
# Condition number metric: κ_λ(M) = max|λ(M)| / min|λ(M)|, from the full dense spectrum (LAPACK).
#
# The same metric is used for both solver paths, so `cond_store` values are comparable:
# - SPD A (CG): κ_λ = κ₂ = λ_max / λ_min, the quantity in the CG convergence bound.
# - P⁻¹A with A, P⁻¹ SPD (PCG): P⁻¹A is similar to U A Uᵀ (P⁻¹ = UᵀU), so its spectrum is real and
#   positive and κ_λ is the PCG convergence metric (κ₂(P⁻¹A) is not).
# - Non-SPD (GMRES): κ_λ is well defined whenever M is nonsingular and equals κ₂ for normal M.
#   For non-normal M it is only a lower bound on κ₂, and GMRES convergence depends on the whole
#   spectrum distribution and on the eigenvector conditioning, so it is a heuristic indicator there.
#   GMRES is left-preconditioned (see gmres.jl), so the relevant operator is P⁻¹A.
#
# Dense eigendecomposition is O(n³) in the local block size; it only runs when requested.
# ---------------------------------------------------------------------------

"""
    _uses_cg_metric(kwargs) -> Bool

True when KrylovKit would select CG (`ishermitian && isposdef`), in which case the local operators
are expected to be SPD and a warning is emitted if their spectrum says otherwise.
"""
function _uses_cg_metric(kwargs)::Bool
    return Bool(get(kwargs, :ishermitian, false)) && Bool(get(kwargs, :isposdef, false))
end

"""
    _eig_condition(M) -> (κ, λ)

Full dense spectrum `λ = eigvals(M)` (LAPACK) and κ_λ = max|λ| / min|λ| (`Inf` if singular).
Equals the spectral κ for SPD / PCG operators; a lower bound on κ₂ for non-normal ones.
"""
function _eig_condition(M::AbstractMatrix)
    λ = eigvals(M)
    λabs_min, λabs_max = extrema(abs, λ)
    return λabs_max / λabs_min, λ
end

"""
    _local_dense_matrix(op, b_tensor) -> (A, rowinds, colinds, rowdim, coldim)

Contract indices shared with `b_tensor` into a dense matrix.
"""
function _local_dense_matrix(op, b_tensor)
    rowinds = commoninds(op, b_tensor)
    colinds = uniqueinds(op, b_tensor)
    rowdim = prod(dim.(rowinds))
    coldim = prod(dim.(colinds))
    A = reshape(array(op, rowinds..., colinds...), rowdim, coldim)
    return A, rowinds, colinds, rowdim, coldim
end

function _warn_if_not_spd(λ, label, lpos)
    λmin_real = minimum(real, λ)
    λmin_real <= 0 && @warn "$(label) is not SPD" λmin_real lpos
end

function _record_cond!(store, problem, κ)
    if !isnothing(store) && isfinite(κ)
        record_krylov_cond!(store, _krylov_lpos(problem), κ)
    end
end

# A non-preconditioned problem has no κ_PC of its own. When `pc_cond_store` is given (a block of a preconditioned
# solve that skips the preconditioner, see `preconditioner_sites`), κ(A) is recorded in it as well, so that
# `pc_cond_store` holds an entry for every visited block and stays aligned with `cond_store`.
function _compute_and_record_cond!(problem::ReducedLinearProblem, use_cg, compute_cond, verbosity, cond_store, pc_cond_store)
    lpos = _krylov_lpos(problem)
    A, = _local_dense_matrix(contract(operator(problem)), constant_term(problem))
    κ_A, λ_A = _eig_condition(A)

    if compute_cond
        use_cg && _warn_if_not_spd(λ_A, "Operator A", lpos)
        verbosity > 0 && @info "local cond" lpos κ_A
    end
    _record_cond!(cond_store, problem, κ_A)
    _record_cond!(pc_cond_store, problem, κ_A)
end

# The dense matrices are the transposes of the operators (rows are the unprimed indices), so the dense product is
# (A P⁻¹)ᵀ, which is similar to P⁻¹A: the spectrum is the same.
function _compute_and_record_cond!(problem::Union{ReducedPrecondLinearProblem, ReducedFAIPrecondLinearProblem}, use_cg, compute_cond, verbosity, cond_store, pc_cond_store)
    lpos = _krylov_lpos(problem)
    linear_problem = problem.linear_problem
    A, rowinds, colinds, rowdim, coldim = _local_dense_matrix(contract(operator(linear_problem)), constant_term(linear_problem))
    # For the FAI preconditioner, contract returns the dense Glᵀ Gl
    Pinv = reshape(array(contract(problem.preconditioner), rowinds..., colinds...), rowdim, coldim)

    κ_A, λ_A = _eig_condition(A)
    κ_PC, λ_PA = _eig_condition(Pinv * A)

    if compute_cond
        if use_cg
            _warn_if_not_spd(λ_A, "Operator A", lpos)
            # If A is SPD, P⁻¹A has a non-positive eigenvalue iff P⁻¹ is not SPD (Sylvester's law of inertia)
            _warn_if_not_spd(λ_PA, "Preconditioned operator P⁻¹A", lpos)
        end
        verbosity > 0 && @info "local precond cond" lpos κ_A κ_PC
    end
    _record_cond!(cond_store, problem, κ_A)
    _record_cond!(pc_cond_store, problem, κ_PC)
end

"""
    (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    (up::KrylovUpdater)(problem::Union{ReducedPrecondLinearProblem, ReducedFAIPrecondLinearProblem}, init; internal_kwargs, coefficients, kwargs...)

Local updater for linear problems. Passes the preconditioner `ProjMPO` to KrylovKit's
`linsolve` (PCG when `ishermitian && isposdef`, else preconditioned GMRES).

If `preconditioner_sites` is given, the preconditioner is only applied to the blocks whose first site
(`lpos + 1`) is in it; the other blocks are solved unpreconditioned. For those blocks, the condition number
of the unpreconditioned local problem is recorded in both `cond_store` and `pc_cond_store`.

Condition metrics: κ_λ = max|λ| / min|λ| of the dense local A and P⁻¹A. This is the exact spectral κ for
CG/PCG. For GMRES it is a heuristic indicator only: it is a lower bound on κ₂ for non-normal operators, and
GMRES convergence also depends on the eigenvector conditioning and on the spectrum distribution.

# Returns
- `(x::ITensor, (; info, residual))`: The solution tensor, KrylovKit info, and residual norm.
"""
function (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    compute_cond = get(kwargs, :compute_cond, false)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)
    verbosity = get(kwargs, :verbosity, 0)
    use_cg = _uses_cg_metric(kwargs)
    kwargs = filter(p -> first(p) ∉ (:compute_cond, :numiter_store, :cond_store, :pc_cond_store), kwargs)

    x, info = linsolve(
        operator(problem),
        constant_term(problem),
        init,
        coefficients[1],
        coefficients[2];
        kwargs...,
    )

    if compute_cond || !isnothing(cond_store)
        _compute_and_record_cond!(problem, use_cg, compute_cond, verbosity, cond_store, nothing)
    end

    if !isnothing(numiter_store)
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), info.numiter)
    end

    return x, (; info)
end

# Since now we leverage multiple dispatch at the _compute_and_record_cond level, we can use the same function
# for both types of preconditioned linear problems
function (up::KrylovUpdater)(problem::Union{ReducedPrecondLinearProblem, ReducedFAIPrecondLinearProblem}, init; internal_kwargs, coefficients, kwargs...)
    compute_cond = get(kwargs, :compute_cond, false)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)
    pc_cond_store = get(kwargs, :pc_cond_store, nothing)
    preconditioner_sites = get(kwargs, :preconditioner_sites, nothing)
    verbosity = get(kwargs, :verbosity, 0)
    use_cg = _uses_cg_metric(kwargs)
    kwargs = filter(p -> first(p) ∉ (:compute_cond, :numiter_store, :cond_store, :pc_cond_store, :preconditioner_sites), kwargs)

    use_preconditioner = true
    if !isnothing(preconditioner_sites)
        lpos = _krylov_lpos(problem)
        use_preconditioner = (lpos + 1) in preconditioner_sites
    end

    if use_preconditioner
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
            _compute_and_record_cond!(problem, use_cg, compute_cond, verbosity, cond_store, pc_cond_store)
        end
    else
        x, info = linsolve(
            operator(problem.linear_problem),
            constant_term(problem.linear_problem),
            init,
            coefficients[1],
            coefficients[2];
            kwargs...,
        )

        if compute_cond || !isnothing(cond_store) || !isnothing(pc_cond_store)
            # The unpreconditioned κ(A) is also recorded as this block's κ_PC (the effective preconditioner is the identity)
            _compute_and_record_cond!(problem.linear_problem, use_cg, compute_cond, verbosity, cond_store, pc_cond_store)
        end
    end

    if !isnothing(numiter_store)
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), info.numiter)
    end

    return x, (; info, residual=info.residual)
end
