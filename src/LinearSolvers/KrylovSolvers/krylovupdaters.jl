using ITensorMPS: operator, constant_term, contract, ReducedLinearProblem
using KrylovKit: eigsolve, svdsolve

"""
    (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)

Local updater for `alternating_update` that solves the reduced linear system using KrylovKit's `linsolve`.

Dispatches to KrylovKit's algorithm selector (CG for Hermitian positive-definite, GMRES otherwise).
Optionally computes the condition number of the reduced operator when `compute_cond=true` is passed
via `kwargs`. The metric depends on the solver path selected by `ishermitian`/`isposdef`:
spectral κ for CG/PCG, 2-norm κ from singular values for GMRES.

# Returns
- `(x::ITensor, (; info))`: The solution tensor and a named tuple containing KrylovKit convergence info.
"""

struct KrylovUpdater <: AbstractUpdater end

# ---------------------------------------------------------------------------
# Condition metrics: CG/PCG (spectral) vs GMRES (singular values)
#
# KrylovKit `linselector`: CG iff `ishermitian && isposdef`, else GMRES.
#
# CG / PCG (SPD hypothesis on A and, if used, P⁻¹):
#   κ_PCG = λ_max(P^{-1/2} A P^{-1/2}) / λ_min(P^{-1/2} A P^{-1/2})
#   with spec(P^{-1/2} A P^{-1/2}) = spec(P^{-1} A).
# Extremes via eigmin/eigmax (dense) or Lanczos/eigsolve (large).
# For κ_PCG: Cholesky P⁻¹ = U'U, S = U A U', spec(S) = spec(P⁻¹ A) by XY↔YX.
# U is the Cholesky factor, not necessarily P^{-1/2}.
#
# GMRES (general / non-SPD):
#   κ₂(A) = σ_max(A) / σ_min(A), and κ₂(P⁻¹ A) for left preconditioning.
# Extremes without a full SVD:
#   1. Dense moderate n: LAPACK `svdvals` (all σ; accurate, O(n³) — fine for
#      local TT blocks; no stable LAPACK "svdmin/svdmax only" analogue).
#   2. Large n: Golub–Kahan–Lanczos bidiagonalization via KrylovKit
#      `svdsolve` (`:LR` / `:SR`) — literature-standard partial SVD.
# Do NOT hermitianize on this path: A may be genuinely non-symmetric.
#
# Size cutoff between dense and Krylov paths is an empirical engineering
# threshold, not a literature constant.
# ---------------------------------------------------------------------------

"""
    _uses_cg_metric(kwargs) -> Bool

True when KrylovKit would select CG (`ishermitian && isposdef`), so the
PCG spectral condition applies; otherwise GMRES → κ₂ via singular values.
"""
function _uses_cg_metric(kwargs)::Bool
    return Bool(get(kwargs, :ishermitian, false)) && Bool(get(kwargs, :isposdef, false))
end

@enum SpdStatus::Int8 begin
    SPD_DEFINITE = 1      # λ_min > +tol  → SPD → κ is defined
    SPD_AMBIGUOUS = 0     # |λ_min| ≤ tol → numerically semidefinite / unclear
    SPD_INDEFINITE = -1   # λ_min < -tol  → not SPD → no κ to report
end

const _SPD_EIG_RTOL = 1e-12
# Empirical cutoff: dense LAPACK selected-eigenvalue path vs Lanczos.
const _EXTREMAL_KRYLOV_THRESHOLD = 256

"""
    _hermitianize(mat) -> Hermitian

The local tensor contraction is theoretically symmetric; take the symmetric
part `(mat + mat')/2` only to remove floating-point / TT-truncation asymmetry.
This must not be used to turn a genuinely non-symmetric operator into an SPD
one and then claim κ of the original matrix.
"""
function _hermitianize(mat::AbstractMatrix)
    return Hermitian((mat .+ mat') ./ 2)
end

"""
    _spd_tolerance(λ_min, λ_max; rtol) -> Float64

Relative spectral tolerance used to separate definite / ambiguous / indefinite.
"""
function _spd_tolerance(λ_min::Real, λ_max::Real; rtol::Float64 = _SPD_EIG_RTOL)::Float64
    return rtol * max(abs(Float64(λ_max)), abs(Float64(λ_min)), 1.0)
end

"""
    _classify_spd(λ_min, λ_max; rtol) -> SpdStatus

- `SPD_DEFINITE`:   λ_min > +tol  → safe to form κ = λ_max / λ_min
- `SPD_AMBIGUOUS`:  |λ_min| ≤ tol → truncation may have pushed a tiny eigenvalue
                    through zero; do not invent a condition number
- `SPD_INDEFINITE`: λ_min < -tol  → SPD hypothesis failed; do not report κ
"""
function _classify_spd(λ_min::Real, λ_max::Real; rtol::Float64 = _SPD_EIG_RTOL)::SpdStatus
    if !(isfinite(λ_min) && isfinite(λ_max))
        return SPD_AMBIGUOUS
    end
    tol = _spd_tolerance(λ_min, λ_max; rtol=rtol)
    if λ_min > tol
        return SPD_DEFINITE
    elseif λ_min < -tol
        return SPD_INDEFINITE
    else
        return SPD_AMBIGUOUS
    end
end

"""
    _is_significantly_not_spd(λ_min, λ_max; rtol) -> Bool

True iff `_classify_spd` returns `SPD_INDEFINITE`.
"""
function _is_significantly_not_spd(λ_min::Real, λ_max::Real; rtol::Float64 = _SPD_EIG_RTOL)::Bool
    return _classify_spd(λ_min, λ_max; rtol=rtol) === SPD_INDEFINITE
end

"""
    _spectral_condition(λ_min, λ_max) -> Float64

PCG-style spectral condition κ = λ_max / λ_min, only when the spectrum is
classified as strictly positive-definite. Otherwise returns `NaN` (no silent
flooring of λ_min; a failed SPD hypothesis is not a huge κ).
"""
function _spectral_condition(λ_min::Real, λ_max::Real)::Float64
    if _classify_spd(λ_min, λ_max) !== SPD_DEFINITE
        return NaN
    end
    return Float64(λ_max) / Float64(λ_min)
end

"""
    _extremal_hermitian_eigvals(H::Hermitian) -> (λ_min, λ_max)

Compute only the smallest and largest eigenvalues of a Hermitian matrix.

- `n ≤ _EXTREMAL_KRYLOV_THRESHOLD` (empirical): LAPACK via `eigmin`/`eigmax`.
- larger `n`: Lanczos / `eigsolve` (`:SR` / `:LR`).
"""
function _extremal_hermitian_eigvals(H::Hermitian; krylov_threshold::Int = _EXTREMAL_KRYLOV_THRESHOLD)
    n = size(H, 1)
    if n == 0
        return (NaN, NaN)
    elseif n == 1
        λ = Float64(real(H[1, 1]))
        return (λ, λ)
    elseif n <= krylov_threshold
        return (Float64(eigmin(H)), Float64(eigmax(H)))
    else
        krylovdim = min(40, n)
        λmax_vals, = eigsolve(H, 1, :LR; ishermitian=true, krylovdim=krylovdim)
        λmin_vals, = eigsolve(H, 1, :SR; ishermitian=true, krylovdim=krylovdim)
        return (Float64(real(λmin_vals[1])), Float64(real(λmax_vals[1])))
    end
end

"""
    _extremal_eigvals(mat) -> (λ_min, λ_max)

Extremal eigenvalues after `_hermitianize` (for theoretically symmetric A / P⁻¹).
"""
function _extremal_eigvals(mat::AbstractMatrix)
    return _extremal_hermitian_eigvals(_hermitianize(mat))
end

"""

"""

"""
    _report_spectral_κ(λ_min, λ_max; label, lpos) -> Float64

Classify SPD status, emit diagnostics, and return κ or `NaN`.
"""
function _report_spectral_κ(
    λ_min::Real,
    λ_max::Real;
    label::AbstractString,
    lpos,
)::Float64
    status = _classify_spd(λ_min, λ_max)
    if status === SPD_INDEFINITE
        @error "$(label) IS NOT SPD — κ undefined" λ_min λ_max lpos
        return NaN
    elseif status === SPD_AMBIGUOUS
        @warn "$(label) is numerically semidefinite/ambiguous — κ undefined" λ_min λ_max lpos
        return NaN
    end
    return _spectral_condition(λ_min, λ_max)
end

# ---------------------------------------------------------------------------
# Extremal singular values & κ₂ (GMRES-relevant)
# ---------------------------------------------------------------------------

"""
    _extremal_singular_values(mat) -> (σ_min, σ_max)

Only the extreme singular values of a general (possibly non-symmetric) matrix.

- Moderate dense size: LAPACK `svdvals` (full σ vector; accurate for local blocks).
- Large size: Golub–Kahan–Lanczos via `svdsolve` (`:SR` / `:LR`).
"""
function _extremal_singular_values(
    mat::AbstractMatrix;
    krylov_threshold::Int = _EXTREMAL_KRYLOV_THRESHOLD,
)
    m, n = size(mat)
    if m == 0 || n == 0
        return (NaN, NaN)
    elseif m == 1 && n == 1
        σ = abs(Float64(mat[1, 1]))
        return (σ, σ)
    elseif max(m, n) <= krylov_threshold
        s = svdvals(mat)
        return (Float64(minimum(s)), Float64(maximum(s)))
    else
        kdim = min(40, min(m, n))
        σmax_vals, = svdsolve(mat, 1, :LR; krylovdim=kdim)
        σmin_vals, = svdsolve(mat, 1, :SR; krylovdim=kdim)
        return (Float64(σmin_vals[1]), Float64(σmax_vals[1]))
    end
end

"""
    _svd_condition(σ_min, σ_max) -> Float64

2-norm condition κ₂ = σ_max / σ_min. Returns `Inf` if σ_min == 0 and σ_max > 0
(singular operator); `NaN` if the extremes are not usable.
"""
function _svd_condition(σ_min::Real, σ_max::Real)::Float64
    if !(isfinite(σ_min) && isfinite(σ_max)) || σ_max < 0 || σ_min < 0
        return NaN
    elseif σ_min == 0
        return σ_max > 0 ? Inf : NaN
    end
    return Float64(σ_max) / Float64(σ_min)
end

"""
    _local_operator_matrix(op, b_tensor) -> (A, lpos_helper...)

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

function _matrixfree_eigvals(f, x0; ishermitian::Bool)
    λmax_vals, = eigsolve(f, x0, 1, :LR; ishermitian=ishermitian)
    λmin_vals, = eigsolve(f, x0, 1, :SR; ishermitian=ishermitian)
    return (Float64(real(λmin_vals[1])), Float64(real(λmax_vals[1])))
end

function _apply_fai_operator(P::ReducedFactorizedPreconditioner, A, v)
    v_primed = ITensors.prime(v)
    transpose_itensor_map = _itensor_map(P.reduced_op)
    if ITensors.dim(first(transpose_itensor_map)) == 1
        reverse!(transpose_itensor_map)
    end
    w = v_primed
    for it in transpose_itensor_map
        w *= it
    end
    Aw = A(w)
    return ITensors.noprime(contract(P.reduced_op, Aw))
end

function _compute_and_record_cond!(problem::ReducedLinearProblem, b_tensor, use_cg, compute_cond, verbosity, cond_store, pc_cond_store)
    lpos = problem.reduced_operator.lpos
    if use_cg
        op_func = v -> operator(problem)(v)
        λmin_A, λmax_A = _matrixfree_eigvals(op_func, b_tensor; ishermitian=true)
        κ_A = _report_spectral_κ(λmin_A, λmax_A; label="OPERATOR", lpos=lpos)
        if compute_cond && verbosity > 0
            @info "local cond (CG/spectral)" lpos λmin_A λmax_A κ_A
        end
        if !isnothing(cond_store) && isfinite(κ_A)
            record_krylov_cond!(cond_store, _krylov_lpos(problem), κ_A)
        end
    else
        op = contract(operator(problem))
        A, = _local_dense_matrix(op, b_tensor)
        σmin_A, σmax_A = _extremal_singular_values(A)
        κ_A = _svd_condition(σmin_A, σmax_A)
        if compute_cond && verbosity > 0
            @info "local cond (GMRES/κ₂)" lpos σmin_A σmax_A κ_A
        end
        if !isnothing(cond_store) && isfinite(κ_A)
            record_krylov_cond!(cond_store, _krylov_lpos(problem), κ_A)
        end
    end
end

# TODO Look for a way to compute convergence metrics for GMRES also from the ProjMPO obejcts, if possible
function _compute_and_record_cond!(problem::ReducedPrecondLinearProblem, b_tensor, use_cg, compute_cond, verbosity, cond_store, pc_cond_store)
    lpos = problem.linear_problem.reduced_operator.lpos
    if use_cg
        # Get the callable ProjMPO object from the linear problem
        op_func = v -> operator(problem.linear_problem)(v)
        λmin_A, λmax_A = _matrixfree_eigvals(op_func, b_tensor; ishermitian=true)
        κ_A = _report_spectral_κ(λmin_A, λmax_A; label="OPERATOR", lpos=lpos)

        # Get the callable ProjMPO object from the preconditioner
        pc_func = v -> problem.preconditioner(v)
        λmin_Pinv, λmax_Pinv = _matrixfree_eigvals(pc_func, b_tensor; ishermitian=true)
        status_Pinv = _classify_spd(λmin_Pinv, λmax_Pinv)
        if status_Pinv === SPD_INDEFINITE
            @error "Preconditioner P⁻¹ IS NOT SPD — κ_PCG undefined" λmin_Pinv λmax_Pinv lpos
        elseif status_Pinv === SPD_AMBIGUOUS
            @warn "Preconditioner P⁻¹ is numerically semidefinite/ambiguous — κ_PCG undefined" λmin_Pinv λmax_Pinv lpos
        end

        κ_PC, λmin_PA, λmax_PA = NaN, NaN, NaN
        if status_Pinv === SPD_DEFINITE && isfinite(κ_A)
            # Apply first the ProjMPO from the linear problem and then the one from the preconditioner
            pc_op_func = v -> problem.preconditioner(operator(problem.linear_problem)(v))
            λmin_PA, λmax_PA = _matrixfree_eigvals(pc_op_func, b_tensor; ishermitian=false)
            κ_PC = _report_spectral_κ(λmin_PA, λmax_PA; label="PCG operator S∼P⁻¹A", lpos=lpos)
        end

        if compute_cond && verbosity > 0
            @info "local precond cond (PCG/spectral)" lpos κ_A κ_PC λmin_A λmax_A λmin_Pinv λmax_Pinv λmin_PA λmax_PA
        end
        if !isnothing(cond_store) && isfinite(κ_A)
            record_krylov_cond!(cond_store, _krylov_lpos(problem), κ_A)
        end
        if !isnothing(pc_cond_store) && isfinite(κ_PC)
            record_krylov_cond!(pc_cond_store, _krylov_lpos(problem), κ_PC)
        end
    else
        # For GMRES, proceed with the dense matrices, as we do not have an object tracking the adjoint of the operator
        op = contract(operator(problem.linear_problem))
        b_tensor_for_matrix = constant_term(problem.linear_problem)
        A, rowinds, colinds, rowdim, coldim = _local_dense_matrix(op, b_tensor_for_matrix)
        pc = contract(problem.preconditioner)
        Pinv = reshape(array(pc, rowinds..., colinds...), rowdim, coldim)

        σmin_A, σmax_A = _extremal_singular_values(A)
        κ_A = _svd_condition(σmin_A, σmax_A)

        PA = Matrix(Pinv) * Matrix(A)
        σmin_PA, σmax_PA = _extremal_singular_values(PA)
        κ_PC = _svd_condition(σmin_PA, σmax_PA)

        if compute_cond && verbosity > 0
            @info "local precond cond (GMRES/κ₂)" lpos κ_A κ_PC σmin_A σmax_A σmin_PA σmax_PA
        end
        if !isnothing(cond_store) && isfinite(κ_A)
            record_krylov_cond!(cond_store, _krylov_lpos(problem), κ_A)
        end
        if !isnothing(pc_cond_store) && isfinite(κ_PC)
            record_krylov_cond!(pc_cond_store, _krylov_lpos(problem), κ_PC)
        end
    end
end

function _compute_and_record_cond!(problem::ReducedFAIPrecondLinearProblem, b_tensor, use_cg, compute_cond, verbosity, cond_store, pc_cond_store)
    lpos = problem.linear_problem.reduced_operator.lpos
    if use_cg
        op_func = v -> operator(problem.linear_problem)(v)
        λmin_A, λmax_A = _matrixfree_eigvals(op_func, b_tensor; ishermitian=true)
        κ_A = _report_spectral_κ(λmin_A, λmax_A; label="OPERATOR", lpos=lpos)

        pc_func = v -> problem.preconditioner(v)
        λmin_Pinv, λmax_Pinv = _matrixfree_eigvals(pc_func, b_tensor; ishermitian=true)
        status_Pinv = _classify_spd(λmin_Pinv, λmax_Pinv)
        if status_Pinv === SPD_INDEFINITE
            @error "Preconditioner P⁻¹ IS NOT SPD — κ_PCG undefined" λmin_Pinv λmax_Pinv lpos
        elseif status_Pinv === SPD_AMBIGUOUS
            @warn "Preconditioner P⁻¹ is numerically semidefinite/ambiguous — κ_PCG undefined" λmin_Pinv λmax_Pinv lpos
        end

        κ_PC, λmin_PA, λmax_PA = NaN, NaN, NaN
        if status_Pinv === SPD_DEFINITE && isfinite(κ_A)
            A = operator(problem.linear_problem)
            # for fai operators, the _apply_fai_operator computes Gl * A * Gl' (v), which is guaranteed to be hermitian
            fai_op_func = v -> _apply_fai_operator(problem.preconditioner, A, v)
            λmin_PA, λmax_PA = _matrixfree_eigvals(fai_op_func, b_tensor; ishermitian=true)
            κ_PC = _report_spectral_κ(λmin_PA, λmax_PA; label="PCG operator S∼P⁻¹A", lpos=lpos)
        end

        if compute_cond && verbosity > 0
            @info "local precond cond (PCG/spectral)" lpos κ_A κ_PC λmin_A λmax_A λmin_Pinv λmax_Pinv λmin_PA λmax_PA
        end
        if !isnothing(cond_store) && isfinite(κ_A)
            record_krylov_cond!(cond_store, _krylov_lpos(problem), κ_A)
        end
        if !isnothing(pc_cond_store) && isfinite(κ_PC)
            record_krylov_cond!(pc_cond_store, _krylov_lpos(problem), κ_PC)
        end
    else
        op = contract(operator(problem.linear_problem))
        b_tensor_for_matrix = constant_term(problem.linear_problem)
        A, rowinds, colinds, rowdim, coldim = _local_dense_matrix(op, b_tensor_for_matrix)
        pc = contract(problem.preconditioner)
        Pinv = reshape(array(pc, rowinds..., colinds...), rowdim, coldim)

        σmin_A, σmax_A = _extremal_singular_values(A)
        κ_A = _svd_condition(σmin_A, σmax_A)

        PA = Matrix(Pinv) * Matrix(A)
        σmin_PA, σmax_PA = _extremal_singular_values(PA)
        κ_PC = _svd_condition(σmin_PA, σmax_PA)

        if compute_cond && verbosity > 0
            @info "local precond cond (GMRES/κ₂)" lpos κ_A κ_PC σmin_A σmax_A σmin_PA σmax_PA
        end
        if !isnothing(cond_store) && isfinite(κ_A)
            record_krylov_cond!(cond_store, _krylov_lpos(problem), κ_A)
        end
        if !isnothing(pc_cond_store) && isfinite(κ_PC)
            record_krylov_cond!(pc_cond_store, _krylov_lpos(problem), κ_PC)
        end
    end
end

"""
    (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    (up::KrylovUpdater)(problem::Union{ReducedPrecondLinearProblem, ReducedFAIPrecondLinearProblem}, init; internal_kwargs, coefficients, kwargs...)

Local updater for linear problems. Passes the preconditioner `ProjMPO` to KrylovKit's
`linsolve` (PCG when `ishermitian && isposdef`, else preconditioned GMRES).

Condition metrics:
- CG/PCG: spectral κ(A) and κ_PCG from S = UAU' (Cholesky of P⁻¹).
- GMRES: κ₂(A) and κ₂(P⁻¹ A) from extremal singular values (no SPD requirement on P⁻¹).

# Returns
- `(x::ITensor, (; info, residual))`: The solution tensor, KrylovKit info, and residual norm.
"""
function (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    compute_cond = get(kwargs, :compute_cond, false)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)
    verbosity = get(kwargs, :verbosity, 0)
    use_cg = _uses_cg_metric(kwargs)
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
        b_tensor = constant_term(problem)
        _compute_and_record_cond!(problem, b_tensor, use_cg, compute_cond, verbosity, cond_store, nothing)
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
    verbosity = get(kwargs, :verbosity, 0)
    use_cg = _uses_cg_metric(kwargs)
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
        b_tensor = constant_term(problem.linear_problem)
        _compute_and_record_cond!(problem, b_tensor, use_cg, compute_cond, verbosity, cond_store, pc_cond_store)
    end

    if !isnothing(numiter_store)
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), info.numiter)
    end

    return x, (; info, residual=info.residual)
end