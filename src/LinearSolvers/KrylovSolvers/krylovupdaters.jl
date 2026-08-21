using ITensorMPS: operator, constant_term, contract, ReducedLinearProblem
using KrylovKit: svdsolve

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
# Condition number metrics for Krylov linear solvers:
#
# KrylovKit uses:
# - Conjugate Gradient (CG) for Hermitian, positive-definite operators
#   (`ishermitian` && `isposdef`);
# - Generalized Minimum Residual (GMRES) otherwise.
#
# For CG/PCG (SPD case):
#   Spectral condition number: κ(A) = λ_max(A) / λ_min(A)
#   For preconditioning (SPD P): κ_PCG = λ_max(P⁻¹A) / λ_min(P⁻¹A)
#   Although P⁻¹A isn't symmetric, its eigenvalues are real/positive if A, P⁻¹ are SPD.
#   Extremal eigenvalues always come from a dense diagonalization of the contracted
#   local matrix (`eigvals`, then min/max of the real parts).
#
# For GMRES (general/non-SPD case):
#   2-norm condition: κ₂(A) = σ_max(A) / σ_min(A) (σ = singular values)
#   For preconditioning, use κ₂(P⁻¹A). No symmetry/SPD assumption.
#   Extremal singular values:
#     - Small/moderate: Full SVD via LAPACK's `svdvals` (costly but accurate)
#     - Large: Partial SVD via KrylovKit's `svdsolve` (`:LR`/`:SR`)
#
# `_EXTREMAL_KRYLOV_THRESHOLD` only switches dense vs Krylov SVD (GMRES).
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
# Empirical cutoff for GMRES κ₂: dense LAPACK `svdvals` vs KrylovKit `svdsolve`.
const _EXTREMAL_KRYLOV_THRESHOLD = 256

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
    _dense_extremal_eigvals(mat) -> (λ_min, λ_max)

Full dense diagonalization (`eigvals`) then min/max of the real parts.

This is the CG/PCG spectral path: always dense, never Lanczos. `P⁻¹A` need not be
Hermitian, so the general eigenproblem is used and only the real parts are kept.
"""
function _dense_extremal_eigvals(mat::AbstractMatrix)::Tuple{Float64, Float64}
    m, n = size(mat)
    if m == 0 || n == 0
        return (NaN, NaN)
    end
    λ = eigvals(mat)
    return (Float64(minimum(real.(λ))), Float64(maximum(real.(λ))))
end

"""
    _report_spectral_κ(λ_min, λ_max; label, lpos) -> Float64

Classify SPD status, emit diagnostics, and return κ or `NaN`.
"""
function _report_spectral_κ(
    λ_min::Real,
    λ_max::Real;
    label::AbstractString,
    lpos,
    compute_cond::Bool=true
)::Float64
    status = _classify_spd(λ_min, λ_max)
    if status === SPD_INDEFINITE
        compute_cond && @error "$(label) IS NOT SPD — κ undefined" λ_min λ_max lpos
        return NaN
    elseif status === SPD_AMBIGUOUS
        compute_cond && @warn "$(label) is numerically semidefinite/ambiguous — κ undefined" λ_min λ_max lpos
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

function _compute_and_record_cond!(problem::ReducedLinearProblem, b_tensor, use_cg, compute_cond, verbosity, cond_store, pc_cond_store)
    lpos = problem.reduced_operator.lpos
    op = contract(operator(problem))
    A, = _local_dense_matrix(op, b_tensor)
    if use_cg
        λmin_A, λmax_A = _dense_extremal_eigvals(A)
        κ_A = _report_spectral_κ(λmin_A, λmax_A; label="OPERATOR", lpos=lpos, compute_cond=compute_cond)
        if compute_cond && verbosity > 0
            @info "local cond (CG/spectral)" lpos λmin_A λmax_A κ_A
        end
        if !isnothing(cond_store) && isfinite(κ_A)
            record_krylov_cond!(cond_store, _krylov_lpos(problem), κ_A)
        end
    else
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

function _compute_and_record_cond!(
    problem::Union{ReducedPrecondLinearProblem, ReducedFAIPrecondLinearProblem},
    b_tensor,
    use_cg,
    compute_cond,
    verbosity,
    cond_store,
    pc_cond_store,
)
    lpos = problem.linear_problem.reduced_operator.lpos
    op = contract(operator(problem.linear_problem))
    A, rowinds, colinds, rowdim, coldim = _local_dense_matrix(op, b_tensor)
    pc = contract(problem.preconditioner)
    Pinv = reshape(array(pc, rowinds..., colinds...), rowdim, coldim)
    PA = Matrix(Pinv) * Matrix(A)

    if use_cg
        λmin_A, λmax_A = _dense_extremal_eigvals(A)
        κ_A = _report_spectral_κ(λmin_A, λmax_A; label="OPERATOR", lpos=lpos, compute_cond=compute_cond)

        λmin_Pinv, λmax_Pinv = _dense_extremal_eigvals(Pinv)
        status_Pinv = _classify_spd(λmin_Pinv, λmax_Pinv)
        if status_Pinv === SPD_INDEFINITE
            compute_cond && @error "Preconditioner P⁻¹ IS NOT SPD — κ_PCG undefined" λmin_Pinv λmax_Pinv lpos
        elseif status_Pinv === SPD_AMBIGUOUS
            compute_cond && @warn "Preconditioner P⁻¹ is numerically semidefinite/ambiguous — κ_PCG undefined" λmin_Pinv λmax_Pinv lpos
        end

        κ_PC, λmin_PA, λmax_PA = NaN, NaN, NaN
        if status_Pinv === SPD_DEFINITE && isfinite(κ_A)
            λmin_PA, λmax_PA = _dense_extremal_eigvals(PA)
            κ_PC = _report_spectral_κ(λmin_PA, λmax_PA; label="PCG operator P⁻¹A", lpos=lpos, compute_cond=compute_cond)
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
        σmin_A, σmax_A = _extremal_singular_values(A)
        κ_A = _svd_condition(σmin_A, σmax_A)

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
- CG/PCG: spectral κ(A) and κ(P⁻¹A) from dense `eigvals` (min/max of real parts).
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