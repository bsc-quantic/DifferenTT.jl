"""
    (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)

Local updater for `alternating_update` that solves the reduced linear system using KrylovKit's `linsolve`.

Dispatches to KrylovKit's algorithm selector (CG for Hermitian positive-definite, GMRES otherwise).
Optionally computes the condition number κ_λ = max|λ| / min|λ| of the reduced operator from its
dense local spectrum, when a `cond_store` or an `spd_tol` is passed via `kwargs`.

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

"""
    NotSPDError(label, lpos, λmin, λmax_abs)

Thrown by the local SPD checks of a CG solve when `spd_tol` is set and a local operator has an eigenvalue with
real part below `-spd_tol * max|λ|`. It aborts the sweep, so callers can stop a solve as soon as a non-SPD local
operator (e.g. a preconditioner that is not positive definite) is found.
"""
struct NotSPDError <: Exception
    label::String
    lpos::Int
    λmin::Float64
    λmax_abs::Float64
end

function Base.showerror(io::IO, e::NotSPDError)
    print(
        io,
        "NotSPDError: $(e.label) is not SPD at lpos = $(e.lpos): min Re λ = $(e.λmin), ",
        "min Re λ / max|λ| = $(e.λmin / e.λmax_abs)",
    )
end

# Throws a `NotSPDError` when the most negative real part of the spectrum is below -spd_tol * max|λ|, i.e. clearly
# away from the rounding level around 0. Without `spd_tol` there is nothing to check: the sign information of every
# block is recorded by `spectrum_store` instead (see `_local_spectrum`).
_check_spd(λ, label, lpos, ::Nothing) = nothing

function _check_spd(λ, label, lpos, spd_tol::Real)
    λmin_real = minimum(real, λ)
    λmax_abs = maximum(abs, λ)
    λmin_real < -spd_tol * λmax_abs && throw(NotSPDError(label, lpos, λmin_real, λmax_abs))
    return nothing
end

# The dense local spectrum is only computed when one of its consumers is attached: a condition-number store records
# κ, `spd_tol` checks the sign of the spectrum. `verbosity` only logs what was computed.
function _wants_cond(cond_store, pc_cond_store, spd_tol)
    !isnothing(cond_store) || !isnothing(pc_cond_store) || !isnothing(spd_tol)
end

function _record_cond!(store, problem, κ)
    if !isnothing(store) && isfinite(κ)
        record_krylov_cond!(store, _krylov_lpos(problem), κ)
    end
end

# A non-preconditioned problem has no κ_PC of its own. When `pc_cond_store` is given (a block of a preconditioned
# solve that skips the preconditioner, see `preconditioner_sites`), κ(A) is recorded in it as well, so that
# `pc_cond_store` holds an entry for every visited block and stays aligned with `cond_store`.
function _compute_and_record_cond!(
    problem::ReducedLinearProblem, use_cg, verbosity, cond_store, pc_cond_store, spd_tol=nothing
)
    lpos = _krylov_lpos(problem)
    A, = _local_dense_matrix(dense_projected_operator(operator(problem)), constant_term(problem))
    κ_A, λ_A = _eig_condition(A)

    use_cg && _check_spd(λ_A, "Operator A", lpos, spd_tol)
    verbosity > 0 && @info "local cond" lpos κ_A
    _record_cond!(cond_store, problem, κ_A)
    _record_cond!(pc_cond_store, problem, κ_A)
end

# The dense matrices are the transposes of the operators (rows are the unprimed indices), so the dense product is
# (A P⁻¹)ᵀ, which is similar to P⁻¹A: the spectrum is the same.
function _compute_and_record_cond!(
    problem::Union{ReducedPrecondLinearProblem,ReducedFAIPrecondLinearProblem},
    use_cg,
    verbosity,
    cond_store,
    pc_cond_store,
    spd_tol=nothing,
)
    lpos = _krylov_lpos(problem)
    linear_problem = problem.linear_problem
    A, rowinds, colinds, rowdim, coldim = _local_dense_matrix(
        dense_projected_operator(operator(linear_problem)), constant_term(linear_problem)
    )
    # For the FAI preconditioner, contract returns the dense Glᵀ Gl
    Pinv = reshape(array(contract(problem.preconditioner), rowinds..., colinds...), rowdim, coldim)

    κ_A, λ_A = _eig_condition(A)
    κ_PC, λ_PA = _eig_condition(Pinv * A)

    if use_cg
        _check_spd(λ_A, "Operator A", lpos, spd_tol)
        # If A is SPD, P⁻¹A has a non-positive eigenvalue iff P⁻¹ is not SPD (Sylvester's law of inertia)
        _check_spd(λ_PA, "Preconditioned operator P⁻¹A", lpos, spd_tol)
    end
    verbosity > 0 && @info "local precond cond" lpos κ_A κ_PC
    _record_cond!(cond_store, problem, κ_A)
    _record_cond!(pc_cond_store, problem, κ_PC)
end

# ---------------------------------------------------------------------------
# Sign-aware local spectra (`spectrum_store`). The local blocks are congruences VᵀAV, VᵀM⁻¹V of the global
# operators (V the basis of the current projection), so by Sylvester's law of inertia a negative eigenvalue of the
# symmetric part of the local M⁻¹ certifies that the global M⁻¹ is not positive definite. The converse does not
# hold: a positive local block only says that M⁻¹ is positive on the span of V.
# ---------------------------------------------------------------------------

const _EMPTY_LOCAL_SPECTRUM = (;
    eigmin_A=NaN,
    eigmax_A=NaN,
    asym_A=NaN,
    eigmin_P=NaN,
    eigmax_P=NaN,
    nneg_P=-1,
    asym_P=NaN,
    eigmin_PA=NaN,
    eigmax_PA=NaN,
    nneg_PA=-1,
)

_relative_asymmetry(M::AbstractMatrix) = norm(M - transpose(M)) / norm(M)

# Negative eigenvalues beyond the rounding level of a dense eigensolver
_count_negative(λ) = count(<(-length(λ) * eps(Float64) * maximum(abs, λ)), λ)

"""
    _local_spectrum(A, Pinv) -> NamedTuple

Sign-aware spectra of the dense local operator `A` and, if given, of the local preconditioner `Pinv`. Only the
symmetric parts S_A, S_P are diagonalized, since they set the sign of the quadratic forms. When S_A = LLᵀ is SPD,
the spectrum of S_P S_A (the PCG operator if `Pinv` is symmetric) is that of the symmetric Lᵀ S_P L, which has the
inertia of S_P.
"""
function _local_spectrum(A::AbstractMatrix, Pinv::Union{Nothing,AbstractMatrix})
    SA = Symmetric((A + transpose(A)) / 2)
    λA = eigvals(SA)
    spectrum = merge(_EMPTY_LOCAL_SPECTRUM, (; eigmin_A=first(λA), eigmax_A=last(λA), asym_A=_relative_asymmetry(A)))
    isnothing(Pinv) && return spectrum

    SP = Symmetric((Pinv + transpose(Pinv)) / 2)
    λP = eigvals(SP)
    spectrum = merge(
        spectrum,
        (; eigmin_P=first(λP), eigmax_P=last(λP), nneg_P=_count_negative(λP), asym_P=_relative_asymmetry(Pinv)),
    )

    chol = cholesky(SA; check=false)
    issuccess(chol) || return spectrum
    L = chol.L
    λPA = eigvals(Symmetric(transpose(L) * Matrix(SP) * L))
    return merge(spectrum, (; eigmin_PA=first(λPA), eigmax_PA=last(λPA), nneg_PA=_count_negative(λPA)))
end

_record_spectrum!(::Nothing, problem, info) = nothing

function _record_spectrum!(store::KrylovSpectrumStats, problem::ReducedLinearProblem, info)
    b = constant_term(problem)
    spectrum = nothing
    if dim(b) <= store.max_blockdim
        A, = _local_dense_matrix(dense_projected_operator(operator(problem)), b)
        spectrum = _local_spectrum(A, nothing)
    end
    record_krylov_spectrum!(store, _krylov_lpos(problem), dim(b), info, spectrum)
end

function _record_spectrum!(
    store::KrylovSpectrumStats, problem::Union{ReducedPrecondLinearProblem,ReducedFAIPrecondLinearProblem}, info
)
    linear_problem = problem.linear_problem
    b = constant_term(linear_problem)
    spectrum = nothing
    if dim(b) <= store.max_blockdim
        A, rowinds, colinds, rowdim, coldim = _local_dense_matrix(dense_projected_operator(operator(linear_problem)), b)
        Pinv = reshape(array(contract(problem.preconditioner), rowinds..., colinds...), rowdim, coldim)
        spectrum = _local_spectrum(A, Pinv)
    end
    record_krylov_spectrum!(store, _krylov_lpos(problem), dim(b), info, spectrum)
end

"""
    (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    (up::KrylovUpdater)(problem::Union{ReducedPrecondLinearProblem, ReducedFAIPrecondLinearProblem}, init; internal_kwargs, coefficients, kwargs...)

Local updater for linear problems. Passes the preconditioner `ProjMPO` to KrylovKit's
`linsolve` (PCG when `ishermitian && isposdef`, else preconditioned GMRES).

If `preconditioner_sites` is given, the preconditioner is only applied to the blocks whose first site
(`lpos + 1`) is in it; the other blocks are solved unpreconditioned. For those blocks, the condition number
of the unpreconditioned local problem is recorded in both `cond_store` and `pc_cond_store`.

If a `KrylovSpectrumStats` is passed as `spectrum_store`, the sign-aware local spectra of every visited block are
recorded too (see `_local_spectrum`).

The dense local spectra are computed only when a consumer is attached: `cond_store` / `pc_cond_store` record the
condition numbers, and `spd_tol` (CG only, `ishermitian && isposdef`) checks the local A and P⁻¹A: an eigenvalue whose
real part is below `-spd_tol * max|λ|` throws a `NotSPDError`, which stops the solve. With `verbosity > 0` the
computed condition numbers are also logged.

Condition metrics: κ_λ = max|λ| / min|λ| of the dense local A and P⁻¹A. This is the exact spectral κ for
CG/PCG. For GMRES it is a heuristic indicator only: it is a lower bound on κ₂ for non-normal operators, and
GMRES convergence also depends on the eigenvector conditioning and on the spectrum distribution.

# Returns

  - `(x::ITensor, (; info, residual))`: The solution tensor, KrylovKit info, and residual norm.
"""
function (up::KrylovUpdater)(problem::ReducedLinearProblem, init; internal_kwargs, coefficients, kwargs...)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)
    verbosity = get(kwargs, :verbosity, 0)
    spectrum_store = get(kwargs, :spectrum_store, nothing)
    spd_tol = get(kwargs, :spd_tol, nothing)
    use_cg = _uses_cg_metric(kwargs)
    kwargs = filter(p -> first(p) ∉ (:numiter_store, :cond_store, :pc_cond_store, :spectrum_store, :spd_tol), kwargs)

    x, info = linsolve(operator(problem), constant_term(problem), init, coefficients[1], coefficients[2]; kwargs...)

    if _wants_cond(cond_store, nothing, spd_tol)
        _compute_and_record_cond!(problem, use_cg, verbosity, cond_store, nothing, spd_tol)
    end
    _record_spectrum!(spectrum_store, problem, info)

    if !isnothing(numiter_store)
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), info.numiter)
    end

    return x, (; info)
end

# Since now we leverage multiple dispatch at the _compute_and_record_cond level, we can use the same function
# for both types of preconditioned linear problems
function (up::KrylovUpdater)(
    problem::Union{ReducedPrecondLinearProblem,ReducedFAIPrecondLinearProblem},
    init;
    internal_kwargs,
    coefficients,
    kwargs...,
)
    numiter_store = get(kwargs, :numiter_store, nothing)
    cond_store = get(kwargs, :cond_store, nothing)
    pc_cond_store = get(kwargs, :pc_cond_store, nothing)
    preconditioner_sites = get(kwargs, :preconditioner_sites, nothing)
    verbosity = get(kwargs, :verbosity, 0)
    spectrum_store = get(kwargs, :spectrum_store, nothing)
    spd_tol = get(kwargs, :spd_tol, nothing)
    use_cg = _uses_cg_metric(kwargs)
    kwargs = filter(
        p -> first(p) ∉ (:numiter_store, :cond_store, :pc_cond_store, :spectrum_store, :spd_tol, :preconditioner_sites),
        kwargs,
    )

    use_preconditioner = true
    if !isnothing(preconditioner_sites)
        lpos = _krylov_lpos(problem)
        use_preconditioner = (lpos + 1) in preconditioner_sites
    end

    if use_preconditioner
        x, info = preconditioned_linsolve(
            operator(problem.linear_problem),
            constant_term(problem.linear_problem),
            init,
            problem.preconditioner,
            coefficients[1],
            coefficients[2];
            kwargs...,
        )

        if _wants_cond(cond_store, pc_cond_store, spd_tol)
            _compute_and_record_cond!(problem, use_cg, verbosity, cond_store, pc_cond_store, spd_tol)
        end
        _record_spectrum!(spectrum_store, problem, info)
    else
        x, info = linsolve(
            operator(problem.linear_problem),
            constant_term(problem.linear_problem),
            init,
            coefficients[1],
            coefficients[2];
            kwargs...,
        )

        if _wants_cond(cond_store, pc_cond_store, spd_tol)
            # The unpreconditioned κ(A) is also recorded as this block's κ_PC (the effective preconditioner is the identity)
            _compute_and_record_cond!(problem.linear_problem, use_cg, verbosity, cond_store, pc_cond_store, spd_tol)
        end
        _record_spectrum!(spectrum_store, problem.linear_problem, info)
    end

    if !isnothing(numiter_store)
        record_krylov_numiter!(numiter_store, _krylov_lpos(problem), info.numiter)
    end

    return x, (; info, residual=info.residual)
end
