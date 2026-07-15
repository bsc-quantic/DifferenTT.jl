using ITensorMPS: ITensorMPS, AbstractProjMPO, ReducedLinearProblem
using ITensorMPS: set_nsite!, position!, nsite, site_range, noiseterm
using ITensorMPS: TDVPOrder, sub_time_steps, orderings
using ITensors: ITensors, ITensor, Index, inds, dag, prime, noprime, dim
using ITensors: commoninds, uniqueinds, factorize, svd, array, order
using ITensors: replaceind, OneITensor
using LinearAlgebra: norm, svd as la_svd

"""
    ResidualALS <: AbstractDMRGSolver

Standard Alternating Linear Scheme (DMRG) for solving linear systems.
"""
Base.@kwdef struct ResidualALS <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    updater::AbstractUpdater = KrylovUpdater()
    residual_tol::Real = 1e-12
    residual_damp::Real = 1.1
end

function linearsolve(solver::ResidualALS, A, b, x0; kwargs...)
    # TODO: Implement Standard Alternating Linear Scheme based on residual truncation
    error("Residual truncation ALS solver not yet implemented")
end

function linearsolve(solver::ResidualALS, A, b, M⁻¹, x0; kwargs...)
    # TODO: Implement Preconditioned Alternating Linear Scheme based on residual truncation
    error("Residual truncation ALS solver not yet implemented")
end

"""
    residual_replacebond!(state::MPS, b::Int, phi::ITensor, reduced_operator; kwargs...)

Replace a bond in the state MPS using residual-based truncation.
Walks down from full rank to find the smallest rank that still keeps the relative residual norm
below a certain threshold (`resid_tol`), while applying a Frobenius cutoff as a floor.
"""
function residual_replacebond!(
    state::MPS,
    b::Int,
    phi::ITensor,
    reduced_operator;  # AbstractProjMPO already positioned
    maxdim::Int = typemax(Int),
    mindim::Int = 1,
    cutoff::Float64 = 0.0,    # Frobenius cutoff (fallback floor)
    normalize::Bool = false,
    ortho::String = "left",
    resid_tol::Float64 = 1e-8,
    resid_damp::Float64 = 2.0,
)
    # ── 1. Full SVD ──────────────────────────────────────────────────
    indsMb = inds(state[b])
    U_full, S_full, V_full = svd(phi, indsMb; alg="recursive")

    bond_ind = commoninds(U_full, S_full)
    full_rank = dim(bond_ind[1])

    # ── 2. Compute RHS (projected constant term) ─────────────────────
    rhs = _local_rhs(reduced_operator)
    norm_rhs = norm(rhs)

    if norm_rhs ≈ 0
        # Trivial case – just do standard truncation, to avoid /0
        spec = replacebond!(state, b, phi; maxdim, mindim, cutoff, ortho, normalize)
        return spec
    end

    # ── 3. Residual of the untruncated solution ──────────────────────
    res_exact = _local_residual_norm(reduced_operator, phi, rhs) / norm_rhs

    # ── 4. Residual-based rank selection ─────────────────────────────
    # Walk from full rank downward; stop when residual exceeds threshold
    threshold = max(resid_tol * resid_damp, res_exact * resid_damp)

    # Extract singular values as a vector for fast iteration
    s_inds = inds(S_full)
    s_dim = dim(s_inds[1])
    svals = [S_full[s_inds[1] => k, s_inds[2] => k] for k in 1:s_dim]

    chosen_rank = full_rank
    for r in (full_rank - 1):-1:max(1, mindim)
        # Reconstruct truncated phi
        phi_trunc = _truncated_phi(U_full, S_full, V_full, r)
        res_trunc = _local_residual_norm(reduced_operator, phi_trunc, rhs) / norm_rhs
        if res_trunc > threshold
            chosen_rank = r + 1  # previous rank was still OK
            break
        end
        chosen_rank = r
    end

    # Clamp
    chosen_rank = clamp(chosen_rank, mindim, min(maxdim, full_rank))

    # ── 5. Also apply Frobenius cutoff as a floor ────────────────────
    # (in case the user still wants a minimum spectral cutoff)
    if cutoff > 0
        total_weight = sum(abs2, svals)
        frob_rank = full_rank
        cumulative = 0.0
        for k in full_rank:-1:2
            cumulative += abs2(svals[k])
            if cumulative / total_weight > cutoff^2
                frob_rank = k
                break
            end
            frob_rank = k - 1
        end
        frob_rank = max(frob_rank, mindim)
        # Take the more conservative (larger) rank
        chosen_rank = max(chosen_rank, frob_rank)
    end
    chosen_rank = min(chosen_rank, maxdim)

    # ── 6. Perform the actual truncation ─────────────────────────────
    truncerr = sqrt(sum(abs2, svals[(chosen_rank + 1):end]) / sum(abs2, svals))

    spec = replacebond!(
        state, b, phi;
        maxdim = chosen_rank,
        mindim,
        cutoff = 0.0,  # we already chose the rank
        ortho,
        normalize,
    )

    return (; truncerr, chosen_rank, res_exact, spec)
end


"""Compute the projected RHS b_local from the reduced operator."""
function _local_rhs(reduced_operator)
    if reduced_operator isa ReducedLinearProblem
        return ITensorMPS.constant_term(reduced_operator)
    elseif hasproperty(reduced_operator, :linear_problem)
        return ITensorMPS.constant_term(reduced_operator.linear_problem)
    else
        error("Cannot extract constant_term from operator type $(typeof(reduced_operator))")
    end
end

"""Compute ‖A_local * phi - b_local‖."""
function _local_residual_norm(reduced_operator::ReducedLinearProblem, phi::ITensor, rhs::ITensor)
    Aphi = ITensors.contract(reduced_operator, phi)
    Aphi_unprimed = noprime(Aphi)
    return norm(Aphi_unprimed - rhs)
end

"""Compute ‖A_local * phi - b_local‖."""
function _local_residual_norm(reduced_operator::Union{ReducedPrecondLinearProblem, ReducedFAIPrecondLinearProblem}, phi::ITensor, rhs::ITensor)
    Aphi = ITensors.contract(reduced_operator.linear_problem, phi)
    Aphi_unprimed = noprime(Aphi)
    return norm(Aphi_unprimed - rhs)
end

"""Reconstruct a rank-r truncated version of U * S * V†."""
function _truncated_phi(U_full::ITensor, S_full::ITensor, V_full::ITensor, r::Int)
    s_inds = inds(S_full)
    s_dim = dim(s_inds[1])

    # Build a projector that zeros out singular values beyond rank r
    S_trunc = ITensor(eltype(S_full), s_inds...)
    for k in 1:r
        S_trunc[s_inds[1] => k, s_inds[2] => k] = S_full[s_inds[1] => k, s_inds[2] => k]
    end
    return U_full * S_trunc * V_full
end


