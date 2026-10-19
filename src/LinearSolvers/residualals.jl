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

function linearsolve(solver::ResidualALS, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_problem = ReducedLinearProblem(A, b)
    updater_kwargs = (; coefficients=(false, true), NamedTuple(solver.updater_kwargs)...)
    sweep_kwargs = linearsolve_sweep_kwargs(solver, A, b, kwargs)

    return residual_alternating_update(
        reduced_problem,
        init;
        updater=solver.updater,
        updater_kwargs,
        resid_tol=solver.residual_tol,
        resid_damp=solver.residual_damp,
        sweep_kwargs...,
    )
end

function linearsolve(
    solver::ResidualALS, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing}; kwargs...
)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_precond_problem = initialize_reducedprecondlinearproblem(A, b, M⁻¹)
    updater_kwargs, sweep_kwargs = precond_linearsolve_kwargs(solver, A, b, kwargs)

    return residual_alternating_update(
        reduced_precond_problem,
        init;
        updater=solver.updater,
        updater_kwargs,
        resid_tol=solver.residual_tol,
        resid_damp=solver.residual_damp,
        sweep_kwargs...,
    )
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
    maxdim::Int=typemax(Int),
    mindim::Int=1,
    cutoff::Float64=0.0,    # Frobenius cutoff (fallback floor)
    normalize::Bool=false,
    ortho::String="left",
    resid_tol::Float64=1e-8,
    resid_damp::Float64=2.0,
)
    # ── 1. Full SVD ──────────────────────────────────────────────────
    indsMb = inds(state[b])
    phi_data = array(phi)
    if !all(isfinite, phi_data)
        throw(
            ArgumentError(
                "residual_replacebond!: bond tensor at site $b contains NaN/Inf; " *
                "refusing SVD (often caused by a broken preconditioner or Krylov update).",
            ),
        )
    end
    U_full, S_full, V_full = svd(phi, indsMb; alg="recursive")

    # ── 2. Compute RHS (projected constant term) ─────────────────────
    rhs = _local_rhs(reduced_operator)
    norm_rhs = norm(rhs)

    if norm_rhs ≈ 0
        # Trivial case – just do standard truncation, to avoid /0
        spec = replacebond!(state, b, phi; maxdim, mindim, cutoff, ortho, normalize)
        return spec
    end

    # ── 3–5. Residual-based rank selection, with the Frobenius cutoff as a floor ──
    # NOTE: the floor compares the discarded relative weight Σσ² with `cutoff^2`, whereas ITensors
    # compares it with `cutoff`. This is kept as is so that ResidualALS results do not change (with
    # the usual cutoff = 1e-10 the floor is effectively inactive). Decide which convention is
    # intended; AMEn uses the ITensors one.
    chosen_rank, truncerr, res_exact = _residual_rank(
        reduced_operator, phi, rhs, U_full, S_full, V_full; resid_tol, resid_damp, frob_weight=cutoff^2, mindim, maxdim
    )

    # ── 6. Perform the actual truncation ─────────────────────────────
    spec = replacebond!(
        state,
        b,
        phi;
        maxdim=chosen_rank,
        mindim,
        cutoff=0.0,  # we already chose the rank
        ortho,
        normalize,
    )

    return (; truncerr, chosen_rank, res_exact, spec)
end

"""
    _frobenius_rank(svals, frob_weight, mindim)

Smallest rank (never below `mindim`) whose discarded weight `Σ_{k>r} σ_k² / Σ σ_k²` does not exceed
`frob_weight`. `svals` must be sorted in decreasing order.
"""
function _frobenius_rank(svals::AbstractVector, frob_weight::Real, mindim::Int)
    full_rank = length(svals)
    total_weight = sum(abs2, svals)
    frob_rank = full_rank
    cumulative = 0.0
    for k in full_rank:-1:2
        cumulative += abs2(svals[k])
        if cumulative / total_weight > frob_weight
            frob_rank = k
            break
        end
        frob_rank = k - 1
    end
    return max(frob_rank, mindim)
end

"""
    _residual_rank(reduced_operator, phi, rhs, U_full, S_full, V_full; resid_tol, resid_damp,
                   frob_weight = 0.0, mindim = 1, maxdim = typemax(Int))

Smallest rank `r` such that truncating the SVD `phi = U_full * S_full * V_full` to its `r` largest
singular values keeps the relative local residual `‖A_loc φ_r - rhs‖ / ‖rhs‖` below
`max(resid_tol, res_exact) * resid_damp`, where `res_exact` is the relative local residual of `phi`.
The reduced operator must be positioned at the site(s) of `phi`, and `rhs` must be its local
right-hand side (`_local_rhs`), with `norm(rhs) > 0`.

On top of that, a Frobenius floor keeps every rank whose discarded weight `Σ_{k>r} σ_k² / Σ σ_k²`
would exceed `frob_weight` (`0.0` disables it). The result is clamped to `[mindim, maxdim]`.

# Returns

  - `(chosen_rank, truncerr, res_exact)`, with `truncerr = sqrt(discarded weight / total weight)`.
"""
function _residual_rank(
    reduced_operator,
    phi::ITensor,
    rhs::ITensor,
    U_full::ITensor,
    S_full::ITensor,
    V_full::ITensor;
    resid_tol::Real,
    resid_damp::Real,
    frob_weight::Real=0.0,
    mindim::Int=1,
    maxdim::Int=typemax(Int),
)
    norm_rhs = norm(rhs)
    bond_ind = commoninds(U_full, S_full)
    full_rank = dim(bond_ind[1])

    # Residual of the untruncated solution
    res_exact = _local_residual_norm(reduced_operator, phi, rhs) / norm_rhs

    # Residual-based rank selection: walk from full rank downward; stop when residual exceeds threshold
    threshold = max(resid_tol * resid_damp, res_exact * resid_damp)

    # Extract singular values as a vector for fast iteration
    svals = _singular_values(S_full)

    low = max(1, mindim)
    high = full_rank - 1
    chosen_rank = full_rank

    # Binary search of required χ
    while low <= high
        mid = (low + high) ÷ 2
        # Reconstruct truncated phi
        phi_trunc = _truncated_phi(U_full, S_full, V_full, mid)
        res_trunc = _local_residual_norm(reduced_operator, phi_trunc, rhs) / norm_rhs

        if res_trunc <= threshold
            chosen_rank = mid
            high = mid - 1
        else
            low = mid + 1
        end
    end

    # Clamp
    chosen_rank = clamp(chosen_rank, mindim, min(maxdim, full_rank))

    # Also apply the Frobenius cutoff as a floor (in case the user still wants a minimum spectral cutoff)
    total_weight = sum(abs2, svals)
    if frob_weight > 0 && total_weight > 0
        frob_rank = _frobenius_rank(svals, frob_weight, mindim)
        # Take the more conservative (larger) rank
        chosen_rank = max(chosen_rank, frob_rank)
    end
    chosen_rank = min(chosen_rank, maxdim)

    return chosen_rank, _truncation_error(svals, chosen_rank), res_exact
end

"""
Singular values stored in the diagonal ITensor `S` (as returned by `svd`), as a vector.
"""
function _singular_values(S::ITensor)
    s_inds = inds(S)
    return [S[s_inds[1]=>k, s_inds[2]=>k] for k in 1:dim(s_inds[1])]
end

"""
Relative truncation error `sqrt(Σ_{k>r} σ_k² / Σ σ_k²)` of keeping the `r` largest singular values.
"""
function _truncation_error(svals::AbstractVector, r::Int)
    total_weight = sum(abs2, svals)
    return total_weight > 0 ? sqrt(sum(abs2, svals[(r+1):end]) / total_weight) : 0.0
end

"""
Compute the projected RHS b_local from the reduced operator.
"""
function _local_rhs(reduced_operator)
    if reduced_operator isa ReducedLinearProblem
        return ITensorMPS.constant_term(reduced_operator)
    elseif hasproperty(reduced_operator, :linear_problem)
        return ITensorMPS.constant_term(reduced_operator.linear_problem)
    else
        error("Cannot extract constant_term from operator type $(typeof(reduced_operator))")
    end
end

"""
Compute ‖A_local * phi - b_local‖.
"""
function _local_residual_norm(reduced_operator::ReducedLinearProblem, phi::ITensor, rhs::ITensor)
    Aphi = ITensors.contract(reduced_operator, phi)
    Aphi_unprimed = noprime(Aphi)
    return norm(Aphi_unprimed - rhs)
end

"""
Compute ‖A_local * phi - b_local‖.
"""
function _local_residual_norm(
    reduced_operator::Union{ReducedPrecondLinearProblem,ReducedFAIPrecondLinearProblem}, phi::ITensor, rhs::ITensor
)
    Aphi = ITensors.contract(reduced_operator.linear_problem, phi)
    Aphi_unprimed = noprime(Aphi)
    return norm(Aphi_unprimed - rhs)
end

"""
Compute the local residual tensor b_local - A_local * phi.
"""
function _local_residual(reduced_operator::ReducedLinearProblem, phi::ITensor, rhs::ITensor)
    return rhs - noprime(ITensors.contract(reduced_operator, phi))
end

"""
Compute the local residual tensor b_local - A_local * phi (with the unpreconditioned operator).
"""
function _local_residual(
    reduced_operator::Union{ReducedPrecondLinearProblem,ReducedFAIPrecondLinearProblem}, phi::ITensor, rhs::ITensor
)
    return rhs - noprime(ITensors.contract(reduced_operator.linear_problem, phi))
end

"""
Reconstruct a rank-r truncated version of U * S * V†.
"""
function _truncated_phi(U_full::ITensor, S_full::ITensor, V_full::ITensor, r::Int)
    s_inds = inds(S_full)
    s_dim = dim(s_inds[1])

    # Build a projector that zeros out singular values beyond rank r
    S_trunc = ITensor(eltype(S_full), s_inds...)
    for k in 1:r
        S_trunc[s_inds[1]=>k, s_inds[2]=>k] = S_full[s_inds[1]=>k, s_inds[2]=>k]
    end
    return U_full * S_trunc * V_full
end

"""
    residual_alternating_update(operator, init; kwargs...)

Modified `alternating_update` that passes the projected operator into
`residual_replacebond!` for residual-aware truncation at each 2-site step.
Falls back to standard truncation for 1-site sweeps.
"""
function residual_alternating_update(
    operator,
    init::MPS;
    updater,
    updater_kwargs=(;),
    nsweeps=ITensorMPS.default_nsweeps(),
    checkdone=ITensorMPS.default_checkdone(),
    write_when_maxdim_exceeds=ITensorMPS.default_write_when_maxdim_exceeds(),
    nsite=ITensorMPS.default_nsite(),
    reverse_step=ITensorMPS.default_reverse_step(),
    time_start=ITensorMPS.default_time_start(),
    time_step=ITensorMPS.default_time_step(),
    order=ITensorMPS.default_order(),
    (observer!)=ITensorMPS.default_observer(),
    (sweep_observer!)=ITensorMPS.default_sweep_observer(),
    outputlevel=ITensorMPS.default_outputlevel(),
    normalize=ITensorMPS.default_normalize(),
    maxdim=ITensorMPS.default_maxdim(),
    mindim=ITensorMPS.default_mindim(),
    cutoff=ITensorMPS.default_cutoff(ITensors.scalartype(init)),
    noise=ITensorMPS.default_noise(),
    resid_tol::Float64=1e-8,
    resid_damp::Float64=2.0,
)
    reduced_operator = ITensorMPS.reduced_operator(operator)
    if isnothing(nsweeps)
        error("Must specify `nsweeps`.")
    end
    maxdim, mindim, cutoff, noise = ITensorMPS.process_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
    forward_order = ITensorMPS.TDVPOrder(order, Base.Forward)
    state = copy(init)
    current_time = time_start
    info = nothing
    N = length(state)

    # Per-sweep resid_tol scaled by √d (following Dolgov)
    scaled_resid_tol = resid_tol / sqrt(N)

    for sweep in 1:nsweeps
        if !isnothing(write_when_maxdim_exceeds) && maxdim[sweep] > write_when_maxdim_exceeds
            if outputlevel >= 2
                println("Writing environment tensors to disk")
            end
            reduced_operator = disk(reduced_operator)
        end
        sweep_elapsed_time = @elapsed begin
            state, reduced_operator, info = residual_sweep_update(
                ITensorMPS.TDVPOrder(order, Base.Forward),
                reduced_operator,
                state;
                updater,
                updater_kwargs,
                nsite,
                current_time,
                time_step,
                reverse_step,
                sweep,
                observer!,
                normalize,
                outputlevel,
                maxdim=maxdim[sweep],
                mindim=mindim[sweep],
                cutoff=cutoff[sweep],
                noise=noise[sweep],
                resid_tol=scaled_resid_tol,
                resid_damp,
            )
        end
        if !isnothing(time_step)
            current_time += time_step
        end
        ITensorMPS.update_observer!(sweep_observer!; state, reduced_operator, sweep, outputlevel, current_time)
        if outputlevel >= 1
            print("After sweep ", sweep, ":")
            print(" maxlinkdim=", maxlinkdim(state))
            @printf(" maxerr=%.2E", info.maxtruncerr)
            if !isnothing(current_time)
                print(" current_time=", round(current_time; digits=3))
            end
            print(" time=", round(sweep_elapsed_time; digits=3))
            println()
            flush(stdout)
        end
        isdone = checkdone(; state, sweep, outputlevel, observer=(observer!), sweep_observer=(sweep_observer!), info)
        isdone && break
    end
    return state
end

function residual_sweep_update(
    order::ITensorMPS.TDVPOrder,
    reduced_operator,
    state::MPS;
    current_time=nothing,
    time_step=nothing,
    resid_tol=1e-8,
    resid_damp=2.0,
    kwargs...,
)
    order_orderings = ITensorMPS.orderings(order)
    order_sub_time_steps = ITensorMPS.sub_time_steps(order)
    if !isnothing(time_step)
        order_sub_time_steps = eltype(time_step).(order_sub_time_steps)
        order_sub_time_steps *= time_step
    end
    info = nothing
    for substep in 1:length(order_sub_time_steps)
        sub_time_step = isnothing(time_step) ? nothing : order_sub_time_steps[substep]
        state, reduced_operator, info = residual_sub_sweep_update(
            order_orderings[substep],
            reduced_operator,
            state;
            current_time,
            time_step=sub_time_step,
            resid_tol,
            resid_damp,
            kwargs...,
        )
        if !isnothing(time_step)
            current_time += sub_time_step
        end
    end
    return state, reduced_operator, info
end

function residual_sub_sweep_update(
    direction::Base.Ordering,
    reduced_operator,
    state::MPS;
    updater,
    updater_kwargs,
    which_decomp=nothing,
    svd_alg=nothing,
    sweep=ITensorMPS.default_sweep(),
    current_time=nothing,
    time_step=nothing,
    nsite=ITensorMPS.default_nsite(),
    reverse_step=ITensorMPS.default_reverse_step(),
    normalize=ITensorMPS.default_normalize(),
    (observer!)=ITensorMPS.default_observer(),
    outputlevel=ITensorMPS.default_outputlevel(),
    maxdim=ITensorMPS.default_maxdim(),
    mindim=ITensorMPS.default_mindim(),
    cutoff=ITensorMPS.default_cutoff(ITensors.scalartype(state)),
    noise=ITensorMPS.default_noise(),
    resid_tol=1e-8,
    resid_damp=2.0,
)
    reduced_operator = copy(reduced_operator)
    state = copy(state)
    N = length(state)
    set_nsite!(reduced_operator, nsite)

    if ITensorMPS.isforward(direction)
        if !isortho(state) || orthocenter(state) != 1
            orthogonalize!(state, 1)
        end
        position!(reduced_operator, state, 1)
    elseif ITensorMPS.isreverse(direction)
        if !isortho(state) || orthocenter(state) != N - nsite + 1
            orthogonalize!(state, N - nsite + 1)
        end
        position!(reduced_operator, state, N - nsite + 1)
    end

    maxtruncerr = 0.0
    info = nothing

    for b in ITensorMPS.sweep_bonds(direction, N; ncenter=nsite)
        if nsite == 2
            set_nsite!(reduced_operator, 2)
            position!(reduced_operator, state, b)
            reduced_state = state[b] * state[b+1]
            internal_kwargs = (; current_time, time_step, outputlevel)
            reduced_state, info = updater(reduced_operator, reduced_state; internal_kwargs, updater_kwargs...)
            if !isnothing(time_step)
                current_time += time_step
            end
            normalize && (reduced_state /= norm(reduced_state))

            ortho_str = ITensorMPS.isforward(direction) ? "left" : "right"
            drho = nothing
            if noise > 0.0 && ITensorMPS.isforward(direction)
                drho = noise * noiseterm(reduced_operator, reduced_state, ortho_str)
            end

            result = residual_replacebond!(
                state,
                b,
                reduced_state,
                reduced_operator;
                maxdim,
                mindim,
                cutoff,
                normalize,
                ortho=ortho_str,
                resid_tol,
                resid_damp,
            )
            truncerr = result isa NamedTuple ? result.truncerr : (result isa Nothing ? 0.0 : result.truncerr)
            maxtruncerr = max(maxtruncerr, truncerr)
            spec = result isa NamedTuple ? result.spec : result

        else
            # For 1-site: fall back to standard ITensorMPS logic
            current_time, maxtruncerr, spec, info = ITensorMPS.region_update!(
                reduced_operator,
                state,
                b;
                updater,
                updater_kwargs,
                nsite,
                reverse_step,
                current_time,
                outputlevel,
                time_step,
                normalize,
                direction,
                noise,
                which_decomp,
                svd_alg,
                cutoff,
                maxdim,
                mindim,
                maxtruncerr,
            )
        end

        if outputlevel >= 2
            if nsite == 2
                @printf("Sweep %d, direction %s, bond (%d,%d)\n", sweep, direction, b, b + 1)
            end
            flush(stdout)
        end

        ITensorMPS.update_observer!(
            observer!;
            state,
            reduced_operator,
            bond=b,
            sweep,
            half_sweep=ITensorMPS.isforward(direction) ? 1 : 2,
            spec=nothing,
            outputlevel,
            half_sweep_is_done=ITensorMPS.is_half_sweep_done(direction, b, N; ncenter=nsite),
            current_time,
            info,
        )
    end
    normalize && normalize!(state)
    return state, reduced_operator, (; maxtruncerr)
end
