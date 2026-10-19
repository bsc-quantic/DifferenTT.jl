"""
    ResidualALS(; sweep_kwargs=SweepingArgs(), updater_kwargs=LocalUpdaterArgs(), updater=KrylovUpdater(),
                residual_tol=1e-12, residual_damp=1.1) <: AbstractDMRGSolver

Two-site ALS (MALS) for ``A \\mathbf{x} = \\mathbf{b}`` whose SVD truncation chooses the rank by the local residual,
as proposed by Oseledets & Dolgov (2012). This is the solver called MALS in Torrente et al., *Factorized approximate
inverse preconditioners in the tensor train format* (Sec. 5), where it proved more robust than [`ALS`](@ref) in
meeting the prescribed residual bounds.

A truncation by singular values controls the error in ``\\mathbf{x}``, but not its effect on the residual, which is what
the solve has to bring below a tolerance. We therefore choose the rank of each new pair of cores as follows. After the
local solve, we compute the full SVD of the two-site tensor ``\\phi = U S V^\\top`` and the relative local residual
``\\varepsilon_{\\text{exact}}`` of the untruncated tensor. We then keep the smallest rank ``r`` whose truncation
``\\phi_r`` satisfies

```math
\\frac{\\|A_{\\text{loc}}\\phi_r - \\mathbf{b}_{\\text{loc}}\\|}{\\|\\mathbf{b}_{\\text{loc}}\\|}
\\le \\delta \\max\\left(\\frac{\\varepsilon}{\\sqrt{N}}, \\varepsilon_{\\text{exact}}\\right),
```

where ``\\varepsilon`` is `residual_tol`, ``\\delta`` is `residual_damp` and ``N`` is the number of sites, found by a
bisection over ``r`` that assumes the residual decreases with the rank. The scaling by ``1/\\sqrt{N}`` follows Dolgov's
implementations, so that the ``N`` local errors of a sweep add up to about ``\\varepsilon``. The rank is then clamped to
`[mindim, maxdim]`. The sweeping `cutoff` (from [`SweepingArgs`](@ref)) adds a floor on the rank: we keep at least the
smallest rank whose discarded relative weight ``\\sum_{k>r} \\sigma_k^2 / \\sum_k \\sigma_k^2`` is at most `cutoff`, as
in ITensors, and `cutoff = 0` disables the floor, so that the rank is chosen by the residual alone. In a preconditioned
solve, the local residual is always that of the unpreconditioned ``A_{\\text{loc}}``.

The residual-based truncation applies only to `nsite = 2`. With `nsite = 1`, the sweep falls back to the truncation
of [`ALS`](@ref). The cost per bond is that of the local solve plus ``\\mathcal{O}(\\log r_{\\max})`` applications
of ``A_{\\text{loc}}`` for the bisection.

# Fields

  - `sweep_kwargs::SweepingArgs`: sweeps, maximum rank, Frobenius floor and observers.
  - `updater_kwargs::LocalUpdaterArgs`: local Krylov solver and diagnostics.
  - `updater::AbstractUpdater`: local solver.
  - `residual_tol::Float64=1e-12`: target relative local residual ``\\varepsilon`` of the truncation, before the
    scaling by ``1/\\sqrt{N}``.
  - `residual_damp::Float64=1.1`: factor ``\\delta`` by which a truncated tensor may exceed the bound.

# Examples

```julia
solver = ResidualALS(;
    sweep_kwargs=SweepingArgs(; nsweeps=30, maxdim=150, cutoff=1e-13, (observer!)=ResidualObserver(; rtol=1e-5)),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-6),
)
x = linearsolve(solver, A, b, M⁻¹, nothing)
```

See also [`linearsolve`](@ref), [`ALS`](@ref), [`AMEn`](@ref).
"""
Base.@kwdef struct ResidualALS <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    updater::AbstractUpdater = KrylovUpdater()
    residual_tol::Float64 = 1e-12
    residual_damp::Float64 = 1.1
end

"""
    linearsolve(solver::ResidualALS, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` by MALS sweeps with residual-based truncation from `x0`. The keywords override
the fields of `solver.sweep_kwargs` (see [`linearsolve`](@ref)).
"""
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

"""
    linearsolve(solver::ResidualALS, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing};
                preconditioner_sites=nothing, kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` by MALS sweeps with residual-based truncation from `x0`, preconditioning each
local problem with the projection of `M⁻¹` (Sec. 4 of Torrente et al.). The truncation still uses the residual of the
unpreconditioned local problem. Only the local problems whose first site is in `preconditioner_sites` are
preconditioned, when it is given; the other keywords override the fields of `solver.sweep_kwargs`.
"""
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
    residual_replacebond!(state::MPS, b::Int, phi::ITensor, reduced_operator; maxdim, mindim, cutoff, normalize,
                          ortho, resid_tol, resid_damp)

Split the two-site tensor `phi` into `state[b]` and `state[b+1]`, with the rank chosen by [`_residual_rank`](@ref) (the
smallest rank that keeps the local residual below `max(resid_tol, res_exact) * resid_damp`, with `cutoff` as a Frobenius
floor in the ITensors convention). Falls back to the ITensors truncation if the local right-hand side is zero. Returns
`(; truncerr, chosen_rank, res_exact, spec)`, or the spectrum alone in that fallback; throws an `ArgumentError` if `phi`
is not finite.
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
    # The floor follows the ITensors convention: the discarded relative weight Σσ² is at most `cutoff`
    chosen_rank, truncerr, res_exact = _residual_rank(
        reduced_operator, phi, rhs, U_full, S_full, V_full; resid_tol, resid_damp, frob_weight=cutoff, mindim, maxdim
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
    residual_alternating_update(operator, init::MPS; updater, updater_kwargs, resid_tol=1e-8, resid_damp=2.0, kwargs...)

Sweeping loop of [`ResidualALS`](@ref), a copy of `ITensorMPS.alternating_update` whose 2-site steps truncate with
[`residual_replacebond!`](@ref) at the tolerance `resid_tol / √N`. The other keywords are those of
`alternating_update`. Returns the final MPS.
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

"""
    residual_sweep_update(order, reduced_operator, state::MPS; resid_tol, resid_damp, kwargs...)

One sweep of [`residual_alternating_update`](@ref): one [`residual_sub_sweep_update`](@ref) per half-sweep of `order`.
Returns `(state, reduced_operator, info)`.
"""
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

"""
    residual_sub_sweep_update(direction, reduced_operator, state::MPS; updater, updater_kwargs, nsite, resid_tol,
                              resid_damp, kwargs...)

One half-sweep in `direction`: at each bond, solve the local problem with `updater` and split it with
[`residual_replacebond!`](@ref) (`nsite = 2`), or with `ITensorMPS.region_update!` (`nsite = 1`). Returns
`(state, reduced_operator, (; maxtruncerr))`.
"""
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
