"""
    AMEn(; sweep_kwargs=SweepingArgs(), updater_kwargs=LocalUpdaterArgs(), updater=KrylovUpdater(),
         enrichment_maxdim=5, enrichment_cutoff=1e-10, residual_tol=1e-8, residual_damp=1.1, kick_tol=nothing,
         local_stop=true) <: AbstractDMRGSolver

Alternating Minimal Energy (AMEn) solver for ``A \\mathbf{x} = \\mathbf{b}`` in the TT format (Dolgov & Savostyanov,
2014, Part II, arXiv:1304.1222).

The two-site MALS adapts the rank of the solution, but its local problems have dimension ``\\chi^2 d^2``, while the
one-site ALS is cheaper but cannot change the rank. AMEn keeps the one-site local problems and adapts the rank by
enriching each core with directions taken from the residual, so that the basis grows where the error is. At every
bond of a sweep we proceed as follows:

 1. We update the core by a one-site local solve.
 2. We truncate the core on its own by the local residual, as in [`ResidualALS`](@ref): we keep the smallest rank
    whose local residual stays below `residual_damp * max(residual_tol / √N, ε_exact)`, where ``N`` is the number of
    sites and ``\\varepsilon_{\\text{exact}}`` the local residual of the untruncated core, with the sweeping `cutoff` as
    a Frobenius floor in the ITensors convention. Since the core is truncated alone, the rank does not depend on the
    scale of ``\\mathbf{x}`` nor on the enrichment.
 3. We enrich the truncated core with up to `enrichment_maxdim` directions of the residual of the two-site problem,
    unless that residual is already below `kick_tol`, so that the rank stops growing once the local problem is
    solved. The enriched core is orthonormalized with a QR, so the enrichment changes neither ``\\mathbf{x}`` nor
    depends on its scale.
 4. We move the orthogonality center to the next site.

The sweeping `maxdim` is honored: the truncation of step 2 leaves room for the enrichment of step 3. AMEn always
sweeps with one-site updates, so `SweepingArgs.nsite` is ignored.

# Fields

  - `sweep_kwargs::SweepingArgs`, `updater_kwargs::LocalUpdaterArgs`, `updater::AbstractUpdater`: sweeps, local solver
    arguments and local solver, as for the other DMRG-type solvers.
  - `enrichment_maxdim::Int=5`: maximum number of residual directions added per bond and half-sweep.
  - `enrichment_cutoff::Float64=1e-10`: cutoff, relative to ``\\|Z\\|^2``, of the SVD of the residual ``Z`` that
    extracts the enrichment directions.
  - `residual_tol::Float64=1e-8`: target relative local residual. The rank of each core is chosen so that its local
    residual stays below `residual_tol / √N`, and with `local_stop` the sweeps stop once the local residual of every
    core is below `residual_tol`.
  - `residual_damp::Float64=1.1`: factor by which the local residual of a truncated core may exceed
    `max(residual_tol / √N, ε_exact)`.
  - `kick_tol::Union{Float64,Nothing}=nothing`: no enrichment is added at a bond where
    ``\\|Z\\| \\le \\texttt{kick\\_tol}\\, \\|\\mathbf{b}_{\\text{loc}}\\|``; `nothing` uses `residual_tol / √N`.
  - `local_stop::Bool=true`: stop after the half-sweep in which the local residual of every core, evaluated with the
    old core just before its update, was below `residual_tol` (the stopping rule of Sec. 5.2 of Dolgov & Savostyanov,
    Part II). It needs no global residual, and it is applied in addition to any `checkdone` or
    [`ResidualObserver`](@ref) given in `sweep_kwargs`.

# Examples

```julia
solver = AMEn(;
    sweep_kwargs=SweepingArgs(; nsweeps=10, maxdim=40),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true),
    residual_tol=1e-6,
)
x = linearsolve(solver, A, b, nothing)
```

See also [`linearsolve`](@ref), [`ResidualALS`](@ref), [`ExpandedALS`](@ref).
"""
Base.@kwdef struct AMEn <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    updater::AbstractUpdater = KrylovUpdater()
    enrichment_maxdim::Int = 5
    enrichment_cutoff::Float64 = 1e-10
    residual_tol::Float64 = 1e-8
    residual_damp::Float64 = 1.1
    kick_tol::Union{Float64,Nothing} = nothing
    local_stop::Bool = true
end

"""
    linearsolve(solver::AMEn, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` by AMEn sweeps from `x0`. The keywords override the fields of
`solver.sweep_kwargs` (see [`linearsolve`](@ref)).
"""
function linearsolve(solver::AMEn, A::MPO, b::MPS, x0::Union{MPS,Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_problem = ReducedLinearProblem(A, b)
    updater_kwargs = (; coefficients=(false, true), NamedTuple(solver.updater_kwargs)...)
    sweep_kwargs = linearsolve_sweep_kwargs(solver, A, b, kwargs)

    return amen_alternating_update(
        reduced_problem,
        init;
        updater=solver.updater,
        updater_kwargs,
        enrich_maxdim=solver.enrichment_maxdim,
        enrich_cutoff=solver.enrichment_cutoff,
        resid_tol=solver.residual_tol,
        resid_damp=solver.residual_damp,
        kick_tol=solver.kick_tol,
        local_stop=solver.local_stop,
        sweep_kwargs...,
    )
end

"""
    linearsolve(solver::AMEn, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing};
                preconditioner_sites=nothing, kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` by AMEn sweeps from `x0`, preconditioning each one-site local problem with the
projection of `M⁻¹` (Sec. 4 of Torrente et al.), and only those whose site is in `preconditioner_sites` when it is
given. The truncation and the enrichment use the residual of the unpreconditioned problem.
"""
function linearsolve(solver::AMEn, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS,Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_precond_problem = initialize_reducedprecondlinearproblem(A, b, M⁻¹)
    updater_kwargs, sweep_kwargs = precond_linearsolve_kwargs(solver, A, b, kwargs)

    return amen_alternating_update(
        reduced_precond_problem,
        init;
        updater=solver.updater,
        updater_kwargs,
        enrich_maxdim=solver.enrichment_maxdim,
        enrich_cutoff=solver.enrichment_cutoff,
        resid_tol=solver.residual_tol,
        resid_damp=solver.residual_damp,
        kick_tol=solver.kick_tol,
        local_stop=solver.local_stop,
        sweep_kwargs...,
    )
end

"""
    _clear_cache!(P)

Invalidate the cached environments of the projected problem `P` (and of its preconditioner), so that the next
`position!` rebuilds them; needed after `orthogonalize!` changes the state behind them. A no-op for other objects.
"""
_clear_cache!(P::Any) = nothing

function _clear_cache!(P::ITensorMPS.AbstractProjMPO)
    if hasproperty(P, :lpos)
        P.lpos = 0
    end
    if hasproperty(P, :rpos)
        if hasproperty(P, :H)
            P.rpos = length(P.H) + 1
        elseif hasproperty(P, :state)
            P.rpos = length(P.state) + 1
        else
            P.rpos = 10000
        end
    end
end

function _clear_cache!(P::ITensorMPS.ReducedLinearProblem)
    _clear_cache!(P.reduced_operator)
    for ct in P.reduced_constant_terms
        _clear_cache!(ct)
    end
end

function _clear_cache!(P::Union{ReducedPrecondLinearProblem,ReducedFAIPrecondLinearProblem})
    _clear_cache!(P.linear_problem)
    _clear_cache!(P.preconditioner)
end

function _clear_cache!(P::AbstractReducedPreconditioner)
    _clear_cache!(P.reduced_op)
end

"""
    amen_alternating_update(operator, init::MPS; updater, updater_kwargs, enrich_maxdim, enrich_cutoff, resid_tol,
                            resid_damp, kick_tol, local_stop, kwargs...)

Sweeping loop of [`AMEn`](@ref), a copy of `ITensorMPS.alternating_update` with one-site steps, the residual
tolerances scaled by ``1/\\sqrt{N}``, and the local stopping rule. Returns the final MPS.
"""
function amen_alternating_update(
    operator,
    init::MPS;
    updater,
    updater_kwargs=(;),
    enrich_maxdim::Int=5,
    enrich_cutoff::Float64=1e-10,
    resid_tol::Float64=1e-8,
    resid_damp::Float64=1.1,
    kick_tol::Union{Float64,Nothing}=nothing,
    local_stop::Bool=true,
    nsweeps=ITensorMPS.default_nsweeps(),
    checkdone=ITensorMPS.default_checkdone(),
    write_when_maxdim_exceeds=ITensorMPS.default_write_when_maxdim_exceeds(),
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
    nsite=ITensorMPS.default_nsite(),
)
    # AMEn forces 1-site optimization explicitly during the custom sweep loop,
    # but uses a 2-site operator temporarily to compute the residual.
    nsite = 1

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

    # Per-core tolerances scaled by √d (following Dolgov)
    scaled_resid_tol = resid_tol / sqrt(N)
    scaled_kick_tol = isnothing(kick_tol) ? scaled_resid_tol : kick_tol
    stop_tol = local_stop ? resid_tol : nothing

    for sweep in 1:nsweeps
        if !isnothing(write_when_maxdim_exceeds) && maxdim[sweep] > write_when_maxdim_exceeds
            if outputlevel >= 2
                println("Writing environment tensors to disk")
            end
            reduced_operator = disk(reduced_operator)
        end
        sweep_elapsed_time = @elapsed begin
            state, reduced_operator, info = amen_sweep_update(
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
                enrich_maxdim,
                enrich_cutoff,
                resid_tol=scaled_resid_tol,
                resid_damp,
                kick_tol=scaled_kick_tol,
                stop_tol,
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
            if local_stop
                @printf(" maxlocres=%.2E", info.maxlocres)
            end
            if !isnothing(current_time)
                print(" current_time=", round(current_time; digits=3))
            end
            print(" time=", round(sweep_elapsed_time; digits=3))
            println()
            flush(stdout)
        end
        isdone = checkdone(; state, sweep, outputlevel, observer=(observer!), sweep_observer=(sweep_observer!), info)
        isdone && break
        if local_stop && info.maxlocres < resid_tol
            if outputlevel >= 1
                @printf("AMEn local residual stopping condition met: %.2E < %.2E\n", info.maxlocres, resid_tol)
            end
            break
        end
    end
    return state
end

"""
    amen_sweep_update(order, reduced_operator, state::MPS; stop_tol=nothing, kwargs...)

One AMEn sweep: one [`amen_sub_sweep_update`](@ref) per half-sweep, stopping after a half-sweep whose largest local
residual is below `stop_tol`. Returns `(state, reduced_operator, info)`.
"""
function amen_sweep_update(
    order::ITensorMPS.TDVPOrder,
    reduced_operator,
    state::MPS;
    current_time=nothing,
    time_step=nothing,
    stop_tol=nothing,
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
        state, reduced_operator, info = amen_sub_sweep_update(
            order_orderings[substep],
            reduced_operator,
            state;
            current_time,
            time_step=sub_time_step,
            stop_tol,
            kwargs...,
        )
        if !isnothing(time_step)
            current_time += sub_time_step
        end
        # Stopping rule of Dolgov & Savostyanov, Part II (Sec. 5.2), checked after every half-sweep
        if !isnothing(stop_tol) && info.maxlocres < stop_tol
            break
        end
    end
    return state, reduced_operator, info
end

"""
    amen_sub_sweep_update(direction, reduced_operator, state::MPS; updater, updater_kwargs, kwargs...)

One AMEn half-sweep in `direction`: at each site, the one-site solve and [`_amen_truncate_enrich_shift!`](@ref).
Returns `(state, reduced_operator, (; maxtruncerr, maxlocres))`, with `maxlocres = NaN` unless `stop_tol` is set.
"""
function amen_sub_sweep_update(
    direction::Base.Ordering,
    reduced_operator,
    state::MPS;
    updater,
    updater_kwargs,
    sweep=ITensorMPS.default_sweep(),
    current_time=nothing,
    time_step=nothing,
    nsite=1,
    reverse_step=ITensorMPS.default_reverse_step(),
    normalize=ITensorMPS.default_normalize(),
    (observer!)=ITensorMPS.default_observer(),
    outputlevel=ITensorMPS.default_outputlevel(),
    maxdim=ITensorMPS.default_maxdim(),
    mindim=ITensorMPS.default_mindim(),
    cutoff=ITensorMPS.default_cutoff(ITensors.scalartype(state)),
    noise=ITensorMPS.default_noise(),
    enrich_maxdim::Int=5,
    enrich_cutoff::Float64=1e-10,
    resid_tol::Float64=1e-8,
    resid_damp::Float64=1.1,
    kick_tol::Float64=1e-8,
    stop_tol::Union{Float64,Nothing}=nothing,
)
    reduced_operator = copy(reduced_operator)
    state = copy(state)
    N = length(state)
    isforward = ITensorMPS.isforward(direction)

    # We only operate in nsite=1 for the solve, but nsite=2 for the residual.
    # So we don't fix nsite globally in the loop, we toggle it per site.

    # The orthogonality limits of `state` are kept up to date by the bond steps below, so this only fires
    # for the first half-sweep (or for a non-orthogonal initial guess). Otherwise the state is already
    # orthogonal at the right site, and the cached environments are still valid.
    if isforward
        if !isortho(state) || orthocenter(state) != 1
            orthogonalize!(state, 1)
            _clear_cache!(reduced_operator)
        end
    else
        if !isortho(state) || orthocenter(state) != N
            orthogonalize!(state, N)
            _clear_cache!(reduced_operator)
        end
    end

    maxtruncerr = 0.0
    # Largest local residual (evaluated before updating each core) of the half-sweep; NaN if not tracked
    maxlocres = isnothing(stop_tol) ? NaN : 0.0
    info = (; maxtruncerr, maxlocres)

    for b in ITensorMPS.sweep_bonds(direction, N; ncenter=1)
        # 1. Optimize state[b] (1-site)
        set_nsite!(reduced_operator, 1)
        position!(reduced_operator, state, b)
        rhs = _local_rhs(reduced_operator)

        if !isnothing(stop_tol)
            norm_rhs = norm(rhs)
            locres = _local_residual_norm(reduced_operator, state[b], rhs)
            maxlocres = max(maxlocres, norm_rhs > 0 ? locres / norm_rhs : locres)
        end

        internal_kwargs = (; current_time, time_step, outputlevel)

        state[b], step_info = updater(reduced_operator, state[b]; internal_kwargs, updater_kwargs...)

        # 2. Truncate the solution, enrich it and shift the ortho center to the next site
        nb = isforward ? b + 1 : b - 1
        if 1 <= nb <= N
            truncerr = _amen_truncate_enrich_shift!(
                state,
                reduced_operator,
                b,
                nb,
                rhs;
                maxdim,
                mindim,
                cutoff,
                enrich_maxdim,
                enrich_cutoff,
                resid_tol,
                resid_damp,
                kick_tol,
            )
            maxtruncerr = max(maxtruncerr, truncerr)
        else
            # End of the half-sweep: nothing to enrich or shift, the center stays here
            ITensorMPS.set_ortho_lims!(state, b:b)
        end
        info = (; maxtruncerr, maxlocres)

        if outputlevel >= 2
            @printf("Sweep %d, direction %s, bond (%d) AMEn enrich\n", sweep, direction, b)
            flush(stdout)
        end

        ITensorMPS.update_observer!(
            observer!;
            state,
            reduced_operator,
            bond=b,
            sweep,
            half_sweep=isforward ? 1 : 2,
            spec=nothing,
            outputlevel,
            half_sweep_is_done=ITensorMPS.is_half_sweep_done(direction, b, N; ncenter=1),
            current_time,
            info,
        )
    end
    normalize && normalize!(state)
    return state, reduced_operator, info
end

"""
    _slice_index(T, i, ir)

Return the part of `T` with the first `dim(ir)` values of its index `i`, with `i` replaced by `ir`.
"""
function _slice_index(T::ITensor, i::Index, ir::Index)
    rest = uniqueinds(T, i)
    A = array(permute(T, rest..., i))
    return ITensor(A[ntuple(_->Colon(), ndims(A)-1)..., 1:dim(ir)], rest..., ir)
end

"""
    _truncate_svd(U, S, V, r)

Truncate the SVD `U * S * V` to its `r` largest singular values, without recomputing it. Returns
`(U_r, SV_r)` where `U_r` has orthonormal columns and `SV_r = S_r * V_r`, both sharing a bond index of
dimension `r`.
"""
function _truncate_svd(U::ITensor, S::ITensor, V::ITensor, r::Int)
    u = commonind(U, S)
    SV = S * V
    dim(u) == r && return U, SV
    ur = Index(r, tags(u))
    return _slice_index(U, u, ur), _slice_index(SV, u, ur)
end

"""
    _amen_truncate_enrich_shift!(state, reduced_operator, b, nb, rhs; kwargs...)

AMEn step for the bond between site `b` (holding the orthogonality center and the freshly solved core)
and its neighbor `nb = b ± 1`. It is direction-agnostic: the same code handles the forward
(`nb = b + 1`) and the reverse (`nb = b - 1`) half-sweeps.

 1. Truncate `state[b]` alone by the local residual (`_residual_rank`), so that the truncation depends
    neither on the scale of `x` nor on the enrichment. `reduced_operator` must be positioned at `b`
    with `nsite = 1`, and `rhs` is its local right-hand side.
 2. Enrich the truncated core with up to `enrich_maxdim` directions of the residual (skipped if the
    residual is below `kick_tol`, or if the bond has no room left below `maxdim`).
 3. Orthonormalize with a QR, so that `x` is unchanged and the center moves to `nb`.

Returns the relative truncation error of step 1.
"""
function _amen_truncate_enrich_shift!(
    state::MPS,
    reduced_operator,
    b::Int,
    nb::Int,
    rhs::ITensor;
    maxdim::Int,
    mindim::Int,
    cutoff::Real,
    enrich_maxdim::Int,
    enrich_cutoff::Real,
    resid_tol::Real,
    resid_damp::Real,
    kick_tol::Real,
)
    X = state[b]
    linds = uniqueinds(X, state[nb])
    bondtags = "Link,l=$(min(b, nb))"

    # 1. Truncate the solution core alone by the local residual, leaving room for the enrichment
    U, S, V = svd(X, linds; lefttags=bondtags, righttags=bondtags)
    full_rank = dim(commonind(U, S))
    rmax = max(mindim, maxdim - enrich_maxdim)
    if iszero(norm(rhs))
        # No residual to measure: fall back to a plain Frobenius truncation (ITensors' convention)
        svals = _singular_values(S)
        r = clamp(_frobenius_rank(svals, cutoff, mindim), mindim, min(rmax, full_rank))
        truncerr = _truncation_error(svals, r)
    else
        r, truncerr, _ = _residual_rank(
            reduced_operator, X, rhs, U, S, V; resid_tol, resid_damp, frob_weight=cutoff, mindim, maxdim=rmax
        )
    end
    state[b], Xn = _truncate_svd(U, S, V, r)
    state[nb] = Xn * state[nb]

    # 2. Enrich the truncated solution with the residual
    kk = min(enrich_maxdim, maxdim - r, prod(dim, linds) - r)
    if kk > 0
        # TODO(#67) (residual TT, as in Sec. 4.3 of Dolgov & Savostyanov): the residual below is the
        # 2-site residual `b₂ - A₂ (X_b X_nb)`, projected both on the left interface of the solution
        # *and on the solution's own right interface*. This has two problems:
        #   - It is not the residual of the paper (their eq. (14)), which is projected only on the
        #     left interface, and whose right side is either left free (Sec. 4.1) or is the right
        #     interface of a separate low-rank residual TT z̃ (Sec. 4.3). Projecting on the
        #     solution's own right interface only sees what 2-site DMRG sees, so the enrichment can
        #     miss residual components that lie outside the current right interface (measured on a
        #     64×64 QTT Poisson problem: at the left bonds, where the right interface has low rank,
        #     ‖z₂‖/‖z‖ ≈ 0-0.06 while the true residual is not small). The bounds
        #     ω_X ≤ ω_Z ≤ ω_z̃ of Theorems 1-2, and with them the global convergence guarantee, do not
        #     hold for it, and it makes the scheme prone to the stagnation of DMRG in high dimensions
        #     (paper Sec. 5.2, Fig. 4-5), most likely with warm starts and many sites.
        #   - It is expensive: a 2-site mat-vec plus a full SVD of the (n χ) × (n χ) tensor at every
        #     bond, O((n χ)³), only to keep `enrich_maxdim` vectors.
        # It should be done as in the paper: keep a rank-ρ (ρ ≈ 3-5) MPS z̃ approximating the residual
        # `b - A x`, with two extra sets of environments (Z*AX and Z*b) on each side. At each bond,
        #   1. the enrichment is the residual `X_{<k}^*(b - A x)` contracted with the right interface
        #      of z̃ (rather than with the solution's), i.e. the first core of its low-rank
        #      approximation;
        #   2. the core of z̃ is updated by contracting the residual with the left and right
        #      interfaces of z̃, and then orthonormalized (Eqs. (22)-(23) of Dolgov & Savostyanov, Part II).
        # This costs O(ρ n² χ³) per bond (no SVD of the 2-site residual) and restores the global
        # information used by the convergence theory.
        set_nsite!(reduced_operator, 2)
        position!(reduced_operator, state, min(b, nb))
        phi2 = state[b] * state[nb]
        rhs2 = _local_rhs(reduced_operator)
        Z = _local_residual(reduced_operator, phi2, rhs2)

        # Skip the enrichment when the local problem is already solved: this is what stops the rank
        # from growing once converged
        if norm(Z) > kick_tol * norm(rhs2)
            U_z, S_z, _ = svd(Z, linds; maxdim=kk, cutoff=enrich_cutoff, lefttags=bondtags, righttags=bondtags)
            if dim(S_z) > 0
                Ub, Xnb = state[b], state[nb]
                ur = commonind(Ub, Xnb)
                l_z = commonind(U_z, S_z)

                # Expand the core with the residual directions, and pad the neighbor with zeros so
                # that x is unchanged
                zero_block = ITensor(eltype(Xnb), l_z, uniqueinds(Xnb, Ub)...)
                Xb_exp, l1 = directsum(Ub => ur, U_z => l_z; tags=tags(ur))
                Xnb_exp, l2 = directsum(Xnb => ur, zero_block => l_z; tags=tags(ur))

                # QR (no truncation) restores the orthonormality of the core: the subspace is unchanged
                Q, R, _ = qr(Xb_exp, linds; tags=bondtags)
                state[b] = Q
                state[nb] = R * replaceind(Xnb_exp, l2 => l1)
            end
        end
    end

    ITensorMPS.set_ortho_lims!(state, nb:nb)
    return truncerr
end
