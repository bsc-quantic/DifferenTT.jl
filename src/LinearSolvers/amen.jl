using ITensorMPS: ITensorMPS, AbstractProjMPO, ReducedLinearProblem
using ITensorMPS: set_nsite!, position!, nsite, site_range, noiseterm
using ITensorMPS: TDVPOrder, sub_time_steps, orderings
using ITensorMPS: disk, maxlinkdim, isortho, orthocenter, orthogonalize!, normalize!, replacebond!
using ITensors: ITensors, ITensor, Index, inds, dag, prime, noprime, dim
using ITensors: commoninds, uniqueinds, factorize, svd, array, order
using ITensors: replaceind, replaceind!, OneITensor, directsum, tags, commonind
using LinearAlgebra: norm, svd as la_svd
using Printf: @printf

"""
    AMEn <: AbstractDMRGSolver

Standard Alternating Minimal Energy (AMEn) for solving linear systems.
"""
Base.@kwdef struct AMEn <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    updater::AbstractUpdater = KrylovUpdater()
    enrichment_maxdim::Int = 5
    enrichment_cutoff::Float64 = 1e-10
end

function linearsolve(solver::AMEn, A::MPO, b::MPS, x0::Union{MPS, Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_problem = ReducedLinearProblem(A, b)
    updater_kwargs = (; coefficients = (false, true), NamedTuple(solver.updater_kwargs)...)
    sweep_kwargs = merge(NamedTuple(solver.sweep_kwargs), values(kwargs))
    
    return amen_alternating_update(
        reduced_problem, init;
        updater = solver.updater,
        updater_kwargs,
        enrich_maxdim = solver.enrichment_maxdim,
        enrich_cutoff = solver.enrichment_cutoff,
        sweep_kwargs...
    )
end

function linearsolve(solver::AMEn, A::MPO, b::MPS, M⁻¹::AbstractPreconditioner, x0::Union{MPS, Nothing}; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_precond_problem = initialize_reducedprecondlinearproblem(A, b, M⁻¹)
    updater_kwargs = (; coefficients = (false, true), NamedTuple(solver.updater_kwargs)...)
    sweep_kwargs = merge(NamedTuple(solver.sweep_kwargs), values(kwargs))
    
    return amen_alternating_update(
        reduced_precond_problem, init;
        updater = solver.updater,
        updater_kwargs,
        enrich_maxdim = solver.enrichment_maxdim,
        enrich_cutoff = solver.enrichment_cutoff,
        sweep_kwargs...
    )
end

function _clear_cache!(P)
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
    if hasproperty(P, :reduced_operator)
        _clear_cache!(P.reduced_operator)
    end
    if hasproperty(P, :reduced_constant_terms)
        for ct in P.reduced_constant_terms
            _clear_cache!(ct)
        end
    end
    if hasproperty(P, :M)
        _clear_cache!(P.M)
    end
end

function amen_alternating_update(
    operator,
    init::MPS;
    updater,
    updater_kwargs = (;),
    enrich_maxdim::Int = 5,
    enrich_cutoff::Float64 = 1e-10,
    nsweeps = ITensorMPS.default_nsweeps(),
    checkdone = ITensorMPS.default_checkdone(),
    write_when_maxdim_exceeds = ITensorMPS.default_write_when_maxdim_exceeds(),
    reverse_step = ITensorMPS.default_reverse_step(),
    time_start = ITensorMPS.default_time_start(),
    time_step = ITensorMPS.default_time_step(),
    order = ITensorMPS.default_order(),
    (observer!) = ITensorMPS.default_observer(),
    (sweep_observer!) = ITensorMPS.default_sweep_observer(),
    outputlevel = ITensorMPS.default_outputlevel(),
    normalize = ITensorMPS.default_normalize(),
    maxdim = ITensorMPS.default_maxdim(),
    mindim = ITensorMPS.default_mindim(),
    cutoff = ITensorMPS.default_cutoff(ITensors.scalartype(init)),
    noise = ITensorMPS.default_noise(),
    nsite = ITensorMPS.default_nsite(),
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
                maxdim = maxdim[sweep],
                mindim = mindim[sweep],
                cutoff = cutoff[sweep],
                noise = noise[sweep],
                enrich_maxdim,
                enrich_cutoff,
            )
        end
        if !isnothing(time_step)
            current_time += time_step
        end
        ITensorMPS.update_observer!(
            sweep_observer!; state, reduced_operator, sweep, outputlevel, current_time
        )
        if outputlevel >= 1
            print("After sweep ", sweep, ":")
            print(" maxlinkdim=", maxlinkdim(state))
            @printf(" maxerr=%.2E", info.maxtruncerr)
            if !isnothing(current_time)
                print(" current_time=", round(current_time; digits = 3))
            end
            print(" time=", round(sweep_elapsed_time; digits = 3))
            println()
            flush(stdout)
        end
        isdone = checkdone(;
            state, sweep, outputlevel, observer = (observer!), sweep_observer = (sweep_observer!), info
        )
        isdone && break
    end
    return state
end


function amen_sweep_update(
    order::ITensorMPS.TDVPOrder,
    reduced_operator,
    state::MPS;
    current_time = nothing,
    time_step = nothing,
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
            time_step = sub_time_step,
            kwargs...,
        )
        if !isnothing(time_step)
            current_time += sub_time_step
        end
    end
    return state, reduced_operator, info
end


function amen_sub_sweep_update(
    direction::Base.Ordering,
    reduced_operator,
    state::MPS;
    updater,
    updater_kwargs,
    sweep = ITensorMPS.default_sweep(),
    current_time = nothing,
    time_step = nothing,
    nsite = 1,
    reverse_step = ITensorMPS.default_reverse_step(),
    normalize = ITensorMPS.default_normalize(),
    (observer!) = ITensorMPS.default_observer(),
    outputlevel = ITensorMPS.default_outputlevel(),
    maxdim = ITensorMPS.default_maxdim(),
    mindim = ITensorMPS.default_mindim(),
    cutoff = ITensorMPS.default_cutoff(ITensors.scalartype(state)),
    noise = ITensorMPS.default_noise(),
    enrich_maxdim::Int = 5,
    enrich_cutoff::Float64 = 1e-10,
)
    reduced_operator = copy(reduced_operator)
    state = copy(state)
    N = length(state)
    
    # We only operate in nsite=1 for the solve, but nsite=2 for the residual.
    # So we don't fix nsite globally in the loop, we toggle it per site.

    if ITensorMPS.isforward(direction)
        if !isortho(state) || orthocenter(state) != 1
            orthogonalize!(state, 1)
            _clear_cache!(reduced_operator)
        end
    elseif ITensorMPS.isreverse(direction)
        if !isortho(state) || orthocenter(state) != N
            orthogonalize!(state, N)
            _clear_cache!(reduced_operator)
        end
    end

    maxtruncerr = 0.0
    info = (; maxtruncerr)

    for b in ITensorMPS.sweep_bonds(direction, N; ncenter = 1)
        # 1. Optimize state[b] (1-site)
        set_nsite!(reduced_operator, 1)
        position!(reduced_operator, state, b)
        
        internal_kwargs = (; current_time, time_step, outputlevel)

        state[b], step_info = updater(
            reduced_operator, state[b]; internal_kwargs, updater_kwargs...
        )
        
        # 2. Enrich and shift ortho center
        if ITensorMPS.isforward(direction)
            if b < N
                # Compute 2-site residual
                set_nsite!(reduced_operator, 2)
                position!(reduced_operator, state, b)
                
                phi2 = state[b] * state[b+1]
                rhs = _local_rhs(reduced_operator)
                Aphi = noprime(ITensors.contract(reduced_operator, phi2))
                Z = rhs - Aphi
                
                # Truncate Z to extract expansion basis
                left_inds = uniqueinds(inds(state[b]), inds(state[b+1]))
                U_z, S_z, V_z = svd(Z, left_inds; maxdim=enrich_maxdim, cutoff=enrich_cutoff, lefttags="Link,l=$b", righttags="Link,l=$b")
                
                if dim(S_z) > 0
                    l_old = commonind(state[b], state[b+1])
                    l_z = commonind(U_z, S_z)
                    
                    # Expand the right tensor with 0.0 in the correct shape given by S and V
                    new_state_b, l_new1 = directsum(state[b] => l_old, U_z => l_z, tags=tags(l_old))
                    new_state_b1, l_new2 = directsum(state[b+1] => l_old, (0.0 * S_z * V_z) => l_z, tags=tags(l_old))
                    state[b] = new_state_b
                    state[b+1] = replaceind(new_state_b1, l_new2 => l_new1)
                end
                
                # Shift ortho center to b+1 via SVD
                new_left_inds = uniqueinds(inds(state[b]), inds(state[b+1]))
                U, S, V = svd(state[b], new_left_inds; maxdim, mindim, cutoff, lefttags="Link,l=$b", righttags="Link,l=$b")
                state[b] = U
                state[b+1] = (S * V) * state[b+1]
            end
        else # reverse direction
            if b > 1
                # Compute 2-site residual for bond (b-1, b)
                set_nsite!(reduced_operator, 2)
                position!(reduced_operator, state, b-1)

                phi2 = state[b-1] * state[b]
                rhs = _local_rhs(reduced_operator)
                Aphi = noprime(ITensors.contract(reduced_operator, phi2))
                Z = rhs - Aphi

                # Truncate Z to extract expansion basis
                left_inds = uniqueinds(inds(state[b-1]), inds(state[b]))
                U_z, S_z, V_z = svd(Z, left_inds; maxdim=enrich_maxdim, cutoff=enrich_cutoff, lefttags="Link,l=$(b-1)", righttags="Link,l=$(b-1)")
                
                if dim(S_z) > 0
                    l_old = commonind(state[b-1], state[b])
                    l_z = commonind(S_z, V_z)
                    
                    # Expand the left tensor with 0.0 in the correct shape givne by U and S
                    new_state_b_1, l_new1 = directsum(state[b-1] => l_old, (0.0 * U_z * S_z) => l_z, tags=tags(l_old))
                    new_state_b, l_new2 = directsum(state[b] => l_old, V_z => l_z, tags=tags(l_old))
                    state[b-1] = new_state_b_1
                    state[b] = replaceind(new_state_b, l_new2 => l_new1)
                end
                
                # Shift ortho center to b-1 via SVD. In this case the factors lie in the tensor train as V S U (svd input inds go to U)
                new_right_inds = uniqueinds(inds(state[b]), inds(state[b-1]))
                U, S, V = svd(state[b], new_right_inds; maxdim, mindim, cutoff, lefttags="Link,l=$(b-1)", righttags="Link,l=$(b-1)")
                state[b] = U
                state[b-1] = state[b-1] * (V * S)
            end
        end

        if outputlevel >= 2
            @printf("Sweep %d, direction %s, bond (%d) AMEn enrich\\n", sweep, direction, b)
            flush(stdout)
        end

        ITensorMPS.update_observer!(
            observer!; state, reduced_operator,
            bond = b, sweep,
            half_sweep = ITensorMPS.isforward(direction) ? 1 : 2,
            spec = nothing, outputlevel,
            half_sweep_is_done = ITensorMPS.is_half_sweep_done(direction, b, N; ncenter = 1),
            current_time, info,
        )
    end
    normalize && normalize!(state)
    return state, reduced_operator, (; maxtruncerr)
end