using ITensorMPS: alternating_update, ReducedLinearProblem
"""
    AbstractALSExpansion

Abstract type for tensor train expansion strategies in `ExpandedALS`.
"""
abstract type AbstractALSExpansion end

"""
    ExpandedALS <: AbstractDMRGSolver

ALS(t + z) for solving linear systems. It executes a 1-site DMRG and at the end of the sweep, expands the current
tensor train with another one given by the expander func.
"""
Base.@kwdef struct ExpandedALS <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs()
    updater_kwargs::LocalUpdaterArgs = LocalUpdaterArgs()
    expander_func::AbstractALSExpansion = NullExpansion()
    updater::AbstractUpdater = KrylovUpdater()

    function ExpandedALS(sweep_kwargs, updater_kwargs, expander_func, updater)
        if sweep_kwargs.nsite != 1
            throw(ArgumentError("ALS(t+z) currently only supports nsite = 1."))
        end
        return new(sweep_kwargs, updater_kwargs, expander_func, updater)
    end
end

function linearsolve(solver::ExpandedALS, A, b, x0; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_problem = ReducedLinearProblem(A, b)
    updater_kwargs = (; coefficients = (false, true), NamedTuple(solver.updater_kwargs)...)
    return alternating_update(reduced_problem, init, solver.expander_func; solver.updater, updater_kwargs, NamedTuple(solver.sweep_kwargs)...)
end

# TODO Check if this works
function linearsolve(solver::ExpandedALS, A, b, M⁻¹, x0; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_precond_problem = initialize_reducedprecondlinearproblem(A, b, M⁻¹)
    updater_kwargs = (; coefficients = (false, true), updater_kwargs...)
    return alternating_update(reduced_precond_problem, init, solver.expander_func; solver.updater, updater_kwargs, NamedTuple(solver.sweep_kwargs)...)
end

"""
    NullExpansion <: AbstractALSExpansion

An expansion strategy that does nothing.
"""
struct NullExpansion <: AbstractALSExpansion end
(expander::NullExpansion)(sites::Vector{<:Index}) = nothing

"""
    RandomSPSDexpansion <: AbstractALSExpansion

An expansion strategy that expands the tensor train with a random symmetric positive semi-definite (SPSD) tensor train.
"""
struct RandomSPSDexpansion <: AbstractALSExpansion end
function (expander::RandomSPSDexpansion)(sites::Vector{<:Index})
    helper_sites = siteinds(2, length(sites))

    rand = random_mpo(helper_sites)
    rand_dagger = transpose(conj(rand)) # conj performs the element-wise operation
    randSPSD = apply(rand, rand_dagger; alg="naive") # A @ A^† is always symmetric positive semi-definite!

    N = length(randSPSD)
    combiners = [combiner(helper_sites[i], helper_sites[i]'; tags="Site,n=$i") for i in 1:N]
    for i in 1:N
        randSPSD[i] = randSPSD[i] * combiners[i]
        combiner_ind = combinedind(combiners[i])
        replaceind!(randSPSD[i], combiner_ind => sites[i])
    end
    return MPS([randSPSD...])
end

"""
    RandomLTexpansion <: AbstractALSExpansion

An expansion strategy that expands the tensor train with a random lower triangular tensor train.
"""
struct RandomLTexpansion <: AbstractALSExpansion end
function (expander::RandomLTexpansion)(sites::Vector{<:Index})
    helper_sites = siteinds(2, length(sites))

    rand_mpo = random_mpo(helper_sites)
    lower_mpo = lower_triang_mpo(helper_sites)
    randLT_mpo = tt_mul(rand_mpo, lower_mpo)

    N = length(randLT_mpo)
    combiners = [combiner(helper_sites[i], helper_sites[i]'; tags="Site,n=$i") for i in 1:N]
    for i in 1:N
        randLT_mpo[i] = randLT_mpo[i] * combiners[i]
        combiner_ind = combinedind(combiners[i])
        replaceind!(randLT_mpo[i], combiner_ind => sites[i])
    end
    return MPS([randLT_mpo...])
end

function ITensorMPS.alternating_update(
    operator,
    init::MPS,
    expander_func::AbstractALSExpansion;
    updater,
    updater_kwargs = (;),
    nsweeps = ITensorMPS.default_nsweeps(),
    checkdone = ITensorMPS.default_checkdone(),
    write_when_maxdim_exceeds = ITensorMPS.default_write_when_maxdim_exceeds(),
    nsite = ITensorMPS.default_nsite(),
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
)
    nsite != 1 && error("Expanded als currently only developed for 1-site variant.")
    if isnothing(nsweeps)
        return error("Must specify `nsweeps`.")
    end
    maxdim, mindim, cutoff, noise = ITensorMPS.process_sweeps(; nsweeps, maxdim, mindim, cutoff, noise)
    forward_order = ITensorMPS.TDVPOrder(order, Base.Forward)
    state = copy(init)
    
    current_time = time_start
    info = nothing
    sites = siteinds(init)

    for sweep in 1:nsweeps
        # TODO This is quite expensive -> Set up an environment cleaner for the reduced_operator instead of a full deepcopy
        reduced_operator = deepcopy(ITensorMPS.reduced_operator(operator))
        if maxlinkdim(state) < maxdim[sweep] && sweep > 1
            expander_mps = expander_func(sites)
            state = isnothing(expander_mps) ? state : +(state, expander_mps; alg="directsum")
            orthogonalize!(state, 1)
            if normalize
                state ./= norm(state)
            end
        end
        if !isnothing(write_when_maxdim_exceeds) && maxdim[sweep] > write_when_maxdim_exceeds
            if outputlevel >= 2
                println(
                    "write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sweep)), writing environment tensors to disk",
                )
            end
            reduced_operator = disk(reduced_operator)
        end
        sweep_elapsed_time = @elapsed begin
            state, reduced_operator, info = ITensorMPS.sweep_update(
                forward_order,
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
            state, sweep, outputlevel, observer = (observer!), sweep_observer = (sweep_observer!)
        )
        isdone && break
    end
    return state
end

