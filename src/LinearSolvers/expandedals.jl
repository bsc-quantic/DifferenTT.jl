"""
    AbstractALSExpansion

Supertype of the basis expansions of [`ExpandedALS`](@ref): [`NullExpansion`](@ref), [`RandomSPSDexpansion`](@ref),
[`RandomLTexpansion`](@ref) and [`ResidualExpansion`](@ref).

An expansion is a callable `(expansion)(state::MPS) -> MPS` that returns an MPS on the same sites as `state`, usually
the direct sum of `state` with another tensor train, so that its bond dimensions grow by those of the added one.
"""
abstract type AbstractALSExpansion end

"""
    ExpandedALS(; sweep_kwargs=SweepingArgs(nsite=1), updater_kwargs=LocalUpdaterArgs(), expander_func=NullExpansion(),
                updater=KrylovUpdater()) <: AbstractDMRGSolver

One-site ALS whose basis is expanded between sweeps, in the spirit of the ALS(t+z) method of Dolgov & Savostyanov
(2014).

The one-site ALS is cheaper per step than MALS, since its local problems have dimension ``\\chi^2 d`` instead of
``\\chi^2 d^2``, but it cannot change the bond dimension of the solution, so it stalls whenever the initial guess
lacks rank. We therefore expand the tensor train before every sweep after the first, as long as its maximum bond
dimension is below `sweep_kwargs.maxdim`: `expander_func` returns a larger tensor train, which we orthogonalize (and
normalize, if `normalize = true` is passed to [`linearsolve`](@ref)) before the sweep. With
[`ResidualExpansion`](@ref), the added directions come from the residual, as in ALS(t+z); [`NullExpansion`](@ref)
recovers the plain one-site ALS. Unlike [`AMEn`](@ref), which enriches every bond within the sweep, the expansion
here happens once per sweep and on the whole tensor train. Note that the expansion is not truncated, so each one
adds the full bond dimension of the expanding tensor train, and that every sweep deep-copies the projected problem
and rebuilds its environments from scratch.

# Fields

  - `sweep_kwargs::SweepingArgs`: sweeps and observers; `nsite` must be 1.
  - `updater_kwargs::LocalUpdaterArgs`: local Krylov solver and diagnostics.
  - `expander_func::AbstractALSExpansion`: the expansion applied before each sweep after the first.
  - `updater::AbstractUpdater`: local solver.

# Throws

  - `ArgumentError`: if `sweep_kwargs.nsite != 1`.

# Examples

```julia
solver = ExpandedALS(;
    sweep_kwargs=SweepingArgs(; nsite=1, nsweeps=6, maxdim=20),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true),
    expander_func=ResidualExpansion(A, b; maxdim=4),
)
x = linearsolve(solver, A, b, nothing)
```

See also [`linearsolve`](@ref), [`AMEn`](@ref).
"""
Base.@kwdef struct ExpandedALS <: AbstractDMRGSolver
    sweep_kwargs::SweepingArgs = SweepingArgs(nsite=1)
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

"""
    linearsolve(solver::ExpandedALS, A, b, x0; kwargs...) -> MPS

Solve ``A \\mathbf{x} = \\mathbf{b}`` by one-site ALS sweeps from `x0`, expanding the basis with
`solver.expander_func` before every sweep after the first. The keywords override the fields of `solver.sweep_kwargs`
(see [`linearsolve`](@ref)).
"""
function linearsolve(solver::ExpandedALS, A, b, x0; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_problem = ReducedLinearProblem(A, b)
    updater_kwargs = (; coefficients=(false, true), NamedTuple(solver.updater_kwargs)...)
    sweep_kwargs = linearsolve_sweep_kwargs(solver, A, b, kwargs)
    return alternating_update(
        reduced_problem, init, solver.expander_func; solver.updater, updater_kwargs, sweep_kwargs...
    )
end

"""
    linearsolve(solver::ExpandedALS, A, b, M⁻¹, x0; preconditioner_sites=nothing, kwargs...) -> MPS

Preconditioned variant of `linearsolve(::ExpandedALS, A, b, x0)`: each local problem is preconditioned with the
projection of `M⁻¹` (Sec. 4 of Torrente et al.), and only those whose site is in `preconditioner_sites` when it is
given.
"""
function linearsolve(solver::ExpandedALS, A, b, M⁻¹, x0; kwargs...)
    init = isnothing(x0) ? default_linearsolve_init(b) : x0
    reduced_precond_problem = initialize_reducedprecondlinearproblem(A, b, M⁻¹)
    updater_kwargs, sweep_kwargs = precond_linearsolve_kwargs(solver, A, b, kwargs)
    return alternating_update(
        reduced_precond_problem, init, solver.expander_func; solver.updater, updater_kwargs, sweep_kwargs...
    )
end

"""
    NullExpansion() <: AbstractALSExpansion

Expansion that returns the state unchanged, so that [`ExpandedALS`](@ref) runs the plain one-site ALS.
"""
struct NullExpansion <: AbstractALSExpansion end
(expander::NullExpansion)(state::MPS) = state

"""
    RandomSPSDexpansion(; rng::AbstractRNG=Random.default_rng()) <: AbstractALSExpansion

Expansion that adds to the state the vectorization of a random symmetric positive semidefinite (SPSD) operator,
drawn from `rng`.

It is meant for unknowns that are themselves vectorized operators on QTT sites, such as the approximate inverse
``\\tilde{G}_L`` of [`fai_preconditioner`](@ref): we draw a random MPO ``R`` with sites of dimension 2, form the
SPSD ``R R^\\top``, and fuse its input and output indices into one index of dimension 4 per site. The state must
therefore have sites of dimension 4. The result is the direct sum of the state and this tensor train, so it changes
the state as well as its basis, and the next sweep re-solves the local problems from there.
"""
struct RandomSPSDexpansion{R<:AbstractRNG} <: AbstractALSExpansion
    rng::R
end
RandomSPSDexpansion(; rng::AbstractRNG=Random.default_rng()) = RandomSPSDexpansion(rng)
function (expander::RandomSPSDexpansion)(state::MPS)
    sites = siteinds(state)
    helper_sites = siteinds(2, length(sites))

    rand = random_mpo(expander.rng, helper_sites)
    rand_dagger = transpose_mpo(conj(rand)) # conj performs the element-wise operation
    randSPSD = apply(rand, rand_dagger; alg="naive") # A @ A^† is always symmetric positive semi-definite!

    N = length(randSPSD)
    combiners = [combiner(helper_sites[i], helper_sites[i]'; tags="Site,n=$i") for i in 1:N]
    for i in 1:N
        randSPSD[i] = randSPSD[i] * combiners[i]
        combiner_ind = combinedind(combiners[i])
        replaceind!(randSPSD[i], combiner_ind => sites[i])
    end
    expansion = MPS([randSPSD...])
    return +(state, expansion; alg="directsum")
end

"""
    RandomLTexpansion(; rng::AbstractRNG=Random.default_rng()) <: AbstractALSExpansion

Expansion that adds to the state the vectorization of a random lower-triangular operator, drawn from `rng`.

As [`RandomSPSDexpansion`](@ref), it is meant for vectorized operators on QTT sites, and requires sites of dimension
4: we take the element-wise product of a random MPO with sites of dimension 2 and the lower-triangular mask
[`lower_triang_mpo`](@ref), which makes the expansion lower-triangular, as ``\\tilde{G}_L`` is, and fuse the indices
of each site. The result is the direct sum of the state and this tensor train.
"""
struct RandomLTexpansion{R<:AbstractRNG} <: AbstractALSExpansion
    rng::R
end
RandomLTexpansion(; rng::AbstractRNG=Random.default_rng()) = RandomLTexpansion(rng)
function (expander::RandomLTexpansion)(state::MPS)
    sites = siteinds(state)
    helper_sites = siteinds(2, length(sites))

    rand_mpo = random_mpo(expander.rng, helper_sites)
    lower_mpo = lower_triang_mpo(helper_sites)
    randLT_mpo = tt_mul(rand_mpo, lower_mpo)

    N = length(randLT_mpo)
    combiners = [combiner(helper_sites[i], helper_sites[i]'; tags="Site,n=$i") for i in 1:N]
    for i in 1:N
        randLT_mpo[i] = randLT_mpo[i] * combiners[i]
        combiner_ind = combinedind(combiners[i])
        replaceind!(randLT_mpo[i], combiner_ind => sites[i])
    end
    expansion = MPS([randLT_mpo...])
    return +(state, expansion; alg="directsum")
end

"""
    ResidualExpansion(A::MPO, b::MPS; cutoff=1e-8, maxdim=5) <: AbstractALSExpansion

Expansion that enriches the basis of the state with the residual ``\\mathbf{r} = \\mathbf{b} - A\\mathbf{x}``,
truncated to `cutoff` and `maxdim`, without changing the represented vector.

We form the direct sum of ``\\mathbf{x}`` and ``\\mathbf{r}`` after setting the first core of ``\\mathbf{r}`` to zero:
the result still represents ``\\mathbf{x}``, but its bond spaces contain those of the residual, which are the
directions that the next one-site sweep most needs, as in the ALS(t+z) method of Dolgov & Savostyanov (2014). Each
expansion adds at most `maxdim` to every bond dimension.

# Arguments

  - `A::MPO`, `b::MPS`: the system of the solve; `A` must act on the sites of `b`.

# Keywords

  - `cutoff::Float64=1e-8`: SVD cutoff of the truncation of the residual.
  - `maxdim::Int=5`: maximum bond dimension of the truncated residual.

# Throws

  - `ArgumentError`: if the input sites of `A` are not the sites of `b`.
"""
struct ResidualExpansion <: AbstractALSExpansion
    A::MPO
    b::MPS
    cutoff::Float64
    maxdim::Int
    function ResidualExpansion(A::MPO, b::MPS; cutoff=1e-8, maxdim=5)
        siteinds_A = only.(siteinds(A; plev=0))
        siteinds_b = siteinds(b)
        siteinds_A != siteinds_b && throw(ArgumentError("ResidualExpansion: the sites of A and b do not match"))
        return new(A, b, cutoff, maxdim)
    end
end

function (expander::ResidualExpansion)(state::MPS)
    residual = expander.b - apply(expander.A, state)

    truncate!(residual; cutoff=expander.cutoff, maxdim=expander.maxdim)

    # zero-shift condition
    residual[1] .*= 0.0
    return +(state, residual; alg="directsum")
end

"""
    ITensorMPS.alternating_update(operator, init::MPS, expander_func::AbstractALSExpansion; updater, updater_kwargs,
                                  kwargs...)

Sweeping loop of [`ExpandedALS`](@ref): a copy of `ITensorMPS.alternating_update` for `nsite = 1` that applies
`expander_func` before every sweep after the first, while the maximum bond dimension is below `maxdim`. Returns the
final MPS.
"""
function ITensorMPS.alternating_update(
    operator,
    init::MPS,
    expander_func::AbstractALSExpansion;
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
        # TODO(release): this deepcopy is expensive; reset the environments of reduced_operator instead
        reduced_operator = deepcopy(ITensorMPS.reduced_operator(operator))
        if maxlinkdim(state) < maxdim[sweep] && sweep > 1
            state = expander_func(state)
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
                maxdim=maxdim[sweep],
                mindim=mindim[sweep],
                cutoff=cutoff[sweep],
                noise=noise[sweep],
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

        isdone = checkdone(; state, sweep, outputlevel, observer=(observer!), sweep_observer=(sweep_observer!))
        isdone && break
    end
    return state
end
