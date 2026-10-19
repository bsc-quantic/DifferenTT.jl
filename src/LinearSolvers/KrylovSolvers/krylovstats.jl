"""
    KrylovSweepStats()

Record of the iteration counts of the local Krylov solves of a DMRG-type solve. Pass it as `numiter_store` in
[`LocalUpdaterArgs`](@ref).

The iteration counts of the local solver are what a preconditioner is meant to reduce, and they vary strongly along
the tensor train (Fig. 11 of Torrente et al.), so we store them per position and per sweep. The layout of
[`krylov_numiters`](@ref) is `position → sweep → visits`:

  - `position` is the first site of the local problem, from 1 to `N - nsite + 1` for a tensor train of `N` sites;
  - `sweep` counts the sweeps from 1;
  - `visits` holds one count per local solve at that position and sweep, in the order of the solves. Every sweep
    visits every position twice, once in the forward half-sweep and once in the reverse one, so a full sweep stores
    two counts per position; the end positions are visited twice in a row, at the turn of the sweep.

We detect a new sweep when the first position is solved again after the solve has moved away from it, since every
sweep starts there. The same object can therefore be reused across solves, whose sweeps are numbered on, but a
tensor train with a single position (`N = nsite`) records all its solves under sweep 1. [`AMEn`](@ref) with
`local_stop` may stop after a forward half-sweep, which leaves one count per position in its last sweep.

See also [`positions_mean`](@ref), [`positions_sum`](@ref), [`KrylovCondStats`](@ref).
"""
mutable struct KrylovSweepStats
    numiters::Vector{Vector{Vector{Int}}}
    current_sweep::Int
    last_lpos::Int
    moved_since_sweep_start::Bool
end

KrylovSweepStats() = KrylovSweepStats(Vector{Vector{Vector{Int}}}(), 1, -1, false)

"""
    krylov_numiters(stats::KrylovSweepStats) -> Vector{Vector{Vector{Int}}}

Return the iteration counts recorded in `stats`, with the layout `position → sweep → visits` described in
[`KrylovSweepStats`](@ref). The returned array is the storage of `stats`, not a copy.
"""
krylov_numiters(stats::KrylovSweepStats) = stats.numiters

"""
    _ensure_position_capacity!(stats, position)

Grow the storage of `stats` with empty entries until it has `position` positions. Returns `stats`.
"""
function _ensure_position_capacity!(stats::KrylovSweepStats, position::Int)
    while length(stats.numiters) < position
        push!(stats.numiters, Vector{Vector{Int}}())
    end
    return stats
end

"""
    _ensure_sweep_capacity!(stats, position, sweep)

Grow the storage of `position` in `stats` with empty entries until it has `sweep` sweeps. Returns `stats`.
"""
function _ensure_sweep_capacity!(stats::KrylovSweepStats, position::Int, sweep::Int)
    position_data = stats.numiters[position]
    while length(position_data) < sweep
        push!(position_data, Int[])
    end
    return stats
end

"""
    record_krylov_numiter!(stats::KrylovSweepStats, lpos::Int, numiter::Integer) -> KrylovSweepStats

Record the iteration count `numiter` of one local solve at the position `lpos + 1`, where `lpos` is the `lpos` of the
projected operator (the number of sites to the left of the local problem). A new sweep starts when `lpos = 0` is
recorded again after other positions; see [`KrylovSweepStats`](@ref).
"""
function record_krylov_numiter!(stats::KrylovSweepStats, lpos::Int, numiter::Integer)
    if stats.last_lpos == 0 && lpos == 0 && stats.moved_since_sweep_start
        stats.current_sweep += 1
        stats.moved_since_sweep_start = false
    end

    if stats.last_lpos != -1 && lpos != stats.last_lpos
        stats.moved_since_sweep_start = true
    end

    position = lpos + 1
    _ensure_position_capacity!(stats, position)
    _ensure_sweep_capacity!(stats, position, stats.current_sweep)
    push!(stats.numiters[position][stats.current_sweep], Int(numiter))

    stats.last_lpos = lpos
    return stats
end

"""
    _krylov_lpos(problem) -> Int

Return the `lpos` of the projected operator of `problem`, i.e. the number of sites to the left of the local problem.
"""
_krylov_lpos(problem::ReducedLinearProblem) = problem.reduced_operator.lpos
_krylov_lpos(problem::ReducedPrecondLinearProblem) = problem.linear_problem.reduced_operator.lpos
_krylov_lpos(problem::ReducedFAIPrecondLinearProblem) = problem.linear_problem.reduced_operator.lpos

"""
    positions_mean(stats::KrylovSweepStats) -> Vector{Float64}

Return, for each position, the mean iteration count over all its recorded local solves (all sweeps and visits), or
`0.0` for a position without any.
"""
function positions_mean(stats::KrylovSweepStats)
    visits_sum = [sum.(v) for v in stats.numiters]
    visits_len = [length.(v) for v in stats.numiters]

    means = Float64[]
    for i in 1:length(visits_sum)
        total_sum = sum(visits_sum[i])
        total_len = sum(visits_len[i])
        push!(means, total_len > 0 ? total_sum / total_len : 0.0)
    end
    return means
end

"""
    positions_sum(stats::KrylovSweepStats) -> Vector{Float64}

Return, for each position, the total iteration count over all its recorded local solves (all sweeps and visits).
"""
function positions_sum(stats::KrylovSweepStats)
    visits_sum = [sum.(v) for v in stats.numiters]

    sums = Float64[]
    for i in 1:length(visits_sum)
        total_sum = sum(visits_sum[i])
        push!(sums, Float64(total_sum))
    end
    return sums
end

"""
    KrylovCondStats()

Record of the condition numbers of the local problems of a DMRG-type solve. Pass it as `cond_store` (for
``\\kappa(A_{\\text{loc}})``) or `pc_cond_store` (for ``\\kappa(M^{-1}_{\\text{loc}} A_{\\text{loc}})``) in
[`LocalUpdaterArgs`](@ref).

Each value is ``\\kappa = \\max|\\lambda| / \\min|\\lambda|`` of the dense local operator, computed with a full
eigendecomposition at a cost of ``\\mathcal{O}(n_{\\text{loc}}^3)``, so it is meant for small diagnostic runs. For CG
and PCG it is the spectral condition number that bounds the convergence. A singular local operator
(``\\kappa = \\infty``) is not recorded. The layout of [`krylov_conds`](@ref) is `position → sweep → visits`, as for
[`KrylovSweepStats`](@ref).

See also [`positions_mean_cond`](@ref), [`KrylovSpectrumStats`](@ref).
"""
mutable struct KrylovCondStats
    conds::Vector{Vector{Vector{Float64}}}
    current_sweep::Int
    last_lpos::Int
    moved_since_sweep_start::Bool
end

KrylovCondStats() = KrylovCondStats(Vector{Vector{Vector{Float64}}}(), 1, -1, false)

"""
    krylov_conds(stats::KrylovCondStats) -> Vector{Vector{Vector{Float64}}}

Return the condition numbers recorded in `stats`, with the layout `position → sweep → visits` of
[`KrylovSweepStats`](@ref). The returned array is the storage of `stats`, not a copy.
"""
krylov_conds(stats::KrylovCondStats) = stats.conds

# Same as the KrylovSweepStats methods, for the condition numbers
function _ensure_position_capacity!(stats::KrylovCondStats, position::Int)
    while length(stats.conds) < position
        push!(stats.conds, Vector{Vector{Float64}}())
    end
    return stats
end

function _ensure_sweep_capacity!(stats::KrylovCondStats, position::Int, sweep::Int)
    position_data = stats.conds[position]
    while length(position_data) < sweep
        push!(position_data, Float64[])
    end
    return stats
end

"""
    record_krylov_cond!(stats::KrylovCondStats, lpos::Int, cond::Float64) -> KrylovCondStats

Record the condition number `cond` of one local problem at the position `lpos + 1`, with the sweep detection of
[`record_krylov_numiter!`](@ref).
"""
function record_krylov_cond!(stats::KrylovCondStats, lpos::Int, cond::Float64)
    if stats.last_lpos == 0 && lpos == 0 && stats.moved_since_sweep_start
        stats.current_sweep += 1
        stats.moved_since_sweep_start = false
    end

    if stats.last_lpos != -1 && lpos != stats.last_lpos
        stats.moved_since_sweep_start = true
    end

    position = lpos + 1
    _ensure_position_capacity!(stats, position)
    _ensure_sweep_capacity!(stats, position, stats.current_sweep)
    push!(stats.conds[position][stats.current_sweep], cond)

    stats.last_lpos = lpos
    return stats
end

"""
    positions_mean_cond(stats::KrylovCondStats) -> Vector{Float64}

Return, for each position, the mean condition number over all its recorded local problems, or `0.0` for a position
without any.
"""
function positions_mean_cond(stats::KrylovCondStats)
    visits_sum = [sum.(v) for v in stats.conds]
    visits_len = [length.(v) for v in stats.conds]

    means = Float64[]
    for i in 1:length(visits_sum)
        total_sum = sum(visits_sum[i])
        total_len = sum(visits_len[i])
        push!(means, total_len > 0 ? total_sum / total_len : 0.0)
    end
    return means
end
"""
    KrylovSpectrumStats(; max_blockdim=4096)

Record of the sign-aware spectra of the local problems of a DMRG-type solve. Pass it as `spectrum_store` in
[`LocalUpdaterArgs`](@ref).

A condition number cannot tell whether a truncated approximate inverse has lost its definiteness (Fig. 3 of
Torrente et al.), so we record the extreme eigenvalues with their sign. The local blocks are congruences
``U^\\top A U`` and ``U^\\top M^{-1} U`` of the global operators, where ``U`` is the basis of the current projection.
By Sylvester's law of inertia, a negative eigenvalue of the symmetric part of the local ``M^{-1}`` therefore
certifies that the global ``M^{-1}`` is not positive definite; the converse does not hold, since a positive local
block only shows that ``M^{-1}`` is positive on the span of ``U``.

We store one record (a `NamedTuple`) per local solve, with the fields

  - `sweep`, `position`, `blockdim`: sweep, first site and dimension ``n_{\\text{loc}}`` of the local problem;
  - `numiter`, `converged`, `normres`: the convergence information of the local Krylov solve;
  - `eigmin_A`, `eigmax_A`, `asym_A`: extreme eigenvalues of the symmetric part of ``A_{\\text{loc}}``, and its
    relative asymmetry ``\\|A - A^\\top\\| / \\|A\\|``;
  - `eigmin_P`, `eigmax_P`, `nneg_P`, `asym_P`: the same for ``M^{-1}_{\\text{loc}}``, with the number of negative
    eigenvalues beyond rounding;
  - `eigmin_PA`, `eigmax_PA`, `nneg_PA`: the spectrum of the PCG operator, when the symmetric part of
    ``A_{\\text{loc}}`` is positive definite.

The spectra cost ``\\mathcal{O}(n_{\\text{loc}}^3)``. Blocks with ``n_{\\text{loc}} >`` `max_blockdim` are not
diagonalized, and the fields that are not computed (all those of ``M^{-1}`` for an unpreconditioned solve) are `NaN`,
or `-1` for the counts. [`spectrum_table`](@ref) returns the records column-wise.
"""
mutable struct KrylovSpectrumStats
    records::Vector{NamedTuple}
    max_blockdim::Int
    current_sweep::Int
    last_lpos::Int
    moved_since_sweep_start::Bool
end

KrylovSpectrumStats(; max_blockdim::Int=4096) = KrylovSpectrumStats(NamedTuple[], max_blockdim, 1, -1, false)

"""
    record_krylov_spectrum!(stats::KrylovSpectrumStats, lpos::Int, blockdim::Int, info, spectrum) -> KrylovSpectrumStats

Record the local spectrum `spectrum` (the output of [`_local_spectrum`](@ref), or `nothing` if the block was not
diagonalized) and the KrylovKit convergence `info` of one local solve at the position `lpos + 1`, with the sweep
detection of [`record_krylov_numiter!`](@ref).
"""
function record_krylov_spectrum!(stats::KrylovSpectrumStats, lpos::Int, blockdim::Int, info, spectrum)
    if stats.last_lpos == 0 && lpos == 0 && stats.moved_since_sweep_start
        stats.current_sweep += 1
        stats.moved_since_sweep_start = false
    end

    if stats.last_lpos != -1 && lpos != stats.last_lpos
        stats.moved_since_sweep_start = true
    end

    spectrum = isnothing(spectrum) ? _EMPTY_LOCAL_SPECTRUM : spectrum
    record = (;
        sweep=stats.current_sweep,
        position=lpos + 1,
        blockdim,
        numiter=Int(info.numiter),
        converged=(info.converged > 0),
        normres=Float64(info.normres),
        spectrum...,
    )
    push!(stats.records, record)

    stats.last_lpos = lpos
    return stats
end

"""
    spectrum_table(stats::KrylovSpectrumStats) -> Dict{String,Vector}

Return the records of `stats` column-wise, one vector per field of [`KrylovSpectrumStats`](@ref), keyed by the field
name. The result holds no DifferenTT types, so it can be saved with JLD2 and read without the package. It is empty if
nothing was recorded.
"""
function spectrum_table(stats::KrylovSpectrumStats)
    isempty(stats.records) && return Dict{String,Vector}()
    return Dict{String,Vector}(String(k) => [r[k] for r in stats.records] for k in keys(first(stats.records)))
end
