using ITensorMPS: operator, constant_term, contract

"""
    KrylovSweepStats()

Container to store Krylov local iteration counts during `alternating_update` sweeps.

The stored layout is `position -> sweep -> visits`, where:

  - `position` is the 1-based local position in the MPS (`lpos + 1`)
  - `sweep` is the 1-based sweep index
  - `visits` contains one or more `numiter` values when a position is visited multiple
    times within a sweep (for example forward and backward half-sweeps)
"""
mutable struct KrylovSweepStats
    numiters::Vector{Vector{Vector{Int}}}
    current_sweep::Int
    last_lpos::Int
    moved_since_sweep_start::Bool
end

KrylovSweepStats() = KrylovSweepStats(Vector{Vector{Vector{Int}}}(), 1, -1, false)

"""
    krylov_numiters(stats::KrylovSweepStats)

Return the nested storage `position -> sweep -> visits` of recorded Krylov `numiter` values.
"""
krylov_numiters(stats::KrylovSweepStats) = stats.numiters

function _ensure_position_capacity!(stats::KrylovSweepStats, position::Int)
    while length(stats.numiters) < position
        push!(stats.numiters, Vector{Vector{Int}}())
    end
    return stats
end

function _ensure_sweep_capacity!(stats::KrylovSweepStats, position::Int, sweep::Int)
    position_data = stats.numiters[position]
    while length(position_data) < sweep
        push!(position_data, Int[])
    end
    return stats
end

# TODO Understand how the different iter values are recorded -> TESTS DISABLED FOR NOW
"""
    record_krylov_numiter!(stats::KrylovSweepStats, lpos::Int, numiter::Integer)

Record one local Krylov iteration count for projected position `lpos`.
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

_krylov_lpos(problem::ReducedLinearProblem) = problem.reduced_operator.lpos
_krylov_lpos(problem::ReducedPrecondLinearProblem) = problem.linear_problem.reduced_operator.lpos
_krylov_lpos(problem::ReducedFAIPrecondLinearProblem) = problem.linear_problem.reduced_operator.lpos

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

Container to store Krylov local condition numbers during `alternating_update` sweeps.

The stored layout is `position -> sweep -> visits`, where each visit records the
computed condition number (maximum_eigval / minimum_eigval) of the local operator.
"""
mutable struct KrylovCondStats
    conds::Vector{Vector{Vector{Float64}}}
    current_sweep::Int
    last_lpos::Int
    moved_since_sweep_start::Bool
end

KrylovCondStats() = KrylovCondStats(Vector{Vector{Vector{Float64}}}(), 1, -1, false)

"""
    krylov_conds(stats::KrylovCondStats)

Return the nested storage `position -> sweep -> visits` of recorded condition numbers.
"""
krylov_conds(stats::KrylovCondStats) = stats.conds

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
    record_krylov_cond!(stats::KrylovCondStats, lpos::Int, cond::Float64)

Record one local condition number for projected position `lpos`.
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

Container to store spectral diagnostics of the local (projected) operators during `alternating_update` sweeps.

One record (a `NamedTuple`) is stored per visit, with the sweep index, `lpos`, the local block dimension, the local
Krylov convergence info and the output of `_local_spectrum`. Unlike `KrylovCondStats`, the records keep the sign of
the eigenvalues, so they show whether the projected preconditioner is positive definite. Blocks larger than
`max_blockdim` are not diagonalized, and their spectral fields are `NaN` (counts are `-1`).
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
    record_krylov_spectrum!(stats::KrylovSpectrumStats, lpos::Int, blockdim::Int, info, spectrum)

Record the local spectrum (`nothing` if the block was not diagonalized) of projected position `lpos`.
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
    spectrum_table(stats::KrylovSpectrumStats) -> Dict{String, Vector}

Column-wise copy of the records (one vector per field), which can be saved without DifferenTT types.
"""
function spectrum_table(stats::KrylovSpectrumStats)
    isempty(stats.records) && return Dict{String,Vector}()
    return Dict{String,Vector}(String(k) => [r[k] for r in stats.records] for k in keys(first(stats.records)))
end
