"""
    point_position(point::NTuple{N,Number}, grid::Grid{N}) -> Vector{Int}

Return the 0-based position, along each variable, of the grid node closest to `point`, i.e. `round(x_i / h_i)` after
clamping ``x_i`` to the domain ``[0, L_i]`` of `grid`.
"""
function point_position(point::NTuple{N,Number}, grid::Grid{N}) where {N}
    return collect(Int(round(clamp(point[i], 0.0, grid.box_size[i]) / grid.dx[i])) for i in 1:N)
end

"""
    point_sequence(point::NTuple{N,Number}, grid::Grid{N}, arch::QTTArchitecture{N}) -> Vector{Int}
    point_sequence(point::NTuple{N,Number}, grid::Grid{N}, arch::TTArchitecture{N}) -> Vector{Int}

Return the 0-based value of every site at the grid node closest to `point` ([`point_position`](@ref)), in site order:
the bits of each variable for a [`QTTArchitecture`](@ref), placed as its `site_map` says and reversed for reversed
variables, or the position of the variable carried by each site for a [`TTArchitecture`](@ref). Adding 1 gives the
site values that select that entry of an MPS. The inverse map is [`sequence_to_point`](@ref).

# Examples

```julia
arch = QTTArchitecture((2, 2); interleaved=true)
grid = Grid(arch, (3.0, 3.0))  # 4 × 4 nodes, h = 1
DifferenTT.point_sequence((1.0, 2.0), grid, arch)  # [0, 1, 1, 0]: x = 01, y = 10, interleaved
```
"""
function point_sequence(point::NTuple{N,Number}, grid::Grid{N}, arch::QTTArchitecture{N}) where {N}
    position = point_position(point, grid)
    binary_repr = collect(decimal_to_binary(position[i], abs(arch.var_nsites[i])) for i in 1:N)
    reversed_repr = collect(arch.var_nsites[i] < 0 ? reverse(binary_repr[i]) : binary_repr[i] for i in 1:N)

    sequence = Int64[]
    for var in arch.site_map
        if !isempty(reversed_repr[var])
            push!(sequence, popfirst!(reversed_repr[var]))
        end
    end
    return sequence
end

"""
    sequence_to_point(sequence::Vector{Int}, grid::Grid{N}, arch::QTTArchitecture{N}) -> NTuple{N,Float64}
    sequence_to_point(sequence::Vector{Int}, grid::Grid{N}, arch::TTArchitecture{N}) -> NTuple{N,Float64}

Return the coordinates ``(i_1 h_1, \\dots, i_N h_N)`` of the grid node whose 0-based site values, in site order, are
`sequence`; the inverse of [`point_sequence`](@ref) on grid nodes. The cross interpolation wrappers use it to evaluate
a function of the coordinates at the multi-indices that TCI requests.
"""
function sequence_to_point(sequence::Vector{Int}, grid::Grid{N}, arch::QTTArchitecture{N}) where {N}
    bits_per_var = Dict{Int,Vector{Int}}()

    for (site_idx, var_idx) in enumerate(arch.site_map)
        if !haskey(bits_per_var, var_idx)
            bits_per_var[var_idx] = Int[]
        end
        push!(bits_per_var[var_idx], sequence[site_idx])
    end

    positions = zeros(Int, N)
    for i in 1:N
        bits = bits_per_var[i]
        if arch.var_nsites[i] < 0
            reverse!(bits)
        end
        positions[i] = binary_to_decimal(bits)
    end

    point = ntuple(i -> positions[i] * grid.dx[i], N)
    return point
end

function point_sequence(point::NTuple{N,Number}, grid::Grid{N}, arch::TTArchitecture{N}) where {N}
    position = point_position(point, grid)
    sequence = zeros(Int, length(arch.site_map))
    for (site_idx, var_idx) in enumerate(arch.site_map)
        sequence[site_idx] = position[var_idx]
    end
    return sequence
end

function sequence_to_point(sequence::Vector{Int}, grid::Grid{N}, arch::TTArchitecture{N}) where {N}
    positions = zeros(Int, N)
    for (site_idx, var_idx) in enumerate(arch.site_map)
        positions[var_idx] = sequence[site_idx]
    end
    point = ntuple(i -> positions[i] * grid.dx[i], N)
    return point
end

"""
    TensorCrossInterpolation.crossinterpolate2(func_type::Type{<:Number}, func::Function, grid::Grid{N},
        arch::Architecture; initialpivots=[ones(Int, total_sites(arch))], tolerance::Float64=1e-8,
        maxbonddim::Int=100, maxiter::Int=50, verbosity::Int=0, kwargs...)

Build a tensor train of the function `func` of the coordinates, sampled on the nodes of `grid` and laid out by
`arch`, by the two-site tensor cross interpolation of TensorCrossInterpolation.jl (Núñez Fernández et al., 2025).

Cross interpolation needs only the values of `func` at the entries it selects, and not the full tensor, so it builds
the TT format of a function on a grid far too large to be stored densely, provided that the function has a low-rank
representation in that layout.
Every multi-index requested by TCI (1-based, one value per site) is mapped to the coordinates of its node by
[`sequence_to_point`](@ref), and `func(point::NTuple{N,Float64})` is converted to `func_type`; the requests are
evaluated in batches on all threads. Since TCI starts from `initialpivots`, a function that vanishes on the default
pivot (the node at the origin) needs another one, e.g. from [`optfirstpivot`](@ref
TensorCrossInterpolation.optfirstpivot).

# Keywords

  - `initialpivots::Vector{Vector{Int64}}`: 1-based multi-indices of the initial pivots, `total_sites(arch)` entries
    each.
  - `tolerance::Float64=1e-8`, `maxbonddim::Int=100`, `maxiter::Int=50`, `verbosity::Int=0`, and `kwargs...`: passed to
    `TensorCrossInterpolation.crossinterpolate2`.

# Returns

  - The output of `TensorCrossInterpolation.crossinterpolate2`: the `TensorCI2` object, the ranks and the errors of
    every iteration. `ITensorMPS.MPS(tci)` turns it into an MPS, whose site indices are then replaced by those of the
    problem with `ITensorMPS.replace_siteinds!`.

# Throws

  - `ArgumentError`: if `initialpivots` is empty.
  - `DimensionMismatch`: if a pivot does not have one entry per site.
"""
function TensorCrossInterpolation.crossinterpolate2(
    func_type::Type{<:Number},
    func::Function,
    grid::Grid{N},
    arch::Architecture;
    initialpivots::Vector{Vector{Int64}}=[ones(Int, length(arch.site_map))],
    tolerance::Float64=1e-8,
    maxbonddim::Int=100,
    maxiter::Int=50,
    verbosity::Int=0,
    kwargs...,
) where {N}
    !isempty(initialpivots) || throw(ArgumentError("At least one initial pivot is required."))
    all(length(pivot) == length(arch.site_map) for pivot in initialpivots) || throw(
        DimensionMismatch(
            "Initial pivots must all have $(length(arch.site_map)) entries, got lengths $(length.(initialpivots))."
        ),
    )

    local_dims = collect(Int, site_dims(arch))

    function eval_func(indices)
        sequence = [idx - 1 for idx in indices]
        point = sequence_to_point(sequence, grid, arch)
        return func_type(func(point))
    end

    batchthreaded_func = TensorCrossInterpolation.ThreadedBatchEvaluator{func_type}(eval_func, local_dims)

    return crossinterpolate2(
        func_type,
        batchthreaded_func,
        local_dims,
        initialpivots;
        tolerance=tolerance,
        maxbonddim=maxbonddim,
        maxiter=maxiter,
        verbosity=verbosity,
        kwargs...,
    )
end

"""
    TensorCrossInterpolation.crossinterpolate1(func_type::Type{<:Number}, func::Function, grid::Grid{N},
        arch::Architecture; initialpivots=[ones(Int, total_sites(arch))], tolerance::Float64=1e-8, maxiter::Int=50,
        verbosity::Int=0, kwargs...)

Build a tensor train of the function `func` of the coordinates by the one-site tensor cross interpolation of
TensorCrossInterpolation.jl, as the two-site [`crossinterpolate2`](@ref
TensorCrossInterpolation.crossinterpolate2) does. The first entry of `initialpivots` is the first pivot and the others
are passed as `additionalpivots`. Returns the `TensorCI1` object, the ranks and the errors; throws as
`crossinterpolate2`. It is offered as an alternative with the same interface, to compare the two methods: in our tests
the one-site cross was much less accurate than the two-site one, even when the error it reported was small.
"""
function TensorCrossInterpolation.crossinterpolate1(
    func_type::Type{<:Number},
    func::Function,
    grid::Grid{N},
    arch::Architecture;
    initialpivots::Vector{Vector{Int64}}=[ones(Int, length(arch.site_map))],
    tolerance::Float64=1e-8,
    maxiter::Int=50,
    verbosity::Int=0,
    kwargs...,
) where {N}
    !isempty(initialpivots) || throw(ArgumentError("At least one initial pivot is required."))
    all(length(pivot) == length(arch.site_map) for pivot in initialpivots) || throw(
        DimensionMismatch(
            "Initial pivots must all have $(length(arch.site_map)) entries, got lengths $(length.(initialpivots))."
        ),
    )

    first_pivot = initialpivots[1]
    additionalpivots = initialpivots[2:end]

    local_dims = collect(Int, site_dims(arch))

    function eval_func(indices)
        sequence = [idx - 1 for idx in indices]
        point = sequence_to_point(sequence, grid, arch)
        return func_type(func(point))
    end

    batchthreaded_func = TensorCrossInterpolation.ThreadedBatchEvaluator{func_type}(eval_func, local_dims)

    return crossinterpolate1(
        func_type,
        batchthreaded_func,
        local_dims,
        first_pivot;
        tolerance=tolerance,
        maxiter=maxiter,
        verbosity=verbosity,
        additionalpivots=additionalpivots,
        kwargs...,
    )
end

"""
    _tci_center_pivot(local_dims) -> Vector{Int}

Return the 1-based multi-index at the center of every site, `(d + 1) ÷ 2`. Unlike `ones(Int, N)`, which is the node
at the origin, it is not on the boundary of the domain in a TT layout, where functions with Dirichlet conditions such
as ``\\prod_i \\sin(\\pi x_i)`` vanish; in a QTT layout, where ``d = 2``, it is the all-ones multi-index.
"""
function _tci_center_pivot(local_dims::AbstractVector{Int})::Vector{Int}
    return map(d -> max(1, (d + 1) ÷ 2), local_dims)
end

"""
    _tci_random_pivot(local_dims; rng::AbstractRNG=Random.default_rng()) -> Vector{Int}

Return a random 1-based multi-index, drawn from `rng`. On sites with ``d \\ge 3`` only the interior values `2:(d-1)`
are drawn, since a uniform draw over the full range hits the interior of an ``N``-dimensional domain only with
probability ``((d - 2)/d)^N``.
"""
function _tci_random_pivot(local_dims::AbstractVector{Int}; rng::AbstractRNG=Random.default_rng())::Vector{Int}
    return map(local_dims) do d
        d >= 3 ? rand(rng, 2:(d-1)) : rand(rng, 1:d)
    end
end

"""
    TensorCrossInterpolation.optfirstpivot(func_type::Type{<:Number}, func::Function, grid::Grid{N}, arch::Architecture;
        iters::Int=1000, rng::AbstractRNG=Random.default_rng()) -> Vector{Int}

Return a 1-based multi-index where `|func|` is large, to start the cross interpolation of
[`crossinterpolate2`](@ref TensorCrossInterpolation.crossinterpolate2) from.

TCI fails with `maxsamplevalue is zero!` if every initial pivot is a zero of the function, which the default pivot,
the node at the origin, is for every function that vanishes on the boundary of the domain. We therefore start from the
center multi-index ([`_tci_center_pivot`](@ref)) and keep the largest of `iters` random multi-indices drawn from the
interior ([`_tci_random_pivot`](@ref)). This is a random search, so it only makes a zero pivot unlikely.

# Throws

  - `ArgumentError`: if `func` is zero at every multi-index tried, e.g. because it is identically zero; pass an explicit
    nonzero pivot instead.
"""
function TensorCrossInterpolation.optfirstpivot(
    func_type::Type{<:Number},
    func::Function,
    grid::Grid{N},
    arch::Architecture;
    iters::Int=1000,
    rng::AbstractRNG=Random.default_rng(),
) where {N}
    local_dims = collect(Int, site_dims(arch))

    function eval_func(indices)
        sequence = [idx - 1 for idx in indices]
        point = sequence_to_point(sequence, grid, arch)
        return func_type(func(point))
    end

    pivot = _tci_center_pivot(local_dims)
    valf = abs(eval_func(pivot))

    for _ in 1:iters
        random_pivot = _tci_random_pivot(local_dims; rng)
        newval = abs(eval_func(random_pivot))
        if newval > valf
            valf = newval
            pivot = random_pivot
        end
    end

    if !(valf > 0)
        throw(
            ArgumentError(
                "optfirstpivot: all sampled values were zero (center + $iters interior trials). " *
                "The function may be identically zero on this grid, or vanish on every " *
                "sampled multi-index; provide an explicit nonzero initial pivot.",
            ),
        )
    end

    return pivot
end

"""
    SampleCache{N,ValueType}
    SampleCache(N::Int, ValueType::Type)

Cache of the values of an MPS at points of an ``N``-dimensional grid, keyed by the coordinates of the point, for the
cached method of [`sample`](@ref). It supports `haskey`, `getindex` and `setindex!` by point, and the constructor
returns an empty cache.

# Fields

  - `cache_data::Dict{NTuple{N,Float64},ValueType}`: the stored values.
"""
struct SampleCache{N,ValueType}
    cache_data::Dict{NTuple{N,Float64},ValueType}
end

function SampleCache(N::Int, ValueType::Type)
    return SampleCache{N,ValueType}(Dict{NTuple{N,Float64},ValueType}())
end

function Base.haskey(cache::SampleCache{N,ValueType}, point::NTuple{N,Float64}) where {N,ValueType}
    return haskey(cache.cache_data, point)
end

function Base.getindex(cache::SampleCache{N,ValueType}, point::NTuple{N,Float64}) where {N,ValueType}
    return cache.cache_data[point]
end

function Base.setindex!(cache::SampleCache{N,ValueType}, value::ValueType, point::NTuple{N,Float64}) where {N,ValueType}
    cache.cache_data[point] = value
end

"""
    sample(mps::MPS, point::NTuple{N,Float64}, grid::Grid{N}, arch::Architecture) -> Number
    sample(mps::MPS, point::NTuple{N,Float64}, grid::Grid{N}, arch::Architecture, sample_cache::SampleCache) -> Number

Return the entry of `mps` at the grid node closest to `point`. We select that entry on every site
([`point_sequence`](@ref)) and contract the chain, which costs ``\\mathcal{O}(N d \\chi^2)`` instead of the
``\\mathcal{O}(d^N)`` of a dense contraction. The value is not interpolated between nodes, and the cores are copied to
the CPU one at a time. The second method first looks the point up in `sample_cache` and stores the value it computes
there.

# Throws

  - `DimensionMismatch`: if `mps` does not have `total_sites(arch)` sites.

See also [`samples`](@ref).
"""
function sample(mps::MPS, point::NTuple{N,Float64}, grid::Grid{N}, arch::Architecture) where {N}
    length(mps) == length(arch.site_map) || throw(
        DimensionMismatch(
            "MPS and architecture must have the same number of sites, got $(length(mps)) and $(length(arch.site_map)).",
        ),
    )

    sequence = point_sequence(point, grid, arch)
    state_vals = sequence .+ 1
    s = siteinds(mps)
    nsite = length(mps)

    out = ITensor(1.0)
    for i in 1:nsite
        # on the CPU: one core at a time is copied from the backend, and the result is a scalar
        out *= (adapt(Array, mps[i]) * state(s[i], state_vals[i]))
    end
    return scalar(out)
end

function sample(
    mps::MPS, point::NTuple{N,Float64}, grid::Grid{N}, arch::Architecture, sample_cache::SampleCache{N,ValueType}
) where {N,ValueType}
    if haskey(sample_cache, point)
        return sample_cache[point]
    end

    scalar_out = sample(mps, point, grid, arch)
    sample_cache[point] = scalar_out
    return scalar_out
end

"""
    integrate(mps::MPS) -> Number

Return the sum of all entries of `mps`, by contracting every site with the vector of ones, at a cost of
``\\mathcal{O}(N d \\chi^2)``. It is not weighted by the volume ``\\prod_i h_i`` of a grid cell, which the caller
multiplies by to approximate an integral. The tensors are copied to the CPU first.
"""
function integrate(mps::MPS)
    out = ITensor(1.0)
    s = siteinds(mps)
    nsite = length(mps)
    cpu_mps = adapt(Array, mps)
    for i in 1:nsite
        sum_state = sum([state(s[i], j) for j in 1:dim(s[i])])
        out *= (cpu_mps[i] * sum_state)
    end
    return scalar(out)
end

# TODO(release): use a SampleCache in `samples`
"""
    samples(mps::MPS, grid::Grid{N}, arch::Architecture, npoints::NTuple{N,Int}) -> (ranges, data)
    samples(mps::MPS, grid::Grid{N}, arch::Architecture, npoints::Int) -> (ranges, data)

Evaluate `mps` by [`sample`](@ref) at the points of a regular grid with `npoints[i]` points along variable ``i``,
spanning ``[0, L_i]``, for plotting. Every point is snapped to its closest node, and the points are evaluated on all
threads, on a CPU copy of `mps`. An integer `npoints` uses the same number along every variable.

# Returns

  - `ranges`: the `N` coordinate ranges.
  - `data::Array{Float64,N}`: the values, indexed by variable as the ranges are.
"""
function samples(mps::MPS, grid::Grid{N}, arch::Architecture, npoints::NTuple{N,Int}) where {N}
    ranges = [range(0.0, grid.box_size[i]; length=npoints[i]) for i in 1:N]

    # Initialize data array
    data = zeros(Float64, ntuple(i -> npoints[i], N))

    # Iterate and sample
    indices = CartesianIndices(ntuple(i -> 1:npoints[i], N))
    cpu_mps = adapt(Array, mps)
    Threads.@threads for idx in indices
        point = ntuple(i -> ranges[i][idx[i]], N)
        val = sample(cpu_mps, point, grid, arch)
        data[idx] = val
    end

    return ranges, data
end

function samples(mps::MPS, grid::Grid{N}, arch::Architecture, npoints::Int) where {N}
    return samples(mps, grid, arch, ntuple(_ -> npoints, N))
end
