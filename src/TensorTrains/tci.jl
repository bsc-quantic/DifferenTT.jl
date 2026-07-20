"""
    function point_position(point::NTuple{N,Number}, grid::Grid{N}) where N

Obtain the positions in the given grid that lie closer to the given point.
"""
function point_position(point::NTuple{N,Number}, grid::Grid{N}) where N
    return collect(Int(round(clamp(point[i], 0.0, grid.box_size[i]) / grid.dx[i])) for i in 1:N)
end

"""
    function point_sequence(point::NTuple{N,Number}, grid::Grid{N}, arch::QTTArchitecture{N}) where N

Obtain the bit sequence that represents the position in the given grid of a certain point, according to the specified QTT architecture.
"""
function point_sequence(point::NTuple{N,Number}, grid::Grid{N}, arch::QTTArchitecture{N}) where N
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
    sequence_to_point(sequence::Vector{Int}, grid::Grid{N}, arch::QTTArchitecture{N}) where N

Convert a bit sequence into a vector of points in a grid, given a QTT Architecture.
"""
function sequence_to_point(sequence::Vector{Int}, grid::Grid{N}, arch::QTTArchitecture{N}) where N
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

"""
    function point_sequence(point::NTuple{N,Number}, grid::Grid{N}, arch::TTArchitecture{N}) where N

Obtain the index sequence that represents the position in the given grid of a certain point, according to the specified TT Architecture.
"""
function point_sequence(point::NTuple{N,Number}, grid::Grid{N}, arch::TTArchitecture{N}) where N
    position = point_position(point, grid)
    sequence = zeros(Int, length(arch.site_map))
    for (site_idx, var_idx) in enumerate(arch.site_map)
        sequence[site_idx] = position[var_idx]
    end
    return sequence
end

"""
    sequence_to_point(sequence::Vector{Int}, grid::Grid{N}, arch::TTArchitecture{N}) where N

Convert an index sequence into a vector of points in a grid, given a TT Architecture.
"""
function sequence_to_point(sequence::Vector{Int}, grid::Grid{N}, arch::TTArchitecture{N}) where N
    positions = zeros(Int, N)
    for (site_idx, var_idx) in enumerate(arch.site_map)
        positions[var_idx] = sequence[site_idx]
    end
    point = ntuple(i -> positions[i] * grid.dx[i], N)
    return point
end

"""
    TensorCrossInterpolation.crossinterpolate2(func::Function, grid::Grid{N}, arch::Architecture{N}; tolerance::Float64=1e-8, maxbond::Int=100, verbosity::Int=0, kwargs...)

Interpolate by a 2 site scheme a scalar function `func(point::NTuple{N,Float64})::ValueType` given a specified grid and architecture.
Uses `TensorCrossInterpolation` (TCI) engine.

"""
function TensorCrossInterpolation.crossinterpolate2(func_type::Type{<:Number}, func::Function, grid::Grid{N}, arch::Architecture;
    initialpivots::Vector{Vector{Int64}}=[ones(Int, length(arch.site_map))], tolerance::Float64=1e-8, maxdim::Int=100, maxiter::Int=50, verbosity::Int=0, kwargs...) where N
    @assert !isempty(initialpivots) "At least one initial pivot is required."
    @assert all(length(pivot) == length(arch.site_map) for pivot in initialpivots) "Initial pivots must all have $(length(arch.site_map)) entries, got lengths $(length.(initialpivots))."

    local_dims = collect(Int, site_dims(arch))

    function eval_func(indices)
        sequence = [idx - 1 for idx in indices]
        point = sequence_to_point(sequence, grid, arch)
        return func_type(func(point))
    end

    batchthreaded_func = TensorCrossInterpolation.ThreadedBatchEvaluator{func_type}(eval_func, local_dims)

    return crossinterpolate2(func_type, batchthreaded_func, local_dims, initialpivots;
        tolerance=tolerance,
        maxbonddim=maxdim,
        maxiter=maxiter,
        verbosity=verbosity,
        kwargs...)
end

"""
    TensorCrossInterpolation.crossinterpolate1(func::Function, grid::Grid{N}, arch::Architecture{N}; tolerance::Float64=1e-8, maxbond::Int=100, verbosity::Int=0, kwargs...)

Interpolate a scalar function `func(point::NTuple{N,Float64})::ValueType` given a specified grid and architecture.
Uses `TensorCrossInterpolation` (TCI) engine.

"""
function TensorCrossInterpolation.crossinterpolate1(func_type::Type{<:Number}, func::Function, grid::Grid{N}, arch::Architecture;
    initialpivots::Vector{Vector{Int64}}=[ones(Int, length(arch.site_map))], tolerance::Float64=1e-8, maxiter::Int=50, verbosity::Int=0, kwargs...) where N

    @assert !isempty(initialpivots) "At least one initial pivot is required."
    @assert all(length(pivot) == length(arch.site_map) for pivot in initialpivots) "Initial pivots must all have $(length(arch.site_map)) entries, got lengths $(length.(initialpivots))."

    first_pivot = initialpivots[1]
    additionalpivots = initialpivots[2:end]

    local_dims = collect(Int, site_dims(arch))

    function eval_func(indices)
        sequence = [idx - 1 for idx in indices]
        point = sequence_to_point(sequence, grid, arch)
        return func_type(func(point))
    end

    batchthreaded_func = TensorCrossInterpolation.ThreadedBatchEvaluator{func_type}(eval_func, local_dims)

    return crossinterpolate1(func_type, batchthreaded_func, local_dims, first_pivot;
        tolerance=tolerance,
        maxiter=maxiter,
        verbosity=verbosity,
        additionalpivots=additionalpivots,
        kwargs...)
end

"""
    TensorCrossInterpolation.optfirstpivot(func::Function, grid::Grid{N}, arch::Architecture) where N

Optimize the first pivot to initialize a cross interpolation algortihm, using the TCI engine. In contrast to the 
implementation found in the original Julia Package, in here we use a random search, which we have found to work much 
better for out applications.
"""
function TensorCrossInterpolation.optfirstpivot(func_type::Type{<:Number}, func::Function, grid::Grid{N}, arch::Architecture; iters=1000) where N

    local_dims = collect(Int, site_dims(arch))

    function eval_func(indices)
        sequence = [idx - 1 for idx in indices]
        point = sequence_to_point(sequence, grid, arch)
        return func_type(func(point))
    end

    firstpivot = ones(Int, length(local_dims))

    valf = abs(eval_func(firstpivot))
    pivot = copy(firstpivot)

    for _ in 1:iters
		random_pivot = map(x -> rand(1:x), local_dims)
		newval = abs(eval_func(random_pivot))
		if newval > valf
			valf = newval
			pivot = random_pivot
		end
	end

    return pivot
end


"""
    SampleCache{N,ValueType}

A simple dictionary-based cache that stores previously sampled values keyed by their N-dimensional point coordinates. Avoids redundant MPS sampling evaluations.

# Fields
- `cache_data::Dict{NTuple{N,Float64},ValueType}`: The underlying cache storage.
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
    sample(mps::MPS, point::NTuple{N,Float64}, grid::Grid{N}, arch::Architecture) where N

Evaluate an MPS at a specific physical point by converting the point to a sequence and contracting the MPS.

# Returns
- `Number`: The scalar value of the MPS at the given point.
"""
function sample(mps::MPS, point::NTuple{N,Float64}, grid::Grid{N}, arch::Architecture) where N
    @assert length(mps) == length(arch.site_map) "MPS and architecture must have the same number of sites, got $(length(mps)) and $(length(arch.site_map))."

    sequence = point_sequence(point, grid, arch)
    state_vals = sequence .+ 1
    s = siteinds(mps)
    nsite = length(mps)

    out = ITensor(1.)
    for i in 1:nsite
        out *= (mps[i] * state(s[i], state_vals[i]))
    end
    return scalar(out)
end

"""
    sample(mps::MPS, point::NTuple{N,Float64}, grid::Grid{N}, arch::Architecture, sample_cache::SampleCache{N,ValueType}) where {N,ValueType}

Evaluate an MPS at a specific physical point, using a `SampleCache` to avoid redundant evaluations.
"""
function sample(mps::MPS, point::NTuple{N,Float64}, grid::Grid{N}, arch::Architecture, sample_cache::SampleCache{N,ValueType}) where {N,ValueType}
    if haskey(sample_cache, point)
        return sample_cache[point]
    end

    scalar_out = sample(mps, point, grid, arch)
    sample_cache[point] = scalar_out
    return scalar_out
end

"""
    integrate(mps::MPS)

Compute the sum of all entries of an MPS by contracting each site tensor with the sum-state `|0⟩ + |1⟩`...
Does not account for the physical domain volume (grid spacing).

# Returns
- `Number`: The integral (sum) of the MPS entries.
"""
function integrate(mps::MPS)
    out = ITensor(1.)
    s = siteinds(mps)
    nsite = length(mps)
    cpu_mps = adapt(Array, mps)
    for i in 1:nsite
        sum_state = sum([state(s[i], j) for j in 1:dim(s[i])])
        out *= (cpu_mps[i] * sum_state)
    end
    return scalar(out)
end

# TODO Implement Sample caches in samples
"""
    samples(mps::MPS, grid::Grid{N}, arch::Architecture, npoints::NTuple{N, Int}) where N

Helper function to sample an MPS on a regular grid of `npoints` per dimension.
Returns a tuple `(ranges, data)`.
"""
function samples(mps::MPS, grid::Grid{N}, arch::Architecture, npoints::NTuple{N,Int}) where N
    ranges = [range(0.0, grid.box_size[i], length=npoints[i]) for i in 1:N]

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

function samples(mps::MPS, grid::Grid{N}, arch::Architecture, npoints::Int) where N
    return samples(mps, grid, arch, ntuple(_ -> npoints, N))
end
