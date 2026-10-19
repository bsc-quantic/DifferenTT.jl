abstract type Architecture end

"""
    struct QTTArchitecture{N}

Definition of the physical-to-quantized mapping for N spatial dimensions.

# Fields

  - `var_nbits::NTuple{N,Int}`: Number of bits (qubits) per spatial dimension. If negative, it indicates that the variable tensors appear in reverse order.
  - `var_map::Dict{Int,Int}`: Mapping from logical dimension index to physical position block.
  - `site_map::Vector{Int}`: Explicit mapping of each site in the MPS to a logical dimension.
  - `is_interleaved::Bool`: Whether the bits of different dimensions are interleaved (e.g. x1, y1, x2, y2) or blocked (x1..xn, y1..yn).
"""
struct QTTArchitecture{N} <: Architecture where {N}
    var_nsites::NTuple{N,Int}
    var_map::Dict{Int,Int}
    site_map::Vector{Int}
    is_interleaved::Bool
end

"""
    QTTArchitecture(var_nsites::NTuple{N,Int}; var_map::Union{Dict{Int,Int},Nothing}=nothing, interleaved::Bool=false) where N

Construct an `QTTArchitecture` object.

# Arguments

  - `var_nsites`: Tuple containing the number of bits for each dimension.
  - `var_map`: Optional mapping for variable ordering. Defaults to 1:N.
  - `interleaved`: If true, bits from different dimensions are interleaved.
"""
function QTTArchitecture(
    var_nsites::NTuple{N,Int}, var_map::Union{Dict{Int,Int},Nothing}=nothing; interleaved::Bool=false
) where {N}
    if isnothing(var_map)
        var_map = Dict(i => i for i in 1:N)
    end

    @assert length(var_map) == N "Variable map must have length $N"
    @assert Set(keys(var_map)) == Set(1:N) "Variable map keys must be 1:$N"
    @assert Set(values(var_map)) == Set(1:N) "Variable map values must be 1:$N"

    total_sites = sum(abs, var_nsites)
    site_map = zeros(Int, total_sites)

    if interleaved
        # Sort variables by their mapped "physical" position order
        ordered_vars = sort(collect(keys(var_map)); by=v -> var_map[v])

        # General filler: cycle through ordered_vars, pop one site if available.
        # Does not assume that the variables all have the same number of sites.

        # Remaining logical sites to fill
        remaining_sites = collect(abs.(var_nsites)) # makes a copy

        cursor = 1
        while sum(remaining_sites) > 0
            for v in ordered_vars # Iterate in physical order
                if remaining_sites[v] > 0
                    site_map[cursor] = v
                    remaining_sites[v] -= 1
                    cursor += 1
                end
            end
        end

    else
        # Blocked: Var_A_sites..., Var_B_sites...
        # Ordering determined by var_map values: 1 -> 1st block, 2 -> 2nd block...

        # Invert map to get: Physical Pos -> Logical Var
        phys_to_logical = Dict(v => k for (k, v) in var_map)

        cursor = 1
        for p in 1:N
            v = phys_to_logical[p]
            n_sites = abs(var_nsites[v]) # Assign in physical position the n_sites assocaited with the logical variable
            site_map[cursor:(cursor+n_sites-1)] .= v
            cursor += n_sites
        end
    end

    return QTTArchitecture{N}(var_nsites, var_map, site_map, interleaved)
end

"""
    struct TTArchitecture{N}

Definition of the physical-to-TT mapping for N spatial dimensions.

# Fields

  - `var_dims::NTuple{N,Int}`: Dimension of each spatial dimension.
  - `var_map::Dict{Int,Int}`: Mapping from logical dimension index to physical position block.
"""
struct TTArchitecture{N} <: Architecture
    var_dims::NTuple{N,Int}
    var_map::Dict{Int,Int}
    site_map::Vector{Int}

    function TTArchitecture(var_dims::NTuple{N,Int}, var_map::Union{Dict{Int,Int},Nothing}=nothing) where {N}
        if isnothing(var_map)
            var_map = Dict(i=>i for i in 1:N)
        else
            @assert Set(keys(var_map)) == Set(1:N) "Variable map keys must be 1:$N"
            @assert Set(values(var_map)) == Set(1:N) "Variable map values must be 1:$N"
        end

        site_map = zeros(Int, N)
        for (log_var, phys_pos) in var_map
            site_map[phys_pos] = log_var
        end

        return new{N}(var_dims, var_map, site_map)
    end
end

nsites_per_variable(arch::QTTArchitecture, var::Int) = abs(arch.var_nsites[var])
nsites_per_variable(arch::TTArchitecture, var::Int) = 1

total_sites(arch::QTTArchitecture) = length(arch.site_map)
total_sites(arch::TTArchitecture{N}) where {N} = N

site_dim(arch::QTTArchitecture, site_idx::Int) = 2
site_dim(arch::TTArchitecture, site_idx::Int) = arch.var_dims[arch.site_map[site_idx]]

site_dims(arch::QTTArchitecture) = fill(2, total_sites(arch))
site_dims(arch::TTArchitecture) = arch.var_dims

variable_dims(arch::QTTArchitecture) = map(x -> 2^abs(x), arch.var_nsites)
variable_dims(arch::TTArchitecture) = arch.var_dims

"""
    site_indices(dims::AbstractVector{<:Integer}; kwargs...) -> Vector{Index{Int}}

Return one site index of dimension `dims[n]` per site, tagged `"Site,n=\$n"`. The keywords are passed to `Index`.
"""
function site_indices(dims::AbstractVector{<:Integer}; kwargs...)
    return [Index(d, "Site,n=$n"; kwargs...) for (n, d) in enumerate(dims)]
end

"""
    siteinds(arch::Architecture; kwargs...) -> Vector{Index{Int}}

Return the site indices of `arch`, with the dimensions [`site_dims`](@ref)`(arch)`.
"""
function ITensorMPS.siteinds(arch::Architecture; kwargs...)
    return site_indices(collect(Int, site_dims(arch)); kwargs...)
end

"""
    link_indices(dims::AbstractVector{<:Integer}; kwargs...) -> Vector{Index{Int}}

Return one link index of dimension `dims[l]` per bond, tagged `"Link,l=\$l"`. The keywords are passed to `Index`.
"""
function link_indices(dims::AbstractVector{<:Integer}; kwargs...)
    return [Index(d, "Link,l=$(n)"; kwargs...) for (n, d) in enumerate(dims)]
end
