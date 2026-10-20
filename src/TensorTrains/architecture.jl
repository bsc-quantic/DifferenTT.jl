"""
    Architecture

Abstract supertype of the layouts that place the variables of an ``N``-dimensional grid on the sites of a tensor
train: the quantized [`QTTArchitecture`](@ref), with one bit of one variable per site, and the
[`TTArchitecture`](@ref), with one variable per site. Every subtype has the fields `var_map` and `site_map`, the latter
giving the variable carried by each site, and implements [`nsites_per_variable`](@ref), [`total_sites`](@ref),
[`site_dim`](@ref), [`site_dims`](@ref) and [`variable_dims`](@ref).
"""
abstract type Architecture end

"""
    QTTArchitecture{N} <: Architecture

Layout of an ``N``-dimensional grid of ``2^{n_1} \\times \\dots \\times 2^{n_N}`` points in the QTT format, where
every site carries one bit of the index of one variable, so that all sites have dimension ``d = 2``.

A variable with ``2^n`` grid points is quantized by the binary expansion ``i = \\sum_{p=1}^{n} i_p 2^{n-p}`` of its
index (Sec. 2 of Torrente et al., *Factorized approximate inverse preconditioners in the tensor train format*), and
therefore takes ``n`` sites. The architecture decides where these sites go along the chain. In the blocked layout the
sites of each variable are contiguous, and the blocks follow the positions given by `var_map`. In the interleaved
layout we take one site of each variable in turn, in the same order, and skip the variables that have run out of
sites. Within a variable, the bits run from the most significant to the least significant along the chain, unless its
entry of `var_nsites` is negative, in which case its first site carries the least significant bit.

# Fields

  - `var_nsites::NTuple{N,Int}`: number of sites (bits) of each variable; a negative entry reverses its bit order.
  - `var_map::Dict{Int,Int}`: `variable => position` of each variable, among the blocks (blocked) or within every
    cycle (interleaved).
  - `site_map::Vector{Int}`: variable carried by each site.
  - `is_interleaved::Bool`: whether the layout is interleaved.

See also [`TTArchitecture`](@ref), [`site_indices`](@ref).
"""
struct QTTArchitecture{N} <: Architecture where {N}
    var_nsites::NTuple{N,Int}
    var_map::Dict{Int,Int}
    site_map::Vector{Int}
    is_interleaved::Bool
end

"""
    QTTArchitecture(var_nsites::NTuple{N,Int}, var_map::Union{Dict{Int,Int},Nothing}=nothing; interleaved::Bool=false)

Build the QTT layout of `N` variables with `abs(var_nsites[i])` sites each, and compute its `site_map`.

# Arguments

  - `var_nsites::NTuple{N,Int}`: number of sites of each variable, negative to reverse its bit order.
  - `var_map::Union{Dict{Int,Int},Nothing}=nothing`: `variable => position` map, a permutation of `1:N`; `nothing`
    keeps the variables in their natural order.

# Keywords

  - `interleaved::Bool=false`: interleave the sites of the variables instead of grouping them in blocks.

# Throws

  - `ArgumentError`: if `var_map` is not a permutation of `1:N`.

# Examples

```jldoctest
julia> Tuple(QTTArchitecture((3, 2)).site_map)
(1, 1, 1, 2, 2)

julia> Tuple(QTTArchitecture((3, 2); interleaved=true).site_map)
(1, 2, 1, 2, 1)

julia> Tuple(QTTArchitecture((3, 2), Dict(1 => 2, 2 => 1)).site_map)
(2, 2, 1, 1, 1)
```
"""
function QTTArchitecture(
    var_nsites::NTuple{N,Int}, var_map::Union{Dict{Int,Int},Nothing}=nothing; interleaved::Bool=false
) where {N}
    if isnothing(var_map)
        var_map = Dict(i => i for i in 1:N)
    end

    length(var_map) == N || throw(ArgumentError("Variable map must have length $N"))
    Set(keys(var_map)) == Set(1:N) || throw(ArgumentError("Variable map keys must be 1:$N"))
    Set(values(var_map)) == Set(1:N) || throw(ArgumentError("Variable map values must be 1:$N"))

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
    TTArchitecture{N} <: Architecture
    TTArchitecture(var_dims::NTuple{N,Int}, var_map::Union{Dict{Int,Int},Nothing}=nothing)

Layout of an ``N``-dimensional grid of ``n_1 \\times \\dots \\times n_N`` points in the TT format, with one site per
variable. The site that carries variable ``i`` has dimension ``n_i``, so the sites have different dimensions whenever
the variables do. The constructor places variable ``i`` on site `var_map[i]`, or on site ``i`` if `var_map` is
`nothing`, and throws an `ArgumentError` if `var_map` is not a permutation of `1:N`.

# Fields

  - `var_dims::NTuple{N,Int}`: number of grid points of each variable.
  - `var_map::Dict{Int,Int}`: `variable => site` of each variable.
  - `site_map::Vector{Int}`: variable carried by each site, the inverse permutation of `var_map`.

# Examples

```jldoctest
julia> arch = TTArchitecture((5, 7, 3), Dict(1 => 3, 2 => 1, 3 => 2));

julia> Tuple(arch.site_map), site_dims(arch), variable_dims(arch)
((2, 3, 1), (7, 3, 5), (5, 7, 3))
```

See also [`QTTArchitecture`](@ref).
"""
struct TTArchitecture{N} <: Architecture
    var_dims::NTuple{N,Int}
    var_map::Dict{Int,Int}
    site_map::Vector{Int}

    function TTArchitecture(var_dims::NTuple{N,Int}, var_map::Union{Dict{Int,Int},Nothing}=nothing) where {N}
        if isnothing(var_map)
            var_map = Dict(i=>i for i in 1:N)
        else
            Set(keys(var_map)) == Set(1:N) || throw(ArgumentError("Variable map keys must be 1:$N"))
            Set(values(var_map)) == Set(1:N) || throw(ArgumentError("Variable map values must be 1:$N"))
        end

        site_map = zeros(Int, N)
        for (log_var, phys_pos) in var_map
            site_map[phys_pos] = log_var
        end

        return new{N}(var_dims, var_map, site_map)
    end
end

"""
    nsites_per_variable(arch::Architecture, var::Int) -> Int

Return the number of sites that carry variable `var`: `abs(arch.var_nsites[var])` for a [`QTTArchitecture`](@ref)
and 1 for a [`TTArchitecture`](@ref).
"""
nsites_per_variable(arch::QTTArchitecture, var::Int) = abs(arch.var_nsites[var])
nsites_per_variable(arch::TTArchitecture, var::Int) = 1

"""
    total_sites(arch::Architecture) -> Int

Return the number of sites ``N`` of the tensor trains built on `arch`: the total number of bits of a
[`QTTArchitecture`](@ref), and the number of variables of a [`TTArchitecture`](@ref).
"""
total_sites(arch::QTTArchitecture) = length(arch.site_map)
total_sites(arch::TTArchitecture{N}) where {N} = N

"""
    site_dim(arch::Architecture, site_idx::Int) -> Int

Return the dimension ``d`` of site `site_idx`: 2 for a [`QTTArchitecture`](@ref), and the number of grid points of
the variable it carries for a [`TTArchitecture`](@ref).
"""
site_dim(arch::QTTArchitecture, site_idx::Int) = 2
site_dim(arch::TTArchitecture, site_idx::Int) = arch.var_dims[arch.site_map[site_idx]]

"""
    site_dims(arch::QTTArchitecture) -> Vector{Int}
    site_dims(arch::TTArchitecture{N}) -> NTuple{N,Int}

Return the dimensions of all sites, in site order, i.e. [`site_dim`](@ref)`(arch, n)` for `n = 1:total_sites(arch)`.
These are the dimensions of `siteinds(arch)`.
"""
site_dims(arch::QTTArchitecture) = fill(2, total_sites(arch))
site_dims(arch::TTArchitecture{N}) where {N} = ntuple(n -> site_dim(arch, n), N)

"""
    variable_dims(arch::Architecture{N}) -> NTuple{N,Int}

Return the number of grid points of each variable, in variable order: ``2^{|n_i|}`` for a
[`QTTArchitecture`](@ref), and `var_dims` for a [`TTArchitecture`](@ref).
"""
variable_dims(arch::QTTArchitecture) = map(x -> 2^abs(x), arch.var_nsites)
variable_dims(arch::TTArchitecture) = arch.var_dims

"""
    site_indices(dims::AbstractVector{<:Integer}; kwargs...) -> Vector{Index{Int}}

Return one new site index of dimension `dims[n]` per site, tagged `"Site,n=\$n"`. The keywords are passed to
`ITensors.Index`. This is the owned replacement of the former `siteinds(::Vector{Int})` method.
"""
function site_indices(dims::AbstractVector{<:Integer}; kwargs...)
    return [Index(d, "Site,n=$n"; kwargs...) for (n, d) in enumerate(dims)]
end

"""
    siteinds(arch::Architecture; kwargs...) -> Vector{Index{Int}}

Return new site indices for `arch`, of dimensions [`site_dims`](@ref)`(arch)` and tagged as by
[`site_indices`](@ref), to which the keywords are passed.
"""
function ITensorMPS.siteinds(arch::Architecture; kwargs...)
    return site_indices(collect(Int, site_dims(arch)); kwargs...)
end

"""
    link_indices(dims::AbstractVector{<:Integer}; kwargs...) -> Vector{Index{Int}}

Return one new link (bond) index of dimension `dims[l]` per bond, tagged `"Link,l=\$l"`. The keywords are passed to
`ITensors.Index`. This is the owned replacement of the former `linkinds(::Vector{Int})` method.
"""
function link_indices(dims::AbstractVector{<:Integer}; kwargs...)
    return [Index(d, "Link,l=$(n)"; kwargs...) for (n, d) in enumerate(dims)]
end
