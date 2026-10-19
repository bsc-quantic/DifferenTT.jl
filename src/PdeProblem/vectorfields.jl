"""
    struct VectorField{N,A<:AbstractArray,I<:Vector{<:Index},S<:ScalarField{N},Arch<:Architecture}

Represents a vector field (collection of N ScalarFields) in N dimensions.
"""
struct VectorField{N,A<:AbstractArray,I<:Vector{<:Index},S<:ScalarField{N},Arch<:Architecture}
    components::NTuple{N,S}
    usites::I # All components share sites
    arch::Arch # All components share arch
    array_type::Type{A} # All components share array type
end

backend_type(::VectorField{N,A}) where {N,A} = A

function VectorField(components::Tuple{S,Vararg{S}}) where {S<:ScalarField}
    N = length(components)
    components[1] isa ScalarField{N} || throw(
        DimensionMismatch(
            "a VectorField of $N components needs ScalarField{$N} components, got $(typeof(components[1]))"
        ),
    )
    @assert all(length(field) == length(components[1]) for field in components) "All scalar filed must have the same number of steps"
    @assert all(backend_type(field) == backend_type(components[1]) for field in components) "All scalar fields must share array type"

    return VectorField(components, components[1].usites, components[1].arch, backend_type(components[1]))
end

"""
    VectorField(init_u::NTuple{N,MPS}, arch::Architecture, bcs::NTuple{N,BoundaryConditions{N}}, array_type::Union{Nothing, Type{<:AbstractArray}}=nothing) where N

Construct a VectorField from a tuple of initial MPSs.

# Arguments

  - `init_u::NTuple{N,MPS}`: Tuple of initial state MPSs (one per component).
  - `arch::Architecture`: The Architecture configuration.
  - `bcs::NTuple{N,BoundaryConditions{N}}`: Tuple of boundary conditions (one per component).
  - `array_type`: The array backend type.
"""
function VectorField(init_u::NTuple{N,MPS}, arch::Architecture, bcs::NTuple{N,BoundaryConditions{N}}) where {N}
    total_site_count = total_sites(arch)
    for i in 1:N
        @assert length(init_u[i]) == total_site_count "Mismatch between init MPS length and arch for component $i"
    end
    @assert all(siteinds(init_u[1]) == siteinds(u) for u in init_u) "MPS site indices do not match across components."
    @assert all(backend_type(init_u[1]) == backend_type(u) for u in init_u) "MPS backend types do not match across components."

    array_type = backend_type(init_u[1])
    usites = siteinds(init_u[1])

    components = ntuple(i -> ScalarField(init_u[i], arch, bcs[i], array_type), N)

    return VectorField(components, usites, arch, backend_type(init_u[1]))
end

"""
    VectorField(arch::Architecture, bcs::NTuple{N,BoundaryConditions{N}}, array_type::Type{<:AbstractArray}=Array) where N

Construct an empty VectorField.

# Arguments

  - `arch::Architecture`: The Architecture configuration.
  - `bcs::NTuple{N,BoundaryConditions{N}}`: Tuple of boundary conditions.
  - `array_type`: The array backend type.
"""
function VectorField(
    arch::Architecture, bcs::NTuple{N,BoundaryConditions{N}}, array_type::Type{<:AbstractArray}=Array
) where {N}
    usites = [Index(site_dim(arch, i), "Site,n=$i") for i in 1:total_sites(arch)]
    components = ntuple(i -> ScalarField(usites, arch, bcs[i], array_type), N)
    return VectorField(components, usites, arch, array_type)
end

function VectorField(
    usites::Vector{<:Index},
    arch::Architecture,
    bcs::NTuple{N,BoundaryConditions{N}},
    array_type::Type{<:AbstractArray}=Array,
) where {N}
    @assert all(site_dims(arch) .== dim.(usites)) "Site dimensions do not match with architecture properties"
    components = ntuple(i -> ScalarField(usites, arch, bcs[i], array_type), N)
    return VectorField(components, usites, arch, array_type)
end

function Base.last(fields::VectorField{N}) where {N}
    return map(c -> last(c), fields.components)
end

function Base.push!(fields::VectorField{N}, new_u::NTuple{N,MPS}) where {N}
    for i in 1:N
        push!(fields.components[i], new_u[i])
    end
end

function Base.setindex!(fields::VectorField{N}, value::NTuple{N,MPS}, index::Int) where {N}
    for i in 1:N
        fields.components[i][index] = value[i]
    end
end

function Base.getindex(fields::VectorField, i::Int)
    return fields.components[i]
end

function Base.length(fields::VectorField)
    return length(fields.components[1])
end
