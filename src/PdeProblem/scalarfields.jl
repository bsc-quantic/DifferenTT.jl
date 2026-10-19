"""
    struct ScalarField{N,A<:AbstractArray,I<:Vector{<:Index},BCU<:BoundaryConditions{N},Arch<:Architecture{N}}

Represents a scalar field in N dimensions, stored as a time-series of MPSs.

# Fields

  - `u::Vector{MPS}`: Time series of the field states (each state is an MPS).
  - `usites::I`: The site indices used by the MPSs.
  - `arch::Arch`: The Architecture mapping.
  - `dims::NTuple{N,Int}`: The physical grid dimensions.
  - `bcu::BCU`: Boundary conditions for the field.
  - `array_type::Type{A}`: The backend array type.
"""
struct ScalarField{N,A<:AbstractArray,I<:Vector{<:Index},BCU<:BoundaryConditions{N},Arch<:Architecture}
    u::Vector{MPS}
    usites::I
    arch::Arch
    dims::NTuple{N,Int}
    bcu::BCU
    array_type::Type{A}
end

backend_type(::ScalarField{N,A}) where {N,A} = A

"""
    ScalarField(init_u::MPS, arch::Architecture, bcu::BoundaryConditions{N}, array_type::Type{<:AbstractArray}=Array) where N

Construct a ScalarField from an initial MPS state.

# Arguments

  - `init_u::MPS`: The initial state MPS.
  - `arch::QTTArchitecture{N}`: The QTTArchitecture configuration.
  - `bcu::BoundaryConditions{N}`: The boundary conditions.
  - `array_type`: The array backend type (default: `Array`).
"""
function ScalarField(
    init_u::MPS,
    arch::Architecture,
    bcu::BoundaryConditions{N},
    array_type::Union{Nothing,Type{<:AbstractArray}}=nothing,
) where {N}
    dims = variable_dims(arch)
    total_site_count = total_sites(arch)
    @assert length(init_u) == total_site_count "Mismatch between init MPS length and arch: ($(length(init_u)) != $total_site_count)"

    usites = siteinds(init_u)

    if isnothing(array_type)
        array_type = backend_type(init_u)
    else
        init_u = adapt(array_type, init_u)
    end

    u = MPS[]
    push!(u, init_u)

    return ScalarField{N,array_type,typeof(usites),typeof(bcu),typeof(arch)}(u, usites, arch, dims, bcu, array_type)
end

"""
    ScalarField(arch::QTTArchitecture{N}, bcu::BoundaryConditions{N}, array_type::Type{<:AbstractArray}=Array) where N

Construct an empty ScalarField (no time steps).

# Arguments

  - `arch::QTTArchitecture{N}`: The QTTArchitecture configuration.
  - `bcu::BoundaryConditions{N}`: The boundary conditions.
  - `array_type`: The array backend type.
"""
function ScalarField(arch::Architecture, bcu::BoundaryConditions{N}, array_type::Type{<:AbstractArray}=Array) where {N}
    dims = variable_dims(arch)
    usites = [Index(site_dim(arch, i), "Site,n=$i") for i in 1:total_sites(arch)]

    u = MPS[]

    return ScalarField{N,array_type,typeof(usites),typeof(bcu),typeof(arch)}(u, usites, arch, dims, bcu, array_type)
end

function ScalarField(
    usites::Vector{<:Index}, arch::Architecture, bcu::BoundaryConditions{N}, array_type::Type{<:AbstractArray}=Array
) where {N}
    @assert all(site_dims(arch) .== dim.(usites)) "Site dimensions do not match with architecture properties"
    dims = variable_dims(arch)
    u = MPS[]
    return ScalarField{N,array_type,typeof(usites),typeof(bcu),typeof(arch)}(u, usites, arch, dims, bcu, array_type)
end

function Base.last(field::ScalarField)
    return length(field.u) > 0 ? last(field.u) : nothing
end

function Base.getindex(field::ScalarField, index::Int)
    return field.u[index]
end

function Base.setindex!(field::ScalarField, value::MPS, index::Int)
    if siteinds(value) != field.usites
        throw(ArgumentError("MPS site indices do not match field site indices."))
    end
    field.u[index] = adapt(backend_type(field), value)
end

function Base.push!(field::ScalarField, new_u::MPS)
    if siteinds(new_u) != field.usites
        throw(ArgumentError("MPS site indices do not match field site indices."))
    end
    push!(field.u, adapt(backend_type(field), new_u))
end

function Base.length(field::ScalarField)
    return length(field.u)
end
