"""
    struct ScalarField{N,A<:AbstractArray,I<:Vector{<:Index},BCU<:BoundaryConditions{N},Arch<:Architecture}

Scalar field in ``N`` dimensions, stored as a time series of MPS on fixed site indices.

A field couples the states of one physical quantity (pressure, stream function, one velocity component) with everything
needed to build its operators: the architecture that maps grid nodes to sites, the site indices shared by every state,
and the boundary conditions of each axis. The time series `u` grows with `push!` as the evolution saves its checkpoints;
`field[k]` is the ``k``-th stored state, `last(field)` the latest one (or `nothing` if there is none), and
`length(field)` the number of stored states. Every stored state is moved to the array backend `A` of the field.

# Fields

  - `u::Vector{MPS}`: the stored states, oldest first.
  - `usites::I`: the site indices of every state, site 1 the most significant.
  - `arch::Arch`: the architecture (any `Architecture`, QTT or TT).
  - `dims::NTuple{N,Int}`: the number of grid nodes per axis, `variable_dims(arch)`.
  - `bcu::BCU`: the boundary conditions of each axis.
  - `array_type::Type{A}`: the array backend of the stored states (e.g. `Array`, `CuArray`).

See also [`VectorField`](@ref), [`BoundaryConditions`](@ref).
"""
struct ScalarField{N,A<:AbstractArray,I<:Vector{<:Index},BCU<:BoundaryConditions{N},Arch<:Architecture}
    u::Vector{MPS}
    usites::I
    arch::Arch
    dims::NTuple{N,Int}
    bcu::BCU
    array_type::Type{A}
end

"""
    backend_type(field::ScalarField) -> Type{<:AbstractArray}

Return the array backend of the states of `field`.
"""
backend_type(::ScalarField{N,A}) where {N,A} = A

"""
    ScalarField(init_u::MPS, arch::Architecture, bcu::BoundaryConditions{N},
                array_type::Union{Nothing,Type{<:AbstractArray}}=nothing) where N
    ScalarField(usites::Vector{<:Index}, arch::Architecture, bcu::BoundaryConditions{N},
                array_type::Type{<:AbstractArray}=Array) where N
    ScalarField(arch::Architecture, bcu::BoundaryConditions{N}, array_type::Type{<:AbstractArray}=Array) where N

Build a [`ScalarField`](@ref) on the architecture `arch` (QTT or TT) with the boundary conditions `bcu`.

The first method stores `init_u` as the first state and takes its site indices; with `array_type = nothing` the
backend is that of `init_u`, and otherwise `init_u` is moved to `array_type`. The other two methods build an empty
field (no stored state), on the given site indices `usites` or on new indices tagged `"Site,n=i"`.

# Arguments

  - `init_u::MPS`: initial state, with `total_sites(arch)` sites.
  - `usites::Vector{<:Index}`: site indices, with the dimensions `site_dims(arch)`.
  - `arch::Architecture`: the architecture of the field.
  - `bcu::BoundaryConditions{N}`: one pair of boundary conditions per axis.
  - `array_type`: the array backend of the states.

# Throws

  - `DimensionMismatch`: if `init_u` does not have `total_sites(arch)` sites, or if the dimensions of `usites` do not
    match `site_dims(arch)`.

# Examples

```julia
arch = QTTArchitecture((5, 4))
bcs = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), BoundaryCond1d(PeriodicBC(), PeriodicBC()))
p = ScalarField(siteinds(arch), arch, bcs)       # empty, to be filled by the evolution
u = ScalarField(ones_mps(p.usites, 1), arch, bcs) # one stored state
```
"""
function ScalarField(
    init_u::MPS,
    arch::Architecture,
    bcu::BoundaryConditions{N},
    array_type::Union{Nothing,Type{<:AbstractArray}}=nothing,
) where {N}
    dims = variable_dims(arch)
    total_site_count = total_sites(arch)
    length(init_u) == total_site_count ||
        throw(DimensionMismatch("Mismatch between init MPS length and arch: ($(length(init_u)) != $total_site_count)"))

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

function ScalarField(arch::Architecture, bcu::BoundaryConditions{N}, array_type::Type{<:AbstractArray}=Array) where {N}
    dims = variable_dims(arch)
    usites = [Index(site_dim(arch, i), "Site,n=$i") for i in 1:total_sites(arch)]

    u = MPS[]

    return ScalarField{N,array_type,typeof(usites),typeof(bcu),typeof(arch)}(u, usites, arch, dims, bcu, array_type)
end

function ScalarField(
    usites::Vector{<:Index}, arch::Architecture, bcu::BoundaryConditions{N}, array_type::Type{<:AbstractArray}=Array
) where {N}
    all(site_dims(arch) .== dim.(usites)) ||
        throw(DimensionMismatch("Site dimensions do not match with architecture properties"))
    dims = variable_dims(arch)
    u = MPS[]
    return ScalarField{N,array_type,typeof(usites),typeof(bcu),typeof(arch)}(u, usites, arch, dims, bcu, array_type)
end

"""
    last(field::ScalarField) -> Union{MPS,Nothing}

Return the latest stored state of `field`, or `nothing` if it stores none.
"""
function Base.last(field::ScalarField)
    return length(field.u) > 0 ? last(field.u) : nothing
end

"""
    getindex(field::ScalarField, index::Int) -> MPS

Return the stored state number `index` (oldest first).
"""
function Base.getindex(field::ScalarField, index::Int)
    return field.u[index]
end

"""
    setindex!(field::ScalarField, value::MPS, index::Int) -> ScalarField

Replace the stored state number `index` by `value`, moved to the backend of `field`. Throws an `ArgumentError` if the
site indices of `value` are not those of `field`.
"""
function Base.setindex!(field::ScalarField, value::MPS, index::Int)
    if siteinds(value) != field.usites
        throw(ArgumentError("MPS site indices do not match field site indices."))
    end
    field.u[index] = adapt(backend_type(field), value)
    return field
end

"""
    push!(field::ScalarField, new_u::MPS) -> ScalarField

Append `new_u`, moved to the backend of `field`, to its stored states. Throws an `ArgumentError` if the site indices of
`new_u` are not those of `field`.
"""
function Base.push!(field::ScalarField, new_u::MPS)
    if siteinds(new_u) != field.usites
        throw(ArgumentError("MPS site indices do not match field site indices."))
    end
    push!(field.u, adapt(backend_type(field), new_u))
    return field
end

"""
    length(field::ScalarField) -> Int

Return the number of stored states of `field`.
"""
function Base.length(field::ScalarField)
    return length(field.u)
end
