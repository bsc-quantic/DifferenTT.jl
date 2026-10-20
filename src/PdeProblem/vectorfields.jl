"""
    struct VectorField{N,A<:AbstractArray,I<:Vector{<:Index},S<:ScalarField{N},Arch<:Architecture}

Vector field in ``N`` dimensions: ``N`` [`ScalarField`](@ref) components that share their site indices, architecture
and array backend, and store the same number of states, each with its own boundary conditions.

The time series of the components grow together: `push!` appends one state
per component, `last(fields)` returns the tuple of the latest states, and `length(fields)` the number of stored states.
Note that indexing differs from a `ScalarField`: `fields[i]` returns the **component** `i`, not a state, so the state
``k`` of component ``i`` is `fields[i][k]`, while `fields[k] = states` replaces state ``k`` of every component.

# Fields

  - `components::NTuple{N,S}`: the components, one per axis.
  - `usites::I`: the site indices shared by all components.
  - `arch::Arch`: the architecture shared by all components.
  - `array_type::Type{A}`: the array backend shared by all components.

See also [`ScalarField`](@ref).
"""
struct VectorField{N,A<:AbstractArray,I<:Vector{<:Index},S<:ScalarField{N},Arch<:Architecture}
    components::NTuple{N,S}
    usites::I # All components share sites
    arch::Arch # All components share arch
    array_type::Type{A} # All components share array type
end

"""
    backend_type(fields::VectorField) -> Type{<:AbstractArray}

Return the array backend shared by the components of `fields`.
"""
backend_type(::VectorField{N,A}) where {N,A} = A

"""
    VectorField(components::Tuple{S,Vararg{S}}) where {S<:ScalarField}
    VectorField(init_u::NTuple{N,MPS}, arch::Architecture, bcs::NTuple{N,BoundaryConditions{N}}) where N
    VectorField(usites::Vector{<:Index}, arch::Architecture, bcs::NTuple{N,BoundaryConditions{N}},
                array_type::Type{<:AbstractArray}=Array) where N
    VectorField(arch::Architecture, bcs::NTuple{N,BoundaryConditions{N}},
                array_type::Type{<:AbstractArray}=Array) where N

Build a [`VectorField`](@ref) on the architecture `arch` (QTT or TT), with the boundary conditions `bcs[i]` for
component `i`.

The first method groups existing components. The second stores the states `init_u` (one per component) as the first
state, on their common site indices and backend. The last two build an empty field, on the given site indices `usites`
or on new indices tagged `"Site,n=i"`.

# Throws

  - `DimensionMismatch`: if the components are not `ScalarField{N}` for `N` components, if a state of `init_u` does not
    have `total_sites(arch)` sites, or if the dimensions of `usites` do not match `site_dims(arch)`.
  - `ArgumentError`: if the components (or the states of `init_u`) differ in their number of stored states, site
    indices or array backend.

# Examples

```julia
arch = QTTArchitecture((5, 4))
wall = BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0))
bcs = BoundaryConditions(wall, BoundaryCond1d(PeriodicBC(), PeriodicBC()))
sites = siteinds(arch)
u = VectorField((ones_mps(sites, 1), zeros_mps(sites)), arch, (bcs, bcs))
```
"""
function VectorField(components::Tuple{S,Vararg{S}}) where {S<:ScalarField}
    N = length(components)
    components[1] isa ScalarField{N} || throw(
        DimensionMismatch(
            "a VectorField of $N components needs ScalarField{$N} components, got $(typeof(components[1]))"
        ),
    )
    all(length(field) == length(components[1]) for field in components) ||
        throw(ArgumentError("All scalar fields must have the same number of steps"))
    all(backend_type(field) == backend_type(components[1]) for field in components) ||
        throw(ArgumentError("All scalar fields must share array type"))

    return VectorField(components, components[1].usites, components[1].arch, backend_type(components[1]))
end

function VectorField(init_u::NTuple{N,MPS}, arch::Architecture, bcs::NTuple{N,BoundaryConditions{N}}) where {N}
    total_site_count = total_sites(arch)
    for i in 1:N
        length(init_u[i]) == total_site_count ||
            throw(DimensionMismatch("Mismatch between init MPS length and arch for component $i"))
    end
    all(siteinds(init_u[1]) == siteinds(u) for u in init_u) ||
        throw(ArgumentError("MPS site indices do not match across components."))
    all(backend_type(init_u[1]) == backend_type(u) for u in init_u) ||
        throw(ArgumentError("MPS backend types do not match across components."))

    array_type = backend_type(init_u[1])
    usites = siteinds(init_u[1])

    components = ntuple(i -> ScalarField(init_u[i], arch, bcs[i], array_type), N)

    return VectorField(components, usites, arch, backend_type(init_u[1]))
end

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
    all(site_dims(arch) .== dim.(usites)) ||
        throw(DimensionMismatch("Site dimensions do not match with architecture properties"))
    components = ntuple(i -> ScalarField(usites, arch, bcs[i], array_type), N)
    return VectorField(components, usites, arch, array_type)
end

"""
    last(fields::VectorField{N}) where N -> NTuple{N,Union{MPS,Nothing}}

Return the latest stored state of every component (`nothing` for a component without states).
"""
function Base.last(fields::VectorField{N}) where {N}
    return map(c -> last(c), fields.components)
end

"""
    _check_vectorfield_sites(fields::VectorField{N}, states::NTuple{N,MPS}) where N

Throw an `ArgumentError` if the site indices of any of `states` are not those of `fields`. `push!` and `setindex!`
call it before mutating any component, so that a mismatch cannot leave the components with different numbers of
stored states.
"""
function _check_vectorfield_sites(fields::VectorField{N}, states::NTuple{N,MPS}) where {N}
    all(siteinds(state) == fields.usites for state in states) ||
        throw(ArgumentError("MPS site indices do not match field site indices."))
    return nothing
end

"""
    push!(fields::VectorField{N}, new_u::NTuple{N,MPS}) where N -> VectorField

Append `new_u[i]` to the stored states of component `i`, for every component. Throws an `ArgumentError`, before
modifying any component, if the site indices of a state are not those of `fields`.
"""
function Base.push!(fields::VectorField{N}, new_u::NTuple{N,MPS}) where {N}
    _check_vectorfield_sites(fields, new_u)
    for i in 1:N
        push!(fields.components[i], new_u[i])
    end
    return fields
end

"""
    setindex!(fields::VectorField{N}, value::NTuple{N,MPS}, index::Int) where N -> VectorField

Replace the stored state number `index` of component `i` by `value[i]`, for every component. Throws an
`ArgumentError`, before modifying any component, if the site indices of a state are not those of `fields`.
"""
function Base.setindex!(fields::VectorField{N}, value::NTuple{N,MPS}, index::Int) where {N}
    _check_vectorfield_sites(fields, value)
    for i in 1:N
        fields.components[i][index] = value[i]
    end
    return fields
end

"""
    getindex(fields::VectorField, i::Int) -> ScalarField

Return the component `i` of `fields` (not the state `i`; see [`VectorField`](@ref)).
"""
function Base.getindex(fields::VectorField, i::Int)
    return fields.components[i]
end

"""
    length(fields::VectorField) -> Int

Return the number of stored states, which every component shares.
"""
function Base.length(fields::VectorField)
    return length(fields.components[1])
end
