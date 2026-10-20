"""
    abstract type BoundaryCond

Supertype of the boundary conditions of one side of one axis.

Every closure in DifferenTT is the interior central stencil of order ``q``, completed by values at nodes that are not
stored (ghost nodes). A boundary condition is a rule that gives those ghost values: when the rule involves the
unknowns, substituting it changes the operator (periodic wrapping, the Neumann mirror), and when it involves only data,
it adds a known vector to the right-hand side (the Dirichlet lifts). The subtypes are [`PeriodicBC`](@ref),
[`OpenBC`](@ref), [`DirichletBC`](@ref) and [`NeumannBC`](@ref). Each holds a field `value`, which
[`BoundaryCond1d`](@ref) returns through `getindex`.

See also [`BoundaryCond1d`](@ref), [`BoundaryConditions`](@ref), and the boundary-condition tutorials
[Velocity–pressure formulation](@ref bc_velocity_pressure) and
[Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction).
"""
abstract type BoundaryCond end

"""
    PeriodicBC()

Periodic boundary condition: the ghost node beyond one end of the axis is the image of the stored node at the other
end, so the stencil wraps around and the one-dimensional operator is a circulant. No correction of the operator or of
the right-hand side is needed. It must be set on both sides of an axis, which [`BoundaryCond1d`](@ref) checks; its
`value` field is `nothing`.

Note that the period of the wrapped stencil is ``n h``, so the grid must not store the node at ``L``: build it with
`Grid(arch, box_size; periodic=periodic_axes(bcs))`, which gives ``h = L/n``. Without the flag, [`Grid`](@ref) places
nodes at both ends, ``h = L/(n - 1)``, and the period becomes ``L + h``.

# Examples

```jldoctest
julia> BoundaryCond1d(PeriodicBC(), PeriodicBC())[1] === nothing
true
```

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
struct PeriodicBC <: BoundaryCond
    value::Nothing
    function PeriodicBC()
        return new(nothing)
    end
end

"""
    OpenBC()

Open boundary condition: the stencil entries that fall outside the grid are dropped, i.e. the ghost values are taken as
zero. This is the same truncation that the operators apply on a [`DirichletBC`](@ref) side, but nothing else acts on
an `OpenBC` side: its boundary rows are neither held fixed by the time steppers ([`BoundaryNullScalar`](@ref)) nor
completed by a lift on the right-hand side. Its `value` field is `nothing`.

See also the boundary-condition tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
struct OpenBC <: BoundaryCond
    value::Nothing
    function OpenBC()
        return new(nothing)
    end
end

"""
    DirichletBC(value::Number)

Dirichlet boundary condition with boundary value `value`.

The differential operators truncate the stencil at a Dirichlet side, i.e. they take the ``q/2`` ghost nodes beyond the
last stored node as zero, and the value then enters in a different way depending on where the condition is used:

  - In the pressure equation of the Chorin projection, [`scalar_dirichlet_correction`](@ref) moves the ghost values
    ``g`` = `value` to the right-hand side: it restores the dropped stencil coefficients applied to ghost nodes that all
    hold ``g``. The stored boundary node is therefore an unknown, and the value sits on the ghost nodes beyond it.
  - In the explicit time steppers, the value is **not** imposed. [`BoundaryNullScalar`](@ref) zeroes the right-hand
    side on the ``q/2`` rows next to a Dirichlet side, so those rows keep the values of the initial field, and the
    boundary values are set by the initial condition, not by this object.
  - In the vorticity–streamfunction formulation, the value is not read: the [`VorticityFace`](@ref) of each side holds
    the boundary data, and [`vorticity_boundary_conditions`](@ref) only uses `DirichletBC` to mark ghost faces.

Note that the Dirichlet value therefore sits on the first ghost node, one spacing beyond the last stored node, and that
the lift fills every ghost node with it. This is exact for ``q = 2``, but for ``q = 8`` and a solution with a nonzero
slope at the boundary it makes the closure, and the solution, only first-order accurate (a known limitation of v0.1;
the higher-order closure is planned for v0.2).

# Fields

  - `value::T`: the boundary value, of any `Number` type `T`.

# Examples

```jldoctest
julia> BoundaryCond1d(DirichletBC(1.5), DirichletBC(0))[1]
1.5
```

See also [`NeumannBC`](@ref), and the boundary-condition tutorial
[Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
struct DirichletBC{T<:Number} <: BoundaryCond
    value::T
end

"""
    NeumannBC(value::Number)

Homogeneous Neumann boundary condition, ``\\partial u / \\partial n = 0``.

The boundary lies on the stored node ``u_0``, and the ghost values are its mirror images, ``u_{-k} = u_k``.
Eliminating them adds the mirrored coefficients to the ``q/2`` boundary rows of every one-dimensional operator
(`neumann_correction_matrix`). This makes the first-derivative row of the boundary node zero, and it counts the
boundary row of the second derivative twice, so the operator is no longer symmetric; the weight of
[`neumann_weight_mpo`](@ref) restores the symmetry of the linear systems. Only `value ≈ 0` is supported.

# Fields

  - `value::T`: the normal derivative at the boundary (zero).

# Throws

  - `ArgumentError`: if `value` is not approximately zero.

# Examples

```jldoctest
julia> NeumannBC(0.0).value
0.0

julia> NeumannBC(1.0)
ERROR: ArgumentError: NeumannBC with value != 0.0 are not yet implemented.
```

See also [`DirichletBC`](@ref), and the boundary-condition tutorial
[Velocity–pressure formulation](@ref bc_velocity_pressure).
"""
struct NeumannBC{T<:Number} <: BoundaryCond
    value::T
    function NeumannBC(value::T) where {T<:Number}
        !(value≈0.0) && throw(ArgumentError("NeumannBC with value != 0.0 are not yet implemented."))
        return new{T}(value)
    end
end

"""
    BoundaryCond1d(left::BoundaryCond, right::BoundaryCond)

Pair of boundary conditions of one axis: `left` at the first node (coordinate 0) and `right` at the last node.

# Fields

  - `left::BCL`: boundary condition at the left end.
  - `right::BCR`: boundary condition at the right end.

# Throws

  - `ArgumentError`: if only one of the two sides is a [`PeriodicBC`](@ref).

# Examples

```jldoctest
julia> bc = BoundaryCond1d(NeumannBC(0.0), DirichletBC(2.0));

julia> bc.left isa NeumannBC, bc[2]
(true, 2.0)
```

See also [`BoundaryConditions`](@ref).
"""
struct BoundaryCond1d{BCL<:BoundaryCond,BCR<:BoundaryCond}
    left::BCL
    right::BCR

    function BoundaryCond1d(left::BCL, right::BCR) where {BCL<:BoundaryCond,BCR<:BoundaryCond}
        if (left isa PeriodicBC) != (right isa PeriodicBC)
            throw(ArgumentError("Periodic boundary conditions must be applied on both sides or neither."))
        end
        new{BCL,BCR}(left, right)
    end
end

"""
    getindex(bcs::BoundaryCond1d, key::Int)

Return the `value` of the left (`key = 1`) or right (`key = 2`) boundary condition: the boundary value of a
[`DirichletBC`](@ref) or [`NeumannBC`](@ref), and `nothing` for [`PeriodicBC`](@ref) and [`OpenBC`](@ref). Throws an
`ArgumentError` for any other `key`.
"""
function Base.getindex(bcs::BoundaryCond1d, key::Int)
    if key == 1
        return bcs.left.value
    elseif key == 2
        return bcs.right.value
    else
        throw(ArgumentError("Only valid keys are 1 (left) and 2 (right)."))
    end
end

"""
    BoundaryConditions(conditions::BoundaryCond1d...)

Boundary conditions of a field in ``N`` dimensions: one [`BoundaryCond1d`](@ref) per axis, in the order of the
variables of the architecture. Indexing with `bcs[i]` returns the conditions of axis `i`.

# Fields

  - `conditions::NTuple{N,BC}`: the conditions of each axis.

# Examples

```jldoctest
julia> x_bcs, y_bcs = BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), BoundaryCond1d(PeriodicBC(), PeriodicBC());

julia> bcs = BoundaryConditions(x_bcs, y_bcs);

julia> bcs[2].left isa PeriodicBC
true
```

See also [`ScalarField`](@ref).
"""
struct BoundaryConditions{N,BC<:BoundaryCond1d}
    conditions::NTuple{N,BC}
end

function BoundaryConditions(conditions::BoundaryCond1d...)
    return BoundaryConditions{length(conditions),eltype(conditions)}(conditions)
end

"""
    periodic_axes(bcs::BoundaryConditions{N}) where N -> NTuple{N,Bool}

Return whether each axis of `bcs` is periodic, the flags that [`Grid`](@ref) takes as `periodic`, so that the grid
is built from the boundary conditions without depending on their types.

# Examples

```jldoctest
julia> wall, periodic = BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), BoundaryCond1d(PeriodicBC(), PeriodicBC());

julia> DifferenTT.periodic_axes(BoundaryConditions(wall, periodic))
(false, true)
```
"""
periodic_axes(bcs::BoundaryConditions{N}) where {N} = ntuple(i -> bcs[i].left isa PeriodicBC, N)

"""
    getindex(bcs::BoundaryConditions, i::Int) -> BoundaryCond1d

Return the boundary conditions of axis `i`.
"""
function Base.getindex(bcs::BoundaryConditions, i::Int)
    return bcs.conditions[i]
end
