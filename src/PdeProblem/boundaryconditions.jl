"""
    abstract type BoundaryCond

Abstract type for all boundary condition types.``
"""
abstract type BoundaryCond end

"""
    struct PeriodicBC <: BoundaryCond

Periodic boundary conditions.
"""
struct PeriodicBC <: BoundaryCond
    value::Nothing
    function PeriodicBC()
        return new(nothing)
    end
end

"""
    struct OpenBC <: BoundaryCond

Open/free boundary conditions.
"""
struct OpenBC <: BoundaryCond
    value::Nothing
    function OpenBC()
        return new(nothing)
    end
end

"""
    struct DirichletBC{T <: Number} <: BoundaryCond

Dirichlet boundary condition with a specific value.

Currently only homogeneous Dirichlet conditions are supported.

Note that the explicit time steppers do not impose `value`: they keep the boundary rows of the field at the values of
the initial field, so the boundary values are set by the initial condition, not by this object.

# Fields

  - `value::T`: The value of the field at the boundary.
"""
struct DirichletBC{T<:Number} <: BoundaryCond
    value::T
end

"""
    struct NeumannBC{T <: Number} <: BoundaryCond

Neumann boundary condition with a specific derivative value.

Currently only homogeneous Neumann conditions equal to 0 are supported.

# Fields

  - `value::T`: The value of the normal derivative at the boundary.
"""
struct NeumannBC{T<:Number} <: BoundaryCond
    value::T
    function NeumannBC(value::T) where {T<:Number}
        !(value≈0.0) && throw(ArgumentError("NeumannBC with value != 0.0 are not yet implemented."))
        return new{T}(value)
    end
end

"""
    struct BoundaryCond1d{BCL <: BoundaryCond, BCR <: BoundaryCond}

Pair of boundary conditions (left and right) for a 1D domain.

# Fields

  - `left::BCL`: Boundary condition at the left end.
  - `right::BCR`: Boundary condition at the right end.
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
    struct BoundaryConditions{N, BC <: BoundaryCond1d}

Container for boundary conditions for N dimensions.

# Fields

  - `conditions::NTuple{N,BC}`: Tuple of 1D boundary conditions for each dimension.
"""
struct BoundaryConditions{N,BC<:BoundaryCond1d}
    conditions::NTuple{N,BC}
end

function BoundaryConditions(conditions::BoundaryCond1d...)
    return BoundaryConditions{length(conditions),eltype(conditions)}(conditions)
end

function Base.getindex(bcs::BoundaryConditions, i::Int)
    return bcs.conditions[i]
end
