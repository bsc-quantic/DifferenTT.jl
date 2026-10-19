"""
    neumann_weight_mpo(field::ScalarField{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) where N

Diagonal weight `W = W₁ ⊗ … ⊗ W_N` that symmetrizes a linear system built with the mirrored `NeumannBC`
closures, whose boundary lies on the stored node. `Wᵢ` is the identity with `1/2` on the Neumann boundary
rows of axis `i`. Multiplying the whole system `A x = b` by `W` gives the finite volume half-cell equations:
`W A` is symmetric and has the same solution, and a node shared by two Neumann faces gets weight `1/4`.
Returns `nothing` if `field` has no `NeumannBC` face.

Each `Wᵢ` is the identity plus [`onedim_neumann_symmetrization_correction`](@ref) applied to it. In QTT, `Wᵢ`
has bond dimension 2 (one Neumann side) or 3 (both sides) inside its axis. The lifted `W` has bond dimension
1 between axes for TT and sequential QTT, while interleaved QTT threads the bonds of every axis through the
others.
"""
function neumann_weight_mpo(field::ScalarField{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) where {N}
    array_type = backend_type(field)
    weights = Union{MPO,Nothing}[]

    for i in 1:N
        bcs = field.bcu[i]
        if !(bcs.left isa NeumannBC || bcs.right isa NeumannBC)
            push!(weights, nothing)
            continue
        end

        if field.arch isa QTTArchitecture
            identity_op = identity_mpo(siteinds(2, abs(field.arch.var_nsites[i])), array_type)
        else
            identity_op = adapt(array_type, identity_array(field.arch.var_dims[i]))
        end
        correction = onedim_neumann_symmetrization_correction(
            identity_op, field.arch, i, bcs; cutoff=cutoff, maxdim=maxdim
        )
        onedim_weight = if identity_op isa MPO
            _nothing_sum_handler(identity_op, correction; cutoff=cutoff, maxdim=maxdim)
        else
            identity_op + correction
        end

        push!(weights, ndim_operator_mpo(onedim_weight, i, field.usites, field.arch))
    end

    return _nothing_apply_handler(weights...; cutoff=cutoff, maxdim=maxdim)
end

"""
    symmetrize_rhs(field::ScalarField{N}, rhs::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) where N

Weight an arbitrary right-hand side `MPS` with the Neumann weight `W` of `field` (see
[`neumann_weight_mpo`](@ref)), i.e. halve it on the `NeumannBC` boundary rows of each axis (a quarter on
nodes shared by two Neumann faces). Meant to be used alongside `symmetrized_laplacian`, which applies the
same `W` to the operator. Returns `rhs` itself if `field` has no `NeumannBC` face.

# Returns

  - `MPS`: The symmetrized right-hand side.
"""
function symmetrize_rhs(field::ScalarField{N}, rhs::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int)) where {N}
    weight = neumann_weight_mpo(field; cutoff=cutoff, maxdim=maxdim)
    return isnothing(weight) ? rhs : apply(weight, rhs; cutoff=cutoff, maxdim=maxdim)
end

"""
    face_profile_mps(field::ScalarField{2}, axis::Int, values; cutoff::Float64=1e-15, maxdim::Int=typemax(Int))

MPS on the sites of `field` that is constant along `axis` and equal to `values` along the other axis, i.e. the
profile of a face of `axis` extended to the whole domain. `values` is either a `Number` (constant profile) or a
vector with one value per stored node of the other axis, in increasing node index.
"""
function face_profile_mps(field::ScalarField{2}, axis::Int, values; cutoff::Float64=1e-15, maxdim::Int=typemax(Int))
    array_type = backend_type(field)
    values isa Number && return values * ones_mps(field.usites, 1, array_type)

    other = 3 - axis
    length(values) == variable_dims(field.arch)[other] || throw(
        ArgumentError(
            "Face profile of axis $axis must have $(variable_dims(field.arch)[other]) values, got $(length(values))"
        ),
    )
    onedim = _onedim_diagonal_operator(field.arch, other, values, array_type; cutoff=cutoff)
    profile_op = ndim_operator_mpo(onedim, other, field.usites, field.arch)
    return apply(profile_op, ones_mps(field.usites, 1, array_type); cutoff=cutoff, maxdim=maxdim)
end

function _onedim_diagonal_operator(arch::TTArchitecture, axis::Int, values, array_type; cutoff::Float64=1e-15)
    adapt(array_type, Matrix(Diagonal(Vector{Float64}(values))))
end

# TODO A diagonal built from vec2mps would avoid the dense 2ⁿ × 2ⁿ matrix (fine up to n ≈ 11)
function _onedim_diagonal_operator(arch::QTTArchitecture, axis::Int, values, array_type; cutoff::Float64=1e-15)
    sites = siteinds(2, abs(arch.var_nsites[axis]))
    return adapt(array_type, mat2mpo(Matrix(Diagonal(Vector{Float64}(values))), sites; cutoff=cutoff))
end

"""
    face_dirichlet_lift(field::ScalarField{2}, grid::Grid{2}, derivative_order::Int, axis::Int, side::Symbol, values; stencil_order::Int=2, cutoff::Float64=1e-15, maxdim::Int=typemax(Int))

Ghost contribution of a single face to a derivative of order `derivative_order` along `axis`: the derivative
operators truncate the stencils at non-periodic faces (ghost value 0), and adding this MPS restores the ghost
value `values` (a `Number` or a profile along the face, see [`face_profile_mps`](@ref)) on the `side` (`:left` or
`:right`) face of `axis`. With `stencil_order = 2`, it is only nonzero on the row next to the face.

Returns `nothing` if `values` is `nothing` or zero.
"""
function face_dirichlet_lift(
    field::ScalarField{2},
    grid::Grid{2},
    derivative_order::Int,
    axis::Int,
    side::Symbol,
    values;
    stencil_order::Int=2,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int),
)
    (isnothing(values) || all(iszero, values)) && return nothing
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right, got $side"))
    array_type = backend_type(field)
    onedim = if side == :left
        onedim_left_dirichlet_correction(derivative_order, field.arch, axis, array_type; stencil_order=stencil_order)
    else
        onedim_right_dirichlet_correction(derivative_order, field.arch, axis, array_type; stencil_order=stencil_order)
    end
    onedim =
        onedim isa MPO ? onedim * (1 / grid.dx[axis]^derivative_order) : onedim .* (1 / grid.dx[axis]^derivative_order)
    correction = ndim_operator_mpo(onedim, axis, field.usites, field.arch)
    return apply(
        correction, face_profile_mps(field, axis, values; cutoff=cutoff, maxdim=maxdim); cutoff=cutoff, maxdim=maxdim
    )
end

"""
    remove_mean(rhs::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int))

Subtract the node mean of `rhs` (plain sum over the nodes divided by their number), so that a right-hand side of a
singular system with constant null space becomes consistent.
"""
function remove_mean(rhs::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int))
    sites = siteinds(rhs)
    mean = integrate(rhs) / prod(dim, sites)
    return +(rhs, -mean * ones_mps(sites, 1, backend_type(rhs)); cutoff=cutoff, maxdim=maxdim)
end

"""
    divergence_dirichlet_correction(u_fields::VectorField{N}; stencil_order::Int = 8) where N

Compute the boundary correction MPO for the divergence operator with Dirichlet BCs on velocity.

# Arguments

  - `u_fields::VectorField{N}`: The velocity field (containing BC info).
  - `stencil_order::Int`: Stencil order.
"""
function divergence_dirichlet_correction(
    u_fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    array_type = u_fields.components[1].array_type

    deleters = Union{MPO,Nothing}[]

    for i in 1:N
        comp = u_fields.components[i]
        bc_i = comp.bcu[i]

        onedim_deleter = onedim_boundary_nullifier(
            u_fields.arch, i, bc_i, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        )

        deleter =
            (isnothing(onedim_deleter)) ? nothing : ndim_operator_mpo(onedim_deleter, i, u_fields.usites, u_fields.arch)
        push!(deleters, deleter)
    end

    return _nothing_apply_handler(deleters...; cutoff=cutoff, maxdim=maxdim)
end

"""
    scalar_dirichlet_correction_dim(field::ScalarField{N}, grid::Grid{N}, derivative_order::Int, dim::Int; stencil_order, cutoff, maxdim) where N

Compute the Dirichlet boundary correction MPO for a scalar field along a single dimension `dim`.
Returns `nothing` if no Dirichlet BCs are present or if the boundary values are zero.
"""
function scalar_dirichlet_correction_dim(
    field::ScalarField{N},
    grid::Grid{N},
    derivative_order::Int,
    dim::Int;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
) where {N}
    bc_dim = field.bcu[dim]

    left_correction, right_correction = nothing, nothing

    if bc_dim.left isa DirichletBC
        val_L = bc_dim.left.value
        if abs(val_L) > 1e-14
            onedim_left_correction = onedim_left_dirichlet_correction(
                derivative_order, field.arch, dim, field.array_type; stencil_order=stencil_order
            )
            left_correction = ndim_operator_mpo(onedim_left_correction, dim, field.usites, field.arch)
            left_correction = left_correction * val_L
        end
    end

    if bc_dim.right isa DirichletBC
        val_R = bc_dim.right.value
        if abs(val_R) > 1e-14
            onedim_right_correction = onedim_right_dirichlet_correction(
                derivative_order, field.arch, dim, field.array_type; stencil_order=stencil_order
            )
            right_correction = ndim_operator_mpo(onedim_right_correction, dim, field.usites, field.arch)
            right_correction = right_correction * val_R
        end
    end

    dim_correction = _nothing_sum_handler(left_correction, right_correction; cutoff=cutoff, maxdim=maxdim)
    if !isnothing(dim_correction)
        if dim_correction isa MPO
            dim_correction = dim_correction * (1 / grid.dx[dim]^derivative_order)
        else
            dim_correction = dim_correction .* (1 / grid.dx[dim]^derivative_order)
        end
    end

    return dim_correction
end

"""
    scalar_dirichlet_correction(field::ScalarField{N}, grid::Grid{N}, derivative_order::Int; stencil_order, cutoff, maxdim) where N

Compute the total Dirichlet boundary correction MPO for a scalar field across all dimensions.
Returns `nothing` if no corrections are needed.
"""
function scalar_dirichlet_correction(
    field::ScalarField{N},
    grid::Grid{N},
    derivative_order::Int;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
) where {N}
    corrections = Union{MPO,Nothing}[]

    for i in 1:N
        dim_correction = scalar_dirichlet_correction_dim(
            field, grid, derivative_order, i; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        )
        push!(corrections, dim_correction)
    end

    return _nothing_sum_handler(corrections...; cutoff=cutoff, maxdim=maxdim)
end
