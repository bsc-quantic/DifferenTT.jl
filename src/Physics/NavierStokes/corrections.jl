# Special treatment for the BCs of the divergence of a field -> The boundary nullifiers act ONLY on the normal components, 
# and the dirichlet correction from the pressure field is moved to the right hand side too
# TODO Understand if these methods can be recycled for other experiments too
"""
    divergence_dirichlet_correction(u_fields::VectorField{N}; stencil_order::Int = 8) where N

Compute the boundary correction MPO for the divergence operator with Dirichlet BCs on velocity.

# Arguments
- `u_fields::VectorField{N}`: The velocity field (containing BC info).
- `stencil_order::Int`: Stencil order.
"""
function divergence_dirichlet_correction(u_fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    array_type = u_fields.components[1].array_type

    deleters = Union{MPO,Nothing}[]

    for i in 1:N
        comp = u_fields.components[i]
        bc_i = comp.bcu[i]

        onedim_deleter = onedim_boundary_nullifier(u_fields.arch, i, bc_i, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)

        deleter = (isnothing(onedim_deleter)) ? nothing : ndim_operator_mpo(onedim_deleter, i, u_fields.usites, u_fields.arch)
        push!(deleters, deleter)
    end

    return _nothing_apply_handler(deleters...; cutoff=cutoff, maxdim=maxdim)
end

"""
    pressure_dirichlet_correction_dim(p_field::ScalarField{N}, grid::Grid{N}, derivative_order::Int, dim::Int; stencil_order, cutoff, maxdim) where N

Compute the Dirichlet boundary correction MPO for the pressure field along a single dimension `dim`.
Returns `nothing` if no Dirichlet BCs are present or if the boundary values are zero.
"""
function pressure_dirichlet_correction_dim(p_field::ScalarField{N}, grid::Grid{N}, derivative_order::Int, dim::Int; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    bc_dim = p_field.bcu[dim]

    left_correction, right_correction = nothing, nothing

    if bc_dim.left isa DirichletBC
        # Get the boundary value
        val_L = bc_dim.left.value

        # If value is non-zero, take the corner mpo that contains the coefficients and scale it by the boundary value
        if abs(val_L) > 1e-14
            onedim_left_correction = onedim_left_dirichlet_correction(derivative_order, p_field.arch, dim, p_field.array_type; stencil_order=stencil_order)
            left_correction = ndim_operator_mpo(onedim_left_correction, dim, p_field.usites, p_field.arch)

            left_correction = left_correction * val_L
        end
    end

    if bc_dim.right isa DirichletBC
        val_R = bc_dim.right.value

        if abs(val_R) > 1e-14
            onedim_right_correction = onedim_right_dirichlet_correction(derivative_order, p_field.arch, dim, p_field.array_type; stencil_order=stencil_order)
            right_correction = ndim_operator_mpo(onedim_right_correction, dim, p_field.usites, p_field.arch)

            right_correction = right_correction * val_R
        end
    end

    # Sum left and right (now properly scaled)
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
    pressure_dirichlet_correction(p_field::ScalarField{N}, grid::Grid{N}, derivative_order::Int; stencil_order, cutoff, maxdim) where N

Compute the total Dirichlet boundary correction MPO for the pressure field across all dimensions.
Returns `nothing` if no corrections are needed.
"""
function pressure_dirichlet_correction(p_field::ScalarField{N}, grid::Grid{N}, derivative_order::Int; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    corrections = Union{MPO,Nothing}[]

    for i in 1:N
        dim_correction = pressure_dirichlet_correction_dim(p_field, grid, derivative_order, i; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
        push!(corrections, dim_correction)
    end

    return _nothing_sum_handler(corrections...; cutoff=cutoff, maxdim=maxdim)
end
