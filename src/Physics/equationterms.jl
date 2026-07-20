# None of these functions implement Dirichlet boundary nullifiers, they do implemente periodic, neumann or open BCs

"""
    advection(fields::NTuple{N, MPS}, diff_ops::DifferentialOps{N}) where N

Compute the advection term `(u ⋅ ∇) u` for an N-dimensional vector field.

# Arguments
- `fields::NTuple{N, MPS}`: The vector field components (velocity).
- `diff_ops::DifferentialOps{N}`: Dimensional differential operators.
"""
function advection(fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    advection_components = ntuple(i -> begin
            terms = [tt_mul(fields[j], apply(diff_ops.ops[i].d_ops[j], fields[i]; cutoff=cutoff, maxdim=maxdim); cutoff=cutoff, maxdim=maxdim) for j in 1:N]
            return sum(terms; cutoff=cutoff, maxdim=maxdim)
        end, N)

    return advection_components
end

"""
    diffusion(fields::NTuple{N, MPS}, diff_ops::DifferentialOps{N}) where N

Compute the diffusion term `∇²u` (Laplacian) for an N-dimensional vector field.

# Arguments
- `fields::NTuple{N, MPS}`: The vector field components.
- `diff_ops::DifferentialOps{N}`: Dimensional differential operators.
"""
function diffusion(fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    diffusion_components = ntuple(i -> begin
            terms = [apply(diff_ops.ops[i].d2_ops[j], fields[i]; cutoff=cutoff, maxdim=maxdim) for j in 1:N]
            return sum(terms; cutoff=cutoff, maxdim=maxdim)
        end, N)

    return diffusion_components
end


"""
    divergence(fields::NTuple{N, MPS}, diff_ops::DifferentialOps{N}) where N

Compute the divergence `∇ ⋅ u` of an N-dimensional vector field.

# Arguments
- `fields::NTuple{N, MPS}`: The vector field components.
- `diff_ops::DifferentialOps{N}`: Dimensional differential operators.
"""
function divergence(fields::NTuple{N,MPS}, diff_ops::DifferentialOps{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    terms = [apply(diff_ops.ops[i].d_ops[i], fields[i]; cutoff=cutoff, maxdim=maxdim) for i in 1:N]
    return sum(terms; cutoff=cutoff, maxdim=maxdim)
end


"""
    gradient(field::MPS, diff_ops::DifferentialOpsScalar{N}) where N

Compute the gradient `∇p` of a scalar field, returning an N-tuple of MPSs.

# Arguments
- `field::MPS`: The scalar field.
- `diff_ops::DifferentialOpsScalar{N}`: Scalar differential operators.
"""
function gradient(field::MPS, diff_ops::DifferentialOpsScalar{N}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    return ntuple(i -> apply(diff_ops.d_ops[i], field; cutoff=cutoff, maxdim=maxdim), N)
end

"""
    symmetrized_laplacian(field::ScalarField{N}, grid::Grid{N}; asymmetry::Float64=1.0, stencil_order::Int=8, maxdim::Int=typemax(Int), cutoff::Float64=0.0) where N

Construct the full Laplacian operator for a scalar field with Neumann symmetrization corrections applied.
Builds 1D d²/dx² operators per dimension, applies Neumann symmetrization, scales by grid spacing, and lifts to N-dimensions.

# Returns
- `MPO`: The symmetrized Laplacian operator.
"""
function symmetrized_laplacian(field::ScalarField{N}, grid::Grid{N}; asymmetry::Float64=1.0, stencil_order::Int=8, maxdim::Int=typemax(Int), cutoff::Float64=0.0) where N

    # helper for loop to create d2x, d2y, d2z...
    d2_ops = ntuple(i -> begin
            onedim_op = onedim_differential_operator(2, field.arch, i, field.bcu[i], backend_type(field); stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
            neumann_symmetrization = onedim_neumann_symmetrization_correction(onedim_op, field.arch, i, field.bcu[i]; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
            
            if !isnothing(neumann_symmetrization)
                onedim_op = +(onedim_op, neumann_symmetrization; cutoff=cutoff, maxdim=maxdim)
            end

            if onedim_op isa MPO
                onedim_op = onedim_op * (1/grid.dx[i]^2)
            else
                onedim_op = onedim_op .* (1/grid.dx[i]^2)
            end
            
            stacked_op = ndim_operator_mpo(onedim_op, i, field.usites, field.arch)
            stacked_op = i == 1 ? asymmetry * stacked_op : stacked_op
            return stacked_op
        end, N)

    return sum([d2_ops...]; alg="directsum")
end
