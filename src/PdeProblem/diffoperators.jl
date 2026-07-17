"""
    lower_toeplitz_matrix(coefficients::Vector{<:Number})

Construct a lower Toeplitz matrix from a vector of coefficients.
"""
function lower_toeplitz_matrix(coefficients::Vector{<:Number})
    n = length(coefficients)
    lower_toeplitz = zeros(eltype(coefficients), n, n)
    for j in 1:n
        for i in j:n
            lower_toeplitz[i, j] = coefficients[i-j+1]
        end
    end
    return lower_toeplitz
end

"""
    upper_toeplitz_matrix(coefficients::Vector{<:Number})

Construct an upper Toeplitz matrix from a vector of coefficients.
"""
function upper_toeplitz_matrix(coefficients::Vector{<:Number})
    return transpose(lower_toeplitz_matrix(coefficients))
end

# -------------------------------------------------------------------------------------------------
# 1D Differential Operators
# -------------------------------------------------------------------------------------------------

"""
    onedim_differential_operator(derivative_order::Int, arch::QTTArchitecture, axis::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order=8, cutoff=1e-15, maxdim=typemax(Int64))

Construct the 1D differential operator MPO for QTT Architecture, including boundary corrections.
"""
function onedim_differential_operator(derivative_order::Int, arch::QTTArchitecture, axis::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    nsite = abs(arch.var_nsites[axis])
    site_indices = siteinds(2, nsite)
    return onedim_differential_operator(derivative_order, site_indices, boundary_conditions, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
end

function onedim_differential_operator(derivative_order::Int, arch::QTTArchitecture, axis::Int, n_eyepad::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    nsite = abs(arch.var_nsites[axis])
    dummy_site_indices = siteinds(2, nsite)
    non_padded_site_indices = siteinds(2, nsite - n_eyepad)
    
    non_paded_onedim_operator = onedim_differential_operator(derivative_order, non_padded_site_indices, boundary_conditions, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
    return onedim_eye_padding(non_paded_onedim_operator, dummy_site_indices, :right)
end

function onedim_differential_operator(derivative_order::Int, site_indices::Vector{<:Index}, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    onedim_mpo = toeplitz_mpo(site_indices, coeffs, boundary_conditions, array_type)

    # Note: we use QTT version of neumann boundary correction explicitly here because we have sites
    neumann_corr = (boundary_conditions.left isa NeumannBC) ? left_neumann_boundary_correction(derivative_order, site_indices, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim) : nothing
    right_neumann_corr = (boundary_conditions.right isa NeumannBC) ? right_neumann_boundary_correction(derivative_order, site_indices, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim) : nothing
    neumann_corr = _nothing_sum_handler(neumann_corr, right_neumann_corr; cutoff=cutoff, maxdim=maxdim)

    if !isnothing(neumann_corr)
        onedim_mpo = +(onedim_mpo, -1.0 * neumann_corr; cutoff=cutoff, maxdim=maxdim)
    end
    return onedim_mpo
end


"""
    onedim_differential_operator(derivative_order::Int, arch::TTArchitecture, axis::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order=8)

Construct the 1D differential operator dense matrix for TT Architecture, including boundary corrections.
"""
function onedim_differential_operator(derivative_order::Int, arch::TTArchitecture, axis::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    dim = arch.var_dims[axis]
    return onedim_differential_operator(derivative_order, dim, boundary_conditions, array_type; stencil_order=stencil_order)
end

function onedim_differential_operator(derivative_order::Int, arch::TTArchitecture, axis::Int, n_eyepad::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    noneye_dim = arch.var_dims[axis] ÷ (2^n_eyepad)
    unpadded_mat = onedim_differential_operator(derivative_order, noneye_dim, boundary_conditions, array_type; stencil_order=stencil_order)
    padded_mat = kron(identity_array(2^n_eyepad), unpadded_mat)
    return adapt(array_type, padded_mat)
end

function onedim_differential_operator(derivative_order::Int, dim::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    
    mat = zeros(Float64, dim, dim)
    for i in 1:dim
        for (offset, val) in coeffs
            j = i + offset
            if boundary_conditions.left isa PeriodicBC # If periodic
                if j < 1
                    j += dim
                elseif j > dim
                    j -= dim
                end
                mat[i, j] = val
            else
                if 1 <= j <= dim
                    mat[i, j] = val
                end
            end
        end
    end

    left_neumann_corr = (boundary_conditions.left isa NeumannBC) ? left_neumann_boundary_correction(derivative_order, dim, array_type; stencil_order=stencil_order) : nothing
    right_neumann_corr = (boundary_conditions.right isa NeumannBC) ? right_neumann_boundary_correction(derivative_order, dim, array_type; stencil_order=stencil_order) : nothing
    
    mat_gpu = adapt(array_type, mat)
    if !isnothing(left_neumann_corr)
        mat_gpu -= left_neumann_corr
    end
    if !isnothing(right_neumann_corr)
        mat_gpu -= right_neumann_corr
    end

    return mat_gpu
end

# -------------------------------------------------------------------------------------------------
# DifferentialOps Structs
# -------------------------------------------------------------------------------------------------

"""
    struct DifferentialOpsScalar{N}

Container for 1st and 2nd derivative operators (MPOs) for a scalar field in N dimensions.
"""
struct DifferentialOpsScalar{N}
    d_ops::NTuple{N,MPO}   # First derivatives Dx, Dy, Dz...
    d2_ops::NTuple{N,MPO}  # Second derivatives Dxx, Dyy, Dzz...
end

function DifferentialOpsScalar(grid::Grid{N}, fields::ScalarField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    array_type = fields.array_type

    function make_ops(order::Int)
        ntuple(i -> begin
                onedim_op = onedim_differential_operator(order, fields.arch, i, fields.bcu[i], array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
                if onedim_op isa MPO
                    onedim_op *= (1 / grid.steps[i]^order)
                else
                    onedim_op = onedim_op .* (1 / grid.steps[i]^order)
                end
                stacked_op = ndim_operator_mpo(onedim_op, i, fields.usites, fields.arch)
                return stacked_op
            end, N)
    end

    d_ops = make_ops(1)
    d2_ops = make_ops(2)

    return DifferentialOpsScalar{N}(d_ops, d2_ops)
end

function DifferentialOpsScalar(grid::Grid{N}, fields::ScalarField{N}, n_eyepad::NTuple{N, Int}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    array_type = fields.array_type

    function make_ops(order::Int)
        ntuple(i -> begin
                # calculate effective steps
                noneye_nsites = fields.arch isa QTTArchitecture ? abs(fields.arch.var_nsites[i]) - n_eyepad[i] : Int(log2(fields.arch.var_dims[i])) - n_eyepad[i]
                effective_step_size = grid.lengths[i] / (2 ^ noneye_nsites)
                
                onedim_op = onedim_differential_operator(order, fields.arch, i, n_eyepad[i], fields.bcu[i], array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
                
                if onedim_op isa MPO
                    onedim_op *= (1 / effective_step_size^order)
                else
                    onedim_op = onedim_op .* (1 / effective_step_size^order)
                end

                stacked_op = ndim_operator_mpo(onedim_op, i, fields.usites, fields.arch)
                return stacked_op
            end, N)
    end

    d_ops = make_ops(1)
    d2_ops = make_ops(2)

    return DifferentialOpsScalar{N}(d_ops, d2_ops)
end

"""
    struct DifferentialOps{N}

Container for differential operators for a vector field (N components).
"""
struct DifferentialOps{N}
    ops::NTuple{N,DifferentialOpsScalar{N}}
end

function DifferentialOps(grid::Grid{N}, fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    ops = ntuple(i -> DifferentialOpsScalar(grid, fields.components[i]; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim), N)
    return DifferentialOps{N}(ops)
end

function Base.getindex(ops::DifferentialOps{N}, i::Int) where N
    return ops.ops[i]
end
