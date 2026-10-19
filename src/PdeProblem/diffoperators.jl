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
    onedim_differential_operator(derivative_order::Int, arch::QTTArchitecture, dim::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order=8, cutoff=1e-15, maxdim=typemax(Int64))

Construct the 1D differential operator MPO for QTT Architecture, including boundary corrections.
"""
function onedim_differential_operator(
    derivative_order::Int,
    arch::QTTArchitecture,
    dim::Int,
    boundary_conditions::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    return onedim_differential_operator(
        derivative_order,
        arch,
        dim,
        0,
        boundary_conditions,
        array_type;
        stencil_order=stencil_order,
        cutoff=cutoff,
        maxdim=maxdim,
    )
end

function onedim_differential_operator(
    derivative_order::Int,
    arch::QTTArchitecture,
    dim::Int,
    n_eyepad::Int,
    boundary_conditions::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
)
    nsite = abs(arch.var_nsites[dim])
    dummy_site_indices = siteinds(2, nsite)
    non_padded_site_indices = siteinds(2, nsite - n_eyepad)

    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    onedim_mpo = toeplitz_mpo(non_padded_site_indices, coeffs, boundary_conditions, array_type)

    # Note: we use QTT version of neumann boundary correction explicitly here because we have sites
    neumann_corr = if (boundary_conditions.left isa NeumannBC)
        left_neumann_boundary_correction(
            derivative_order,
            arch,
            non_padded_site_indices,
            array_type;
            stencil_order=stencil_order,
            cutoff=cutoff,
            maxdim=maxdim,
        )
    else
        nothing
    end
    right_neumann_corr = if (boundary_conditions.right isa NeumannBC)
        right_neumann_boundary_correction(
            derivative_order,
            arch,
            non_padded_site_indices,
            array_type;
            stencil_order=stencil_order,
            cutoff=cutoff,
            maxdim=maxdim,
        )
    else
        nothing
    end
    neumann_corr = _nothing_sum_handler(neumann_corr, right_neumann_corr; cutoff=cutoff, maxdim=maxdim)

    if !isnothing(neumann_corr)
        onedim_mpo = _nothing_sum_handler(onedim_mpo, -1.0 * neumann_corr; cutoff=cutoff, maxdim=maxdim)
    end

    if n_eyepad > 0
        return onedim_eye_padding(onedim_mpo, dummy_site_indices, :right)
    else
        return onedim_mpo
    end
end

"""
    onedim_differential_operator(derivative_order::Int, arch::TTArchitecture, dim::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order=8)

Construct the 1D differential operator dense matrix for TT Architecture, including boundary corrections.
"""
function onedim_differential_operator(
    derivative_order::Int,
    arch::TTArchitecture,
    dim::Int,
    boundary_conditions::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    kwargs...,
)
    return onedim_differential_operator(
        derivative_order, arch, dim, 0, boundary_conditions, array_type; stencil_order=stencil_order, kwargs...
    )
end

function onedim_differential_operator(
    derivative_order::Int,
    arch::TTArchitecture,
    dim::Int,
    n_eyepad::Int,
    boundary_conditions::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array;
    stencil_order::Int=8,
    kwargs...,
)
    noneye_dim = arch.var_dims[dim] - n_eyepad
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)

    mat = zeros(Float64, noneye_dim, noneye_dim)
    for i in 1:noneye_dim
        for (offset, val) in coeffs
            j = i + offset
            if boundary_conditions.left isa PeriodicBC # If periodic
                if j < 1
                    j += noneye_dim
                elseif j > noneye_dim
                    j -= noneye_dim
                end
                mat[i, j] = val
            else
                if 1 <= j <= noneye_dim
                    mat[i, j] = val
                end
            end
        end
    end

    left_neumann_corr = if (boundary_conditions.left isa NeumannBC)
        left_neumann_boundary_correction(derivative_order, arch, noneye_dim, array_type; stencil_order=stencil_order)
    else
        nothing
    end
    right_neumann_corr = if (boundary_conditions.right isa NeumannBC)
        right_neumann_boundary_correction(derivative_order, arch, noneye_dim, array_type; stencil_order=stencil_order)
    else
        nothing
    end

    mat_gpu = adapt(array_type, mat)
    if !isnothing(left_neumann_corr)
        mat_gpu -= left_neumann_corr
    end
    if !isnothing(right_neumann_corr)
        mat_gpu -= right_neumann_corr
    end

    if n_eyepad > 0
        full_dim = arch.var_dims[dim]
        padded_mat_cpu = zeros(eltype(mat), full_dim, full_dim)

        # move unpadded to cpu to embed if it was gpu array
        unpadded_cpu = Array(mat_gpu)
        padded_mat_cpu[1:noneye_dim, 1:noneye_dim] = unpadded_cpu
        for i in (noneye_dim+1):full_dim
            padded_mat_cpu[i, i] = 1.0
        end

        return adapt(array_type, padded_mat_cpu)
    else
        return mat_gpu
    end
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

function DifferentialOpsScalar(
    grid::Grid{N}, fields::ScalarField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    array_type = backend_type(fields)

    function make_ops(order::Int)
        ntuple(
            i -> begin
                onedim_op = onedim_differential_operator(
                    order,
                    fields.arch,
                    i,
                    fields.bcu[i],
                    array_type;
                    stencil_order=stencil_order,
                    cutoff=cutoff,
                    maxdim=maxdim,
                )
                if onedim_op isa MPO
                    onedim_op *= (1 / grid.dx[i]^order)
                else
                    onedim_op = onedim_op .* (1 / grid.dx[i]^order)
                end
                stacked_op = ndim_operator_mpo(onedim_op, i, fields.usites, fields.arch)
                return stacked_op
            end,
            N,
        )
    end

    d_ops = make_ops(1)
    d2_ops = make_ops(2)

    return DifferentialOpsScalar{N}(d_ops, d2_ops)
end

function DifferentialOpsScalar(
    grid::Grid{N},
    fields::ScalarField{N},
    n_eyepad::NTuple{N,Int};
    stencil_order::Int=8,
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
) where {N}
    array_type = backend_type(fields)

    function make_ops(order::Int)
        ntuple(
            i -> begin
                # calculate effective steps
                noneye_nsites = if fields.arch isa QTTArchitecture
                    abs(fields.arch.var_nsites[i]) - n_eyepad[i]
                else
                    Int(log2(fields.arch.var_dims[i])) - n_eyepad[i]
                end
                effective_step_size = grid.lengths[i] / (2 ^ noneye_nsites)

                onedim_op = onedim_differential_operator(
                    order,
                    fields.arch,
                    i,
                    n_eyepad[i],
                    fields.bcu[i],
                    array_type;
                    stencil_order=stencil_order,
                    cutoff=cutoff,
                    maxdim=maxdim,
                )

                if onedim_op isa MPO
                    onedim_op *= (1 / effective_step_size^order)
                else
                    onedim_op = onedim_op .* (1 / effective_step_size^order)
                end

                stacked_op = ndim_operator_mpo(onedim_op, i, fields.usites, fields.arch)
                return stacked_op
            end,
            N,
        )
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

function DifferentialOps(
    grid::Grid{N}, fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
) where {N}
    ops = ntuple(
        i -> DifferentialOpsScalar(
            grid, fields.components[i]; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim
        ),
        N,
    )
    return DifferentialOps{N}(ops)
end

function Base.getindex(ops::DifferentialOps{N}, i::Int) where {N}
    return ops.ops[i]
end
