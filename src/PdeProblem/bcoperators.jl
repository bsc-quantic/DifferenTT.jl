"""
    stencil_rows(stencil_order::Int)

Return the row indices involved in a finite difference stencil of a given order.
"""
function stencil_rows(stencil_order::Int)
    if stencil_order == 8
        return [1, 2, 3, 4]
    elseif stencil_order == 2
        return [1]
    else
        throw(ArgumentError("Only stencil orders 2 or 8 are implemented"))
    end
end

"""
    onedim_boundary_nullifier(arch::QTTArchitecture, axis::Int, bcs::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Create a 1D MPO that nullifies rows (sets them to identity/zero handling) corresponding to boundary conditions (e.g. Dirichlet) for QTTArchitecture.
"""
function onedim_boundary_nullifier(arch::QTTArchitecture, axis::Int, bcs::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    nrows = 2^nsites
    
    onedim_left_row_selector = (bcs.left isa DirichletBC) ? row_pick_mpo(sites, stencil_rows(stencil_order), array_type) : nothing
    onedim_right_row_selector = (bcs.right isa DirichletBC) ? row_pick_mpo(sites, nrows + 1 .- reverse(stencil_rows(stencil_order)), array_type) : nothing

    onedim_row_selector = _nothing_sum_handler(onedim_left_row_selector, onedim_right_row_selector; cutoff=cutoff, maxdim=maxdim)

    if !isnothing(onedim_row_selector)
        id = identity_mpo(sites, array_type)
        onedim_row_deleter = +(id, (-1.0) * tt_mul(onedim_row_selector, id; cutoff=cutoff, maxdim=maxdim); cutoff=cutoff, maxdim=maxdim)
    else
        onedim_row_deleter = nothing
    end

    return onedim_row_deleter
end

"""
    onedim_boundary_nullifier(arch::TTArchitecture, axis::Int, bcs::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Create a 1D dense matrix that nullifies rows (sets them to identity/zero handling) corresponding to boundary conditions (e.g. Dirichlet) for TTArchitecture.
"""
function onedim_boundary_nullifier(arch::TTArchitecture, axis::Int, bcs::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    dim = arch.var_dims[axis]
    mat = identity_array(dim)
    
    if bcs.left isa DirichletBC
        mat[stencil_rows(stencil_order), stencil_rows(stencil_order)] .= 0.0
    end
    
    if bcs.right isa DirichletBC
        rows = dim + 1 .- reverse(stencil_rows(stencil_order))
        mat[rows, rows] .= 0.0
    end
    
    if bcs.left isa DirichletBC || bcs.right isa DirichletBC
        return adapt(array_type, mat)
    else
        return nothing
    end
end

"""
    struct BoundaryNullScalar{N}

Holds the operator (MPO) that enforces boundary conditions (nullifies rows) for a scalar field.
"""
struct BoundaryNullScalar{N}
    u_row_deleter::Union{MPO,Nothing}
end

function BoundaryNullScalar(field::ScalarField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    array_type = backend_type(field)
    deleters = Union{MPO,Nothing}[]

    for i in 1:N
        bc_i = field.bcu[i]
        onedim_deleter = onedim_boundary_nullifier(field.arch, i, bc_i, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim)
        deleter = (isnothing(onedim_deleter)) ? nothing : ndim_operator_mpo(onedim_deleter, i, field.usites, field.arch)
        push!(deleters, deleter)
    end

    u_row_deleter = _nothing_apply_handler(deleters...; cutoff=cutoff, maxdim=maxdim)
    return BoundaryNullScalar{N}(u_row_deleter)
end

"""
    struct BoundaryNull{N}

Holds the boundary enforcement operators for a vector field (N components).
"""
struct BoundaryNull{N}
    components::NTuple{N,BoundaryNullScalar{N}}
end

function BoundaryNull(fields::VectorField{N}; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    components = ntuple(i -> BoundaryNullScalar(fields.components[i]; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim), N)
    return BoundaryNull{N}(components)
end

function Base.getindex(bn::BoundaryNull{N}, i::Int) where N
    return bn.components[i]
end

# -------------------------------------------------------------------------------------------------
# Neumann Boundary Corrections (QTT and TT)
# -------------------------------------------------------------------------------------------------

function onedim_neumann_boundary_correction(derivative_order::Int, arch::QTTArchitecture, axis::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    
    left_neumann_corr = (boundary_conditions.left isa NeumannBC) ? left_neumann_boundary_correction(derivative_order, sites, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim) : nothing
    right_neumann_corr = (boundary_conditions.right isa NeumannBC) ? right_neumann_boundary_correction(derivative_order, sites, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim) : nothing

    return _nothing_sum_handler(left_neumann_corr, right_neumann_corr; cutoff=cutoff, maxdim=maxdim)
end

function onedim_neumann_boundary_correction(derivative_order::Int, arch::TTArchitecture, axis::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    dim = arch.var_dims[axis]
    
    left_neumann_corr = (boundary_conditions.left isa NeumannBC) ? left_neumann_boundary_correction(derivative_order, dim, array_type; stencil_order=stencil_order) : nothing
    right_neumann_corr = (boundary_conditions.right isa NeumannBC) ? right_neumann_boundary_correction(derivative_order, dim, array_type; stencil_order=stencil_order) : nothing
    
    if isnothing(left_neumann_corr) && isnothing(right_neumann_corr)
        return nothing
    elseif isnothing(left_neumann_corr)
        return right_neumann_corr
    elseif isnothing(right_neumann_corr)
        return left_neumann_corr
    else
        return left_neumann_corr + right_neumann_corr
    end
end

# QTT versions (from legacy operators.jl)
function left_neumann_boundary_correction(derivative_order::Int, site_indices::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    if stencil_order == 2
        return _second_left_neumann_boundary_correction(derivative_order, site_indices, array_type; cutoff=cutoff, maxdim=maxdim)
    end

    coef_dict = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    ordered_coef = [coef_dict[i] for i in -4:-1]

    left_rolling_coef = vcat([0.0, 1.0], zeros(6))
    toeplitz_matrix = upper_toeplitz_matrix(ordered_coef)

    rolling_matrix = zeros(eltype(left_rolling_coef), 4, length(left_rolling_coef))
    for (idx, j) in enumerate(3:-1:0)
        rolling_matrix[idx, :] = circshift(left_rolling_coef, j)
    end

    correction_matrix = toeplitz_matrix * rolling_matrix
    correction_vector = -correction_matrix[1, :]

    dummy_indices = siteinds(2, Int(log2(length(correction_vector))))

    correction_mps = adapt(array_type, vec2mps(correction_vector, dummy_indices))
    correction_mpo = tt_mul(correction_mps, column_pick_mpo(dummy_indices, 1, array_type); cutoff=cutoff, maxdim=maxdim)
    padded_correction_mpo = zeros_padding(correction_mpo, site_indices, :upleft)

    for j in 2:4
        correction_vector = -correction_matrix[j, :]
        correction_mps = adapt(array_type, vec2mps(correction_vector, dummy_indices))
        correction_mpo = tt_mul(correction_mps, column_pick_mpo(dummy_indices, j, array_type); cutoff=cutoff, maxdim=maxdim)
        padded_correction_mpo = +(padded_correction_mpo, zeros_padding(correction_mpo, site_indices, :upleft); cutoff=cutoff, maxdim=maxdim)
    end
    return padded_correction_mpo
end

function _second_left_neumann_boundary_correction(derivative_order::Int, site_indices::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    coef_dict = derivative_coefficients(derivative_order; stencil_order=2)
    left_coef = coef_dict[-1]
    reflection_vector = [0.0, 1.0]
    correction_vector = -left_coef * reflection_vector

    mpo_matrix = zeros(Float64, 2, 2)
    mpo_matrix[1, :] = correction_vector

    site = site_indices[end]
    correction_mpo = MPO([ITensor(adapt(array_type, mpo_matrix'), [site, site'])])
    return zeros_padding(correction_mpo, site_indices, :upleft)
end

function right_neumann_boundary_correction(derivative_order::Int, site_indices::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    if stencil_order == 2
        return _second_right_neumann_boundary_correction(derivative_order, site_indices, array_type; cutoff=cutoff, maxdim=maxdim)
    end

    coef_dict = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    ordered_coef = [coef_dict[i] for i in 4:-1:1]
    right_rolling_coef = vcat(zeros(6), [1.0, 0.0])
    toeplitz_matrix = lower_toeplitz_matrix(ordered_coef)

    rolling_matrix = zeros(eltype(right_rolling_coef), 4, length(right_rolling_coef))
    for (idx, j) in enumerate(0:3)
        rolling_matrix[idx, :] = circshift(right_rolling_coef, -j)
    end

    correction_matrix = toeplitz_matrix * rolling_matrix
    correction_vector = -correction_matrix[1, :]

    dummy_indices = siteinds(2, Int(log2(length(correction_vector))))

    correction_mps = adapt(array_type, vec2mps(correction_vector, dummy_indices))
    correction_mpo = tt_mul(correction_mps, column_pick_mpo(dummy_indices, stencil_order - 3, array_type); cutoff=cutoff, maxdim=maxdim)
    padded_correction_mpo = zeros_padding(correction_mpo, site_indices, :bottomright)

    for j in 2:4
        correction_vector = -correction_matrix[j, :]
        correction_mps = adapt(array_type, vec2mps(correction_vector, dummy_indices))
        correction_mpo = tt_mul(correction_mps, column_pick_mpo(dummy_indices, stencil_order - 4 + j, array_type); cutoff=cutoff, maxdim=maxdim)
        padded_correction_mpo = +(padded_correction_mpo, zeros_padding(correction_mpo, site_indices, :bottomright); cutoff=cutoff, maxdim=maxdim)
    end
    return padded_correction_mpo
end

function _second_right_neumann_boundary_correction(derivative_order::Int, site_indices::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    coef_dict = derivative_coefficients(derivative_order; stencil_order=2)
    right_coef = coef_dict[1]
    reflection_vector = [1.0, 0.0]
    correction_vector = -right_coef * reflection_vector

    mpo_matrix = zeros(Float64, 2, 2)
    mpo_matrix[2, :] = correction_vector

    site = site_indices[end]
    correction_mpo = MPO([ITensor(adapt(array_type, mpo_matrix'), [site, site'])])
    return zeros_padding(correction_mpo, site_indices, :bottomright)
end

# TT versions (Dense matrices)
function left_neumann_boundary_correction(derivative_order::Int, dim::Int, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    coef_dict = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    mat = zeros(Float64, dim, dim)
    
    if stencil_order == 2
        left_coef = coef_dict[-1]
        mat[1, 1:2] .= -left_coef .* [0.0, 1.0]
    else
        ordered_coef = [coef_dict[i] for i in -4:-1]
        left_rolling_coef = vcat([0.0, 1.0], zeros(6))
        toeplitz_matrix = upper_toeplitz_matrix(ordered_coef)

        rolling_matrix = zeros(eltype(left_rolling_coef), 4, length(left_rolling_coef))
        for (idx, j) in enumerate(3:-1:0)
            rolling_matrix[idx, :] = circshift(left_rolling_coef, j)
        end
        correction_matrix = toeplitz_matrix * rolling_matrix
        mat[1:4, 1:8] .= -correction_matrix
    end
    return adapt(array_type, mat)
end

function right_neumann_boundary_correction(derivative_order::Int, dim::Int, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    coef_dict = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    mat = zeros(Float64, dim, dim)
    
    if stencil_order == 2
        right_coef = coef_dict[1]
        mat[end, end-1:end] .= -right_coef .* [1.0, 0.0]
    else
        ordered_coef = [coef_dict[i] for i in 4:-1:1]
        right_rolling_coef = vcat(zeros(6), [1.0, 0.0])
        toeplitz_matrix = lower_toeplitz_matrix(ordered_coef)

        rolling_matrix = zeros(eltype(right_rolling_coef), 4, length(right_rolling_coef))
        for (idx, j) in enumerate(0:3)
            rolling_matrix[idx, :] = circshift(right_rolling_coef, -j)
        end
        correction_matrix = toeplitz_matrix * rolling_matrix
        mat[end-3:end, end-7:end] .= -correction_matrix
    end
    return adapt(array_type, mat)
end

# -------------------------------------------------------------------------------------------------
# Symmetrization Operators (Neumann)
# -------------------------------------------------------------------------------------------------

function onedim_neumann_symmetrization_correction(operator::Union{MPO, AbstractMatrix}, arch::Architecture, axis::Int, boundary_conditions::BoundaryCond1d; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    left_sym = (boundary_conditions.left isa NeumannBC) ? left_neumann_symmetrization_correction(operator, arch, axis; cutoff=cutoff, maxdim=maxdim) : nothing
    right_sym = (boundary_conditions.right isa NeumannBC) ? right_neumann_symmetrization_correction(operator, arch, axis; cutoff=cutoff, maxdim=maxdim) : nothing
    
    if arch isa QTTArchitecture
        return _nothing_sum_handler(left_sym, right_sym; cutoff=cutoff, maxdim=maxdim)
    else
        if isnothing(left_sym) && isnothing(right_sym)
            return nothing
        elseif isnothing(left_sym)
            return right_sym
        elseif isnothing(right_sym)
            return left_sym
        else
            return left_sym + right_sym
        end
    end
end

function left_neumann_symmetrization_correction(operator::MPO, arch::QTTArchitecture, axis::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    site_indices = only.(siteinds(operator; plev=0))
    array_type = backend_type(operator)
    row_selector = row_pick_mpo(site_indices, 1, array_type) 
    return -0.5 * tt_mul(row_selector, operator; cutoff=cutoff, maxdim=maxdim)
end

function right_neumann_symmetrization_correction(operator::MPO, arch::QTTArchitecture, axis::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    site_indices = only.(siteinds(operator; plev=0))
    array_type = backend_type(operator)
    N = prod(dim.(only.(siteinds(operator; plev=0))))
    row_selector = row_pick_mpo(site_indices, N, array_type)
    return -0.5 * tt_mul(row_selector, operator; cutoff=cutoff, maxdim=maxdim)
end

function left_neumann_symmetrization_correction(operator::AbstractMatrix, arch::TTArchitecture, axis::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    mat = zero(operator)
    mat[1, :] .= -0.5 .* operator[1, :]
    return mat
end

function right_neumann_symmetrization_correction(operator::AbstractMatrix, arch::TTArchitecture, axis::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    mat = zero(operator)
    mat[end, :] .= -0.5 .* operator[end, :]
    return mat
end

# MPS Symmetrization
function left_neumann_symmetrization_correction_mps(axis::Int, arch::Architecture, b::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    onedim_bits = ones(Int, abs(arch.var_dims[axis])) 
    ndim_bits = ndim_bitstring(onedim_bits, axis, arch)
    array_type = backend_type(b)
    entry_selector_mps = adapt(array_type, MPS(siteinds(b), ndim_bits))
    return -0.5 * tt_mul(entry_selector_mps, b; cutoff=cutoff, maxdim=maxdim)
end

function right_neumann_symmetrization_correction_mps(axis::Int, arch::Architecture, b::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    onedim_bits = 2 * ones(Int, abs(arch.var_dims[axis])) 
    ndim_bits = ndim_bitstring(onedim_bits, axis, arch)
    array_type = backend_type(b)
    entry_selector_mps = adapt(array_type, MPS(siteinds(b), ndim_bits))
    return -0.5 * tt_mul(entry_selector_mps, b; cutoff=cutoff, maxdim=maxdim)
end

function neumann_symmetrization_correction_mps(axis::Int, arch::Architecture, bcs::BoundaryCond1d, b::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    left_sym = (bcs.left isa NeumannBC) ? left_neumann_symmetrization_correction_mps(axis, arch, b; cutoff=cutoff, maxdim=maxdim) : nothing
    right_sym = (bcs.right isa NeumannBC) ? right_neumann_symmetrization_correction_mps(axis, arch, b; cutoff=cutoff, maxdim=maxdim) : nothing
    return _nothing_sum_handler(left_sym, right_sym; cutoff=cutoff, maxdim=maxdim)
end

function neumann_symmetrization_correction_mps(arch::Architecture, bcs::BoundaryConditions{N}, b::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where N
    sym_mpss = ntuple(i -> neumann_symmetrization_correction_mps(i, arch, bcs.conditions[i], b; cutoff=cutoff, maxdim=maxdim), N)
    return _nothing_sum_handler(sym_mpss...; cutoff=cutoff, maxdim=maxdim)
end

# -------------------------------------------------------------------------------------------------
# Dirichlet Boundary Corrections
# -------------------------------------------------------------------------------------------------

function onedim_left_dirichlet_correction(derivative_order::Int, arch::QTTArchitecture, axis::Int, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    left_coeffs = [coeffs[i] for i in -4:-1]
    return corner_pick_mpo(left_coeffs, sites, :upper, array_type)
end

function onedim_right_dirichlet_correction(derivative_order::Int, arch::QTTArchitecture, axis::Int, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    right_coeffs = [coeffs[i] for i in 4:-1:1]
    return corner_pick_mpo(right_coeffs, sites, :lower, array_type)
end

function onedim_left_dirichlet_correction(derivative_order::Int, arch::TTArchitecture, axis::Int, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    dim = arch.var_dims[axis]
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    left_coeffs = [coeffs[i] for i in -4:-1]
    mat = zeros(Float64, dim, dim)
    mat[1:4, 1:4] .= corner_pick_matrix(left_coeffs, :upper)
    return adapt(array_type, mat)
end

function onedim_right_dirichlet_correction(derivative_order::Int, arch::TTArchitecture, axis::Int, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    dim = arch.var_dims[axis]
    coeffs = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    right_coeffs = [coeffs[i] for i in 4:-1:1]
    mat = zeros(Float64, dim, dim)
    mat[end-3:end, end-3:end] .= corner_pick_matrix(right_coeffs, :lower)
    return adapt(array_type, mat)
end

function corner_pick_matrix(coefficients::Vector{<:AbstractFloat}, flag::Symbol=:upper)
    mat = zeros(Float64, 4, 4)
    if flag == :upper
        for (i, c) in enumerate(coefficients)
            mat[i, 1] = c
        end
    elseif flag == :lower
        for (i, c) in enumerate(coefficients)
            mat[end-i+1, end] = c
        end
    end
    return mat
end
