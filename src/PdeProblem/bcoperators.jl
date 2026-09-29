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
# Neumann Boundary Corrections (QTT and TT) -> To apply NeumannBC(0.0) into the differential operators themselves
# WARNING -> These corrections break the symmetry in the operators, and so oen must symmetrize the operator
# itself and the rhs acordingly in order to use CG
# -------------------------------------------------------------------------------------------------

function onedim_neumann_boundary_correction(derivative_order::Int, arch::QTTArchitecture, axis::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    left_neumann_corr = (boundary_conditions.left isa NeumannBC) ? left_neumann_boundary_correction(derivative_order, arch, sites, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim) : nothing
    right_neumann_corr = (boundary_conditions.right isa NeumannBC) ? right_neumann_boundary_correction(derivative_order, arch, sites, array_type; stencil_order=stencil_order, cutoff=cutoff, maxdim=maxdim) : nothing

    return _nothing_sum_handler(left_neumann_corr, right_neumann_corr; cutoff=cutoff, maxdim=maxdim)
end

function onedim_neumann_boundary_correction(derivative_order::Int, arch::TTArchitecture, axis::Int, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    dim = arch.var_dims[axis]
    left_neumann_corr = (boundary_conditions.left isa NeumannBC) ? left_neumann_boundary_correction(derivative_order, arch, dim, array_type; stencil_order=stencil_order) : nothing
    right_neumann_corr = (boundary_conditions.right isa NeumannBC) ? right_neumann_boundary_correction(derivative_order, arch, dim, array_type; stencil_order=stencil_order) : nothing
    
    return _nothing_sum_handler(left_neumann_corr, right_neumann_corr)
end

function neumann_correction_matrix(derivative_order::Int, side::Symbol; stencil_order::Int=8)
    coef_dict = derivative_coefficients(derivative_order; stencil_order=stencil_order)
    
    if stencil_order == 2
        mat = zeros(Float64, 2, 2)
        if side == :left
            mat[1, 1:2] .= -coef_dict[-1] .* [0.0, 1.0]
        elseif side == :right
            mat[2, 1:2] .= -coef_dict[1] .* [1.0, 0.0]
        end
        return mat
    elseif stencil_order == 8
        mat = zeros(Float64, 8, 8)
        if side == :left
            ordered_coef = [coef_dict[i] for i in -4:-1]
            left_rolling_coef = vcat([0.0, 1.0], zeros(6))
            toeplitz_matrix = upper_toeplitz_matrix(ordered_coef)

            rolling_matrix = zeros(eltype(left_rolling_coef), 4, length(left_rolling_coef))
            for (idx, j) in enumerate(3:-1:0)
                rolling_matrix[idx, :] = circshift(left_rolling_coef, j)
            end
            correction_matrix = toeplitz_matrix * rolling_matrix
            mat[1:4, 1:8] .= -correction_matrix
        elseif side == :right
            ordered_coef = [coef_dict[i] for i in 4:-1:1]
            right_rolling_coef = vcat(zeros(6), [1.0, 0.0])
            toeplitz_matrix = lower_toeplitz_matrix(ordered_coef)

            rolling_matrix = zeros(eltype(right_rolling_coef), 4, length(right_rolling_coef))
            for (idx, j) in enumerate(0:3)
                rolling_matrix[idx, :] = circshift(right_rolling_coef, -j) 
            end
            correction_matrix = toeplitz_matrix * rolling_matrix
            mat[5:8, 1:8] .= -correction_matrix
        end
        return mat
    else
        throw(ArgumentError("Only stencil orders 2 and 8 are supported."))
    end
end

function left_neumann_boundary_correction(derivative_order::Int, arch::TTArchitecture, mat_dim::Int, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    mat = zeros(Float64, mat_dim, mat_dim)
    
    correction_matrix = neumann_correction_matrix(derivative_order, :left; stencil_order=stencil_order)
    sz = size(correction_matrix, 1)
    mat[1:sz, 1:sz] .= correction_matrix
    return adapt(array_type, mat)
end

function right_neumann_boundary_correction(derivative_order::Int, arch::TTArchitecture, mat_dim::Int, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8)
    mat = zeros(Float64, mat_dim, mat_dim)
    
    correction_matrix = neumann_correction_matrix(derivative_order, :right; stencil_order=stencil_order)
    sz = size(correction_matrix, 1)
    mat[end-sz+1:end, end-sz+1:end] .= correction_matrix
    return adapt(array_type, mat)
end

function left_neumann_boundary_correction(derivative_order::Int, arch::QTTArchitecture, sites::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    correction_matrix = neumann_correction_matrix(derivative_order, :left; stencil_order=stencil_order)
    
    n_dummy = Int(log2(size(correction_matrix, 1)))
    dummy_sites = siteinds(2, n_dummy)
    
    correction_mpo = mat2mpo(correction_matrix, dummy_sites)
    
    return zeros_padding(correction_mpo, sites, :upleft)
end

function right_neumann_boundary_correction(derivative_order::Int, arch::QTTArchitecture, sites::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; stencil_order::Int=8, cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    correction_matrix = neumann_correction_matrix(derivative_order, :right; stencil_order=stencil_order)
    
    n_dummy = Int(log2(size(correction_matrix, 1)))
    dummy_sites = siteinds(2, n_dummy)
    
    correction_mpo = mat2mpo(correction_matrix, dummy_sites)
    
    return zeros_padding(correction_mpo, sites, :bottomright)
end

# -------------------------------------------------------------------------------------------------
# Symmetrization operations, both for the operator and the rhs of a linear system with Neumann BCs
# -------------------------------------------------------------------------------------------------

# MPO Symmetrization
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
function left_neumann_symmetrization_correction_mps(axis::Int, arch::QTTArchitecture, b::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    array_type = backend_type(b)
    
    mat = [1.0 0.0; 0.0 0.0]
    tensors = AbstractArray[]
    for i in 1:nsites
        tensor = permutedims(reshape(mat, 2, 1, 2, 1), (1, 3, 2, 4))
        push!(tensors, tensor)
    end
    onedim_mpo = arrays2mpo(sites, linkinds(fill(1, nsites-1)), tensors)
    onedim_mpo = adapt(array_type, onedim_mpo)
    
    selector_mpo = ndim_operator_mpo(onedim_mpo, axis, siteinds(b), arch)
    return -0.5 * apply(selector_mpo, b; cutoff=cutoff, maxdim=maxdim)
end

function left_neumann_symmetrization_correction_mps(axis::Int, arch::TTArchitecture, b::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    dim = arch.var_dims[axis]
    array_type = backend_type(b)
    
    mat = zeros(Float64, dim, dim)
    mat[1, 1] = 1.0
    mat = adapt(array_type, mat)
    
    selector_mpo = ndim_operator_mpo(mat, axis, siteinds(b), arch)
    return -0.5 * apply(selector_mpo, b; cutoff=cutoff, maxdim=maxdim)
end

function right_neumann_symmetrization_correction_mps(axis::Int, arch::QTTArchitecture, b::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    nsites = abs(arch.var_nsites[axis])
    sites = siteinds(2, nsites)
    array_type = backend_type(b)
    
    mat = [0.0 0.0; 0.0 1.0]
    tensors = AbstractArray[]
    for i in 1:nsites
        tensor = permutedims(reshape(mat, 2, 1, 2, 1), (1, 3, 2, 4))
        push!(tensors, tensor)
    end
    onedim_mpo = arrays2mpo(sites, linkinds(fill(1, nsites-1)), tensors)
    onedim_mpo = adapt(array_type, onedim_mpo)
    
    selector_mpo = ndim_operator_mpo(onedim_mpo, axis, siteinds(b), arch)
    return -0.5 * apply(selector_mpo, b; cutoff=cutoff, maxdim=maxdim)
end

function right_neumann_symmetrization_correction_mps(axis::Int, arch::TTArchitecture, b::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    dim = arch.var_dims[axis]
    array_type = backend_type(b)
    
    mat = zeros(Float64, dim, dim)
    mat[end, end] = 1.0
    mat = adapt(array_type, mat)
    
    selector_mpo = ndim_operator_mpo(mat, axis, siteinds(b), arch)
    return -0.5 * apply(selector_mpo, b; cutoff=cutoff, maxdim=maxdim)
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
# Dirichlet Boundary Corrections -> To move Dirichlet BC values to the RHS of a linear system. These 
# assume for now that the BC Dirichlet value, which then has to be rescaled to the desired value. See 
# [`scalar_dirichlet_correction_dim`](@ref) in the Physics submodule as an exampel
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
    return flag == :upper ? upper_toeplitz_matrix(coefficients) : lower_toeplitz_matrix(coefficients)
end
