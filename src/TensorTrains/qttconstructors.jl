"""
    toeplitz_mpo(nsite::Int, coeffs::Dict{Int,Float64}, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array)

Create an MPO representing a Toeplitz matrix with either 3 or 9 diagonals (for order 2 and 4 finite difference stencils).

# Arguments
- `site_indices`: Site indices to use for the mpo.
- `coeffs`: Dictionary mapping offset to coefficient (e.g., 0 => 2.0, 1 => -1.0).
- `boundary_conditions`: Boundary conditions to apply at edges.
- `array_type`: Backend array type.
"""
function toeplitz_mpo(site_indices::Vector{<:Index}, coeffs::Dict{Int,Float64}, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array)
    nsite = length(site_indices)
    if length(coeffs) == 9 && nsite < 4
        error("nsite must be >= 4 in order to build the toeplitz mpo if length(coeffs)=9")
    elseif length(coeffs) == 3 && nsite < 2
        error("nsite must be >= 2 in order to build the toeplitz mpo if length(coeffs)=3")
    end

    tensors = AbstractArray[]

    J = adapt(array_type, [0.0 1.0; 0.0 0.0])
    X = J + J'
    Z = adapt(array_type, zeros(Float64, 2, 2))
    eye = adapt(array_type, identity_array(2))

    if boundary_conditions.left isa PeriodicBC # If the left is a PeriodicBC, so is the right
        P = X
        O = X
    else
        P = J
        O = J'
    end

    leftmost_mat = hcat(eye, O, P)
    leftmost = reshape(leftmost_mat, 2, 2, 3)
    push!(tensors, leftmost)

    bulk_mat = vcat(
        hcat(eye, J', J),
        hcat(Z, J, Z),
        hcat(Z, Z, J')
    )
    # Reshape to (PhysRow, BlockRow, PhysCol, BlockCol) -> Permute to (PhysRow, PhysCol, BlockRow, BlockCol)
    bulk = permutedims(reshape(bulk_mat, 2, 3, 2, 3), (1, 3, 2, 4))

    for _ in 2:nsite-2
        push!(tensors, bulk)
    end

    if length(coeffs) == 9
        penultimate_mat = vcat(
            hcat(eye, J', J, J, Z, Z, J', Z, Z),
            hcat(Z, J, Z, Z, eye, Z, J, eye, Z),
            hcat(Z, Z, J', J', Z, eye, Z, Z, eye),
        )
        penultimate = permutedims(reshape(penultimate_mat, 2, 3, 2, 9), (1, 3, 2, 4))
        push!(tensors, penultimate)

        rightmost_mat = vcat(
            hcat(coeffs[0] * eye + coeffs[1] * J + coeffs[-1] * J'),
            hcat(coeffs[-1] * J + coeffs[-2] * eye),
            hcat(coeffs[1] * J' + coeffs[2] * eye),
            hcat(coeffs[3] * J),
            hcat(coeffs[-3] * J),
            hcat(coeffs[3] * J'),
            hcat(coeffs[-3] * J'),
            hcat(coeffs[-4] * eye),
            hcat(coeffs[4] * eye),
        )
        rightmost = permutedims(reshape(rightmost_mat, 2, 9, 2), (1, 3, 2))
        push!(tensors, rightmost)

    elseif length(coeffs) == 3
        penultimate_mat = vcat(
            hcat(eye, J', J),
            hcat(Z, J, Z),
            hcat(Z, Z, J')
        )
        # Reshape to (PhysRow, BlockRow, PhysCol, BlockCol) -> Permute to (PhysRow, PhysCol, BlockRow, BlockCol)
        penultimate = permutedims(reshape(bulk_mat, 2, 3, 2, 3), (1, 3, 2, 4))
        push!(tensors, penultimate)

        rightmost = vcat(
            hcat(coeffs[0] * eye + coeffs[1] * J + coeffs[-1] * J'),
            hcat(coeffs[-1] * J),
            hcat(coeffs[1] * J'),
        )
        rightmost = permutedims(reshape(rightmost, 2, 3, 2), (1, 3, 2))
        push!(tensors, rightmost)

    else
        throw(ArgumentError("Toeplitz MPO only implemented for 3 or 9 coefficients"))
    end
    return arrays2mpo(site_indices, tensors)
end

function toeplitz_mpo(nsite::Int, coeffs::Dict{Int,Float64}, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array)
    site_indices = siteinds(2, nsite)
    return toeplitz_mpo(site_indices, coeffs, boundary_conditions, array_type)
end

# TODO Test this function
"""
    column_pick_mpo(site_inds::Vector{<:Index}, column_index::Union{Int,Vector{<:Int}}, array_type::Type{<:AbstractArray}=Array)

Construct an MPO that projects onto one or more basis vectors (columns).

# Arguments
- `site_inds::Vector{<:Index}`: Site indices.
- `column_index`: Integer index or vector of indices of the column(s) to pick.
- `array_type::Type{<:AbstractArray}`: Backend array type.
"""
function column_pick_mpo(site_inds::Vector{<:Index}, column_index::Int=1, array_type::Type{<:AbstractArray}=Array)
    nsite = length(site_inds)
    bond_inds = linkinds(fill(1, nsite-1))
    tensors = AbstractArray[]

    # Move to backend once
    matrix0 = adapt(array_type, [1.0 0.0; 1.0 0.0])
    matrix1 = adapt(array_type, [0.0 1.0; 0.0 1.0])
    
    matrices = (matrix0, matrix1)

    column_index < 1 && error("MPO index must be >= 1")

    site_bits = decimal_to_binary(column_index - 1, nsite) # Binary counting starts at 0, but we are assuming columns start at 1.

    for bit in site_bits
        matrix = matrices[bit+1]
        tensor = permutedims(reshape(matrix, 2, 1, 2, 1), (1, 3, 2, 4))
        push!(tensors, tensor)
    end
    return arrays2mpo(site_inds, bond_inds, tensors)
end

# TODO Test this function
function column_pick_mpo(site_inds::Vector{<:Index}, column_index::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array)
    sum(column_pick_mpo(site_inds, single_column_index, array_type) for single_column_index in column_index)
end

# TODO Test this function
"""
    row_pick_mpo(site_inds::Vector{<:Index}, row_index::Union{Int,Vector{<:Int}}, array_type::Type{<:AbstractArray}=Array)

Construct an MPO that projects onto one or more basis vectors (rows) (transpose of column pick).

# Arguments
- `site_inds::Vector{<:Index}`: Site indices.
- `row_index`: Integer index or vector of indices of the row(s) to pick.
- `array_type::Type{<:AbstractArray}`: Backend array type.
"""
function row_pick_mpo(site_inds::Vector{<:Index}, row_index::Int=1, array_type::Type{<:AbstractArray}=Array)
    column_mpo = column_pick_mpo(site_inds, row_index, array_type)
    row_mpo = transpose(column_mpo)
    return row_mpo
end

# TODO Test this function
function row_pick_mpo(site_inds::Vector{<:Index}, row_index::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array)
    return sum(row_pick_mpo(site_inds, single_row_index, array_type) for single_row_index in row_index)
end

# TODO Test this function
# TODO FINISH REVISING THIS FUNCTION -> Last one from og operations.jl
"""
    corner_pick_mpo(coefficients::Vector{<:AbstractFloat}, site_inds::Vector{<:Index}, flag::Symbol=:upper, array_type::Type{<:AbstractArray}=Array)

Construct an MPO that acts on a "corner" of the full domain, usually for handling boundary conditions in high-dimensional stencils.
"""
function corner_pick_mpo(coefficients::Vector{<:AbstractFloat}, site_inds::Vector{<:Index}, flag::Symbol=:upper, array_type::Type{<:AbstractArray}=Array)
    corner_tensors = AbstractArray[]
    dummy_sites = siteinds(2, 2)

    eye = adapt(array_type, identity_array(2))

    if flag == :upper
        J = adapt(array_type, [0.0 1.0; 0.0 0.0])
        zeros_padding_corner = :upleft
    elseif flag == :lower
        J = adapt(array_type, [0.0 0.0; 1.0 0.0])
        zeros_padding_corner = :bottomright
    else
        throw(ArgumentError("Flag must be :upper or :lower"))
    end

    block1 = hcat(eye, J)
    tensor = permutedims(reshape(block1, 2, 2, 2), (1, 2, 3)) # Does this permute do anything?
    push!(corner_tensors, tensor)

    term1 = coefficients[1] * eye + coefficients[2] * J
    term2 = coefficients[3] * eye + coefficients[2] * J' + coefficients[4] * J

    block2 = vcat(term1, term2)
    tensor = permutedims(reshape(block2, 2, 2, 2), (1, 3, 2)) 

    push!(corner_tensors, tensor)
    corner_mpo = arrays2mpo(dummy_sites, corner_tensors)

    return zeros_padding(corner_mpo, site_inds, zeros_padding_corner)
end