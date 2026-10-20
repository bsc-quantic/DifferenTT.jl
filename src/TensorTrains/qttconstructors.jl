"""
    toeplitz_mpo(site_indices::Vector{<:Index}, coeffs::Dict{Int,Float64}, boundary_conditions::BoundaryCond1d,
                 array_type::Type{<:AbstractArray}=Array) -> MPO
    toeplitz_mpo(nsite::Int, coeffs::Dict{Int,Float64}, boundary_conditions::BoundaryCond1d,
                 array_type::Type{<:AbstractArray}=Array) -> MPO

Return the banded Toeplitz matrix ``A_{i, i+k} = c_k`` of a finite-difference stencil on a one-dimensional QTT grid of
``2^N`` points, as an MPO of bond dimension 3, or 9 on its last bond for the wider stencil.

A stencil applied on a uniform grid is a sum of shifted identities, and in the QTT format every shift by a fixed
offset has an exact representation of bond dimension 2, since adding the offset to a binary index only carries one bit
from each site to the next (Kazeev & Khoromskij, 2012). We build the whole band at once: every core is written in terms
of the single-site shift ``J = \\begin{bmatrix} 0 & 1 \\\\ 0 & 0 \\end{bmatrix}``, its transpose and the identity, the
bond carries the carry of the binary addition, and the coefficients enter only in the last core (site ``N``, the least
significant bit). With `coeffs` at the offsets ``-1, 0, 1`` (a second-order stencil) every bond has dimension 3; with
the offsets ``-4, \\dots, 4`` (an eighth-order stencil, ``q = 8``) the last two cores also resolve the offsets up to
4, and the last bond has dimension 9. If `boundary_conditions.left` is a `PeriodicBC`, the band wraps around and the
result is the circulant matrix, ``A_{i, j} = c_k`` whenever ``j - i \\equiv k \\pmod{2^N}``; otherwise the entries
that would fall outside the matrix are dropped, and the boundary rows are left to the corrections of the boundary
conditions. Only the type of `boundary_conditions.left` is read.

# Arguments

  - `site_indices::Vector{<:Index}`: sites of dimension 2, site 1 the most significant bit; the second method creates
    `nsite` new ones.
  - `coeffs::Dict{Int,Float64}`: the coefficient ``c_k`` of each offset ``k``, with keys `-1:1` or `-4:4`.
  - `boundary_conditions::BoundaryCond1d`: periodic or not, read from its left condition.
  - `array_type::Type{<:AbstractArray}=Array`: array backend of the tensors.

# Throws

  - `ArgumentError`: if `coeffs` does not have 3 or 9 entries, or if there are fewer than 2 sites (3 coefficients) or 4
    sites (9 coefficients).
  - `KeyError`: if an offset of the expected range is missing from `coeffs`.

# Examples

```julia
bc = BoundaryCond1d(PeriodicBC(), PeriodicBC())
A = DifferenTT.toeplitz_mpo(siteinds(2, 3), Dict(-1 => 1.0, 0 => -2.0, 1 => 1.0), bc)
contract_mpo(A)  # 8 × 8 circulant second difference
```
"""
function toeplitz_mpo(
    site_indices::Vector{<:Index},
    coeffs::Dict{Int,Float64},
    boundary_conditions::BoundaryCond1d,
    array_type::Type{<:AbstractArray}=Array,
)
    nsite = length(site_indices)
    if length(coeffs) == 9 && nsite < 4
        throw(ArgumentError("nsite must be >= 4 in order to build the toeplitz mpo if length(coeffs)=9, got $nsite"))
    elseif length(coeffs) == 3 && nsite < 2
        throw(ArgumentError("nsite must be >= 2 in order to build the toeplitz mpo if length(coeffs)=3, got $nsite"))
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

    bulk_mat = vcat(hcat(eye, J', J), hcat(Z, J, Z), hcat(Z, Z, J'))
    # Reshape to (PhysRow, BlockRow, PhysCol, BlockCol) -> Permute to (PhysRow, PhysCol, BlockRow, BlockCol)
    bulk = permutedims(reshape(bulk_mat, 2, 3, 2, 3), (1, 3, 2, 4))

    for _ in 2:(nsite-2)
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
        penultimate_mat = vcat(hcat(eye, J', J), hcat(Z, J, Z), hcat(Z, Z, J'))
        # Reshape to (PhysRow, BlockRow, PhysCol, BlockCol) -> Permute to (PhysRow, PhysCol, BlockRow, BlockCol)
        penultimate = permutedims(reshape(bulk_mat, 2, 3, 2, 3), (1, 3, 2, 4))
        nsite > 2 && push!(tensors, penultimate) # With two sites, the leftmost tensor sits right before the rightmost

        rightmost = vcat(
            hcat(coeffs[0] * eye + coeffs[1] * J + coeffs[-1] * J'), hcat(coeffs[-1] * J), hcat(coeffs[1] * J')
        )
        rightmost = permutedims(reshape(rightmost, 2, 3, 2), (1, 3, 2))
        push!(tensors, rightmost)

    else
        throw(ArgumentError("Toeplitz MPO only implemented for 3 or 9 coefficients"))
    end
    return arrays2mpo(site_indices, tensors)
end

function toeplitz_mpo(
    nsite::Int, coeffs::Dict{Int,Float64}, boundary_conditions::BoundaryCond1d, array_type::Type{<:AbstractArray}=Array
)
    site_indices = siteinds(2, nsite)
    return toeplitz_mpo(site_indices, coeffs, boundary_conditions, array_type)
end

"""
    column_pick_mpo(site_inds::Vector{<:Index}, column_index::Int=1, array_type=Array) -> MPO
    column_pick_mpo(site_inds::Vector{<:Index}, column_index::Vector{<:Int}, array_type=Array) -> MPO

Return the matrix whose column `column_index` is the vector of ones and whose other entries are zero,
``\\mathbf{1}\\mathbf{e}_c^\\top``, as an MPO of bond dimension 1 on a QTT grid of ``2^N`` points (sites of dimension 2,
site 1 the most significant bit). Applied to a vector, it copies entry ``c`` to every entry, which the boundary
operators use to spread a boundary value. Every core is ``\\begin{bmatrix} 1 & 0 \\\\ 1 & 0 \\end{bmatrix}`` or
``\\begin{bmatrix} 0 & 1 \\\\ 0 & 1 \\end{bmatrix}``, as the corresponding bit of ``c - 1`` is 0 or 1. With a vector of
indices, the result is the sum of these matrices, with ones in every listed column and a bond dimension of at most the
number of indices.

# Throws

  - `ArgumentError`: if an index is not in `1:2^N`.

See also [`row_pick_mpo`](@ref), [`entry_mpo`](@ref).
"""
function column_pick_mpo(site_inds::Vector{<:Index}, column_index::Int=1, array_type::Type{<:AbstractArray}=Array)
    nsite = length(site_inds)
    bond_inds = link_indices(fill(1, nsite-1))
    tensors = AbstractArray[]

    # Move to backend once
    matrix0 = adapt(array_type, [1.0 0.0; 1.0 0.0])
    matrix1 = adapt(array_type, [0.0 1.0; 0.0 1.0])

    matrices = (matrix0, matrix1)

    1 <= column_index <= 2^nsite ||
        throw(ArgumentError("column_index must lie in 1:$(2^nsite) for $nsite sites, got $column_index"))

    site_bits = decimal_to_binary(column_index - 1, nsite) # Binary counting starts at 0, but we are assuming columns start at 1.

    for bit in site_bits
        matrix = matrices[bit+1]
        tensor = permutedims(reshape(matrix, 2, 1, 2, 1), (1, 3, 2, 4))
        push!(tensors, tensor)
    end
    return arrays2mpo(site_inds, bond_inds, tensors)
end

function column_pick_mpo(
    site_inds::Vector{<:Index}, column_index::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array
)
    sum(column_pick_mpo(site_inds, single_column_index, array_type) for single_column_index in column_index)
end

"""
    row_pick_mpo(site_inds::Vector{<:Index}, row_index::Int=1, array_type=Array) -> MPO
    row_pick_mpo(site_inds::Vector{<:Index}, row_index::Vector{<:Int}, array_type=Array) -> MPO

Return the matrix whose row `row_index` is the vector of ones and whose other entries are zero,
``\\mathbf{e}_r\\mathbf{1}^\\top``, the transpose of [`column_pick_mpo`](@ref); applied to a vector, it writes the sum
of its entries into entry ``r``. With a vector of indices, the result is the sum of these matrices.

# Throws

  - `ArgumentError`: if an index is not in `1:2^N`.
"""
function row_pick_mpo(site_inds::Vector{<:Index}, row_index::Int=1, array_type::Type{<:AbstractArray}=Array)
    column_mpo = column_pick_mpo(site_inds, row_index, array_type)
    row_mpo = transpose_mpo(column_mpo)
    return row_mpo
end

function row_pick_mpo(site_inds::Vector{<:Index}, row_index::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array)
    return sum(row_pick_mpo(site_inds, single_row_index, array_type) for single_row_index in row_index)
end

"""
    entry_mpo(site_inds::Vector{<:Index}, row_index::Int, column_index::Int, array_type=Array) -> MPO

Return the single-entry matrix ``\\mathbf{e}_r \\mathbf{e}_c^\\top`` on a QTT grid, as an MPO of bond dimension 1:
the operator that moves entry `column_index` of a vector to entry `row_index` and discards the others. We build it as
the element-wise product ([`tt_mul`](@ref)) of [`row_pick_mpo`](@ref) and [`column_pick_mpo`](@ref), whose only common
nonzero entry is ``(r, c)``.

# Arguments

  - `site_inds::Vector{<:Index}`: sites of dimension 2, site 1 the most significant bit.
  - `row_index::Int`, `column_index::Int`: position of the nonzero entry, in `1:2^N`.
  - `array_type::Type{<:AbstractArray}=Array`: array backend of the tensors.

# Throws

  - `ArgumentError`: if an index is not in `1:2^N`.
"""
function entry_mpo(
    site_inds::Vector{<:Index}, row_index::Int, column_index::Int, array_type::Type{<:AbstractArray}=Array
)
    return tt_mul(row_pick_mpo(site_inds, row_index, array_type), column_pick_mpo(site_inds, column_index, array_type))
end

"""
    corner_pick_mpo(coefficients::Vector{<:AbstractFloat}, site_inds::Vector{<:Index}, flag::Symbol=:upper,
                    array_type::Type{<:AbstractArray}=Array) -> MPO

Return the ``4 \\times 4`` triangular Toeplitz block of `coefficients` ``(c_1, \\dots, c_4)`` placed in a corner of an
otherwise zero QTT operator on `site_inds` (at least 2 sites), as an MPO of bond dimension at most 2. With
`flag = :upper` the block is upper triangular with first row ``(c_1, c_2, c_3, c_4)`` and sits in the top-left
corner; with `:lower` it is the transposed block, lower triangular with first column ``(c_1, c_2, c_3, c_4)``, in the
bottom-right corner. These are the couplings of the first and last four nodes to the four ghost nodes that an
eighth-order stencil reaches beyond each Dirichlet wall, which the Dirichlet corrections move to the right-hand side.
The block is built on two sites and padded by [`zeros_padding`](@ref). Throws an `ArgumentError` for any other `flag`.
"""
function corner_pick_mpo(
    coefficients::Vector{<:AbstractFloat},
    site_inds::Vector{<:Index},
    flag::Symbol=:upper,
    array_type::Type{<:AbstractArray}=Array,
)
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
    tensor = permutedims(reshape(block1, 2, 2, 2), (1, 2, 3)) # Identity permutation: already (row, column, bond)
    push!(corner_tensors, tensor)

    term1 = coefficients[1] * eye + coefficients[2] * J
    term2 = coefficients[3] * eye + coefficients[2] * J' + coefficients[4] * J

    block2 = vcat(term1, term2)
    tensor = permutedims(reshape(block2, 2, 2, 2), (1, 3, 2))

    push!(corner_tensors, tensor)
    corner_mpo = arrays2mpo(dummy_sites, corner_tensors)

    return zeros_padding(corner_mpo, site_inds, zeros_padding_corner)
end
