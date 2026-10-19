"""
    make_identity_site(site::Index, link::Index, array_type)
    make_identity_site(site::Index, left_link::Index, right_link::Index, array_type)

Return the identity on `site` (indices `site` and `site'`) as an MPO tensor with one or two link indices, whose
entries are nonzero only at link value 1; used to pad an operator with identities on the sites it does not act on.
"""
function make_identity_site(site::Index, link::Index, array_type)
    T = ITensor(Float64, site, site', link)
    for i in 1:dim(site)
        T[site=>i, site'=>i, link=>1] = 1.0
    end
    return adapt(array_type, T)
end

function make_identity_site(site::Index, left_link::Index, right_link::Index, array_type)
    T = ITensor(Float64, site, site', left_link, right_link)
    for i in 1:dim(site)
        T[site=>i, site'=>i, left_link=>1, right_link=>1] = 1.0
    end
    return adapt(array_type, T)
end

"""
    make_pass_identity_site(site::Index, left_link::Index, right_link::Index, pass_dim::Int, array_type)

Return the MPO tensor ``\\mathbb{I}_{\\text{site}} \\otimes \\mathbb{I}_{\\text{bond}}`` that applies the identity on
`site` and passes the first `pass_dim` values of `left_link` on to `right_link`; it threads the bond of an operator
through the sites of the other variables of an interleaved [`QTTArchitecture`](@ref).
"""
function make_pass_identity_site(site::Index, left_link::Index, right_link::Index, pass_dim::Int, array_type)
    T = ITensor(Float64, site, site', left_link, right_link)
    for s in 1:dim(site)
        for d in 1:pass_dim
            T[site=>s, site'=>s, left_link=>d, right_link=>d] = 1.0
        end
    end
    return adapt(array_type, T)
end

"""
    identity_mpo(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) -> MPO
    identity_mpo(site_dims::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array) -> MPO
    identity_mpo(arch::Architecture, array_type::Type{<:AbstractArray}=Array) -> MPO

Return the identity operator as an MPO of bond dimension 1. It acts on `site_inds`, on new site indices of dimensions
`site_dims` ([`site_indices`](@ref)), or on new site indices for `arch` (`siteinds(arch)`); in the last two cases the
caller gets the sites back through `siteinds(mpo; plev=0)`. On a single site it is a delta tensor.

# Arguments

  - `site_inds`, `site_dims` or `arch`: the sites, their dimensions, or the architecture that gives them.
  - `array_type::Type{<:AbstractArray}=Array`: array backend of the tensors.

# Returns

  - `MPO`: the identity, with site indices at prime levels 0 (input) and 1 (output).

# Examples

```jldoctest
julia> contract_mpo(identity_mpo([2, 3])) == [i == j for i in 1:6, j in 1:6]
true
```

See also [`zeros_mpo`](@ref), [`ones_mpo`](@ref).
"""
function identity_mpo(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array)
    nsite = length(site_inds)

    nsite == 1 && return MPO([adapt(array_type, delta(Float64, site_inds[1], site_inds[1]'))])

    link_inds = link_indices(fill(1, nsite-1))
    tensor_list = ITensor[]

    push!(tensor_list, make_identity_site(site_inds[1], link_inds[1], array_type))
    for i in 2:(nsite-1)
        push!(tensor_list, make_identity_site(site_inds[i], link_inds[i-1], link_inds[i], array_type))
    end
    push!(tensor_list, make_identity_site(site_inds[nsite], link_inds[nsite-1], array_type))
    return MPO(tensor_list)
end

function identity_mpo(site_dims::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array)
    site_inds = site_indices(site_dims)
    return identity_mpo(site_inds, array_type)
end

function identity_mpo(arch::Architecture, array_type::Type{<:AbstractArray}=Array)
    return identity_mpo(collect(Int, site_dims(arch)), array_type)
end

"""
    zeros_mpo(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) -> MPO
    zeros_mpo(site_dims::Vector{Int}, array_type::Type{<:AbstractArray}=Array) -> MPO
    zeros_mpo(arch::Architecture, array_type::Type{<:AbstractArray}=Array) -> MPO

Return the zero operator as an MPO of bond dimension 1, on `site_inds`, on new sites of dimensions `site_dims`, or on
new sites for `arch`. We set every tensor to zero, and not only one of them, so that the MPO stays exactly zero
after any change of gauge.

# Arguments

  - `site_inds`, `site_dims` or `arch`: the sites, their dimensions, or the architecture that gives them.
  - `array_type::Type{<:AbstractArray}=Array`: array backend of the tensors.

See also [`zeros_mps`](@ref), [`identity_mpo`](@ref).
"""
function zeros_mpo(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array)
    mpo = adapt(array_type, random_mpo(site_inds))
    for i in 1:length(site_inds)
        mpo[i] *= 0.0
    end
    return mpo
end

function zeros_mpo(site_dims::Vector{Int}, array_type::Type{<:AbstractArray}=Array)
    site_inds = site_indices(site_dims)
    return zeros_mpo(site_inds, array_type)
end

function zeros_mpo(arch::Architecture, array_type::Type{<:AbstractArray}=Array)
    return zeros_mpo(collect(Int, site_dims(arch)), array_type)
end

"""
    zeros_mps(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1) -> MPS
    zeros_mps(site_dims::Vector{Int}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1) -> MPS
    zeros_mps(arch::Architecture, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1) -> MPS

Return the zero vector as an MPS on `site_inds`, on new sites of dimensions `site_dims`, or on new sites for `arch`.
As in [`zeros_mpo`](@ref), every tensor is zero. The bond dimensions are those of `ITensorMPS.random_mps` with
`linkdims`, i.e. `linkdims` capped by the largest rank that each bond can have (e.g. `[3, 2]` on three sites of
dimension 2 for `linkdims = 3`); a larger bond dimension leaves room for the sweeps of a solver that starts from it.

# Arguments

  - `site_inds`, `site_dims` or `arch`: the sites, their dimensions, or the architecture that gives them.
  - `array_type::Type{<:AbstractArray}=Array`: array backend of the tensors.

# Keywords

  - `linkdims::Int=1`: requested bond dimension.
"""
function zeros_mps(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)
    mps = adapt(array_type, random_mps(site_inds; linkdims=linkdims))
    for i in 1:length(site_inds)
        mps[i] *= 0.0
    end
    return mps
end

function zeros_mps(site_dims::Vector{Int}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)
    site_inds = site_indices(site_dims)
    return zeros_mps(site_inds, array_type; linkdims=linkdims)
end

function zeros_mps(arch::Architecture, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)
    return zeros_mps(collect(Int, site_dims(arch)), array_type; linkdims=linkdims)
end

"""
    ones_mps(site_inds::Vector{<:Index}, bond_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) -> MPS
    ones_mps(site_inds::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array) -> MPS
    ones_mps(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) -> MPS
    ones_mps(site_dims::Vector{Int}, [bond_dim::Int,] array_type::Type{<:AbstractArray}=Array) -> MPS
    ones_mps(arch::Architecture, [bond_dim::Int,] array_type::Type{<:AbstractArray}=Array) -> MPS

Return the MPS whose tensors have all their entries equal to 1, on the given sites (or on new sites of dimensions
`site_dims`, or for `arch`) and with the link indices `bond_inds` (or new links of dimension `bond_dim`, 1 by
default). Every entry of the vector it represents is then ``\\prod_k \\chi_k``, the product of its bond dimensions,
so with bond dimension 1 it is the vector of ones.

# Throws

  - `DimensionMismatch`: if `bond_inds` does not have one index fewer than `site_inds`.

See also [`ones_mpo`](@ref), [`zeros_mps`](@ref).
"""
function ones_mps(site_inds::Vector{<:Index}, bond_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array)
    length(site_inds) == length(bond_inds) + 1 || throw(
        DimensionMismatch(
            "Site indices length should be +1 longer than bond indices length, got $((length(site_inds), length(bond_inds))).",
        ),
    )
    N = length(site_inds)
    M = MPS(N)
    N == 1 && return MPS([ITensor(1.0, site_inds[1])])
    for i in 1:N
        if i == 1
            inds = (site_inds[i], bond_inds[i])
        elseif i == N
            inds = (site_inds[i], bond_inds[i-1])
        else
            inds = (site_inds[i], bond_inds[i], bond_inds[i-1])
        end

        M[i] = adapt(array_type, ITensor(1.0, inds...))
    end

    return M
end

function ones_mps(site_inds::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    bond_inds = link_indices(fill(bond_dim, length(site_inds) - 1))
    return ones_mps(site_inds, bond_inds, array_type)
end

ones_mps(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) = ones_mps(site_inds, 1, array_type)

function ones_mps(site_dims::Vector{Int}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    site_inds = site_indices(site_dims)
    bond_inds = link_indices(fill(bond_dim, length(site_inds) - 1))
    return ones_mps(site_inds, bond_inds, array_type)
end

ones_mps(site_dims::Vector{Int}, array_type::Type{<:AbstractArray}=Array) = ones_mps(site_dims, 1, array_type)

function ones_mps(arch::Architecture, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    ones_mps(collect(Int, site_dims(arch)), bond_dim, array_type)
end
function ones_mps(arch::Architecture, array_type::Type{<:AbstractArray}=Array)
    ones_mps(collect(Int, site_dims(arch)), 1, array_type)
end

"""
    ones_mpo(site_inds::Vector{<:Index}, bond_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) -> MPO
    ones_mpo(site_inds::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array) -> MPO
    ones_mpo(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) -> MPO
    ones_mpo(site_dims::Vector{Int}, [bond_dim::Int,] array_type::Type{<:AbstractArray}=Array) -> MPO
    ones_mpo(arch::Architecture, [bond_dim::Int,] array_type::Type{<:AbstractArray}=Array) -> MPO

Return the MPO whose tensors have all their entries equal to 1, as [`ones_mps`](@ref) does for an MPS. With bond
dimension 1 it is the matrix of ones, written ``\\mathbb{O}`` in Sec. 3.1 of Torrente et al., *Factorized
approximate inverse preconditioners in the tensor train format*.

# Throws

  - `DimensionMismatch`: if `bond_inds` does not have one index fewer than `site_inds`.

See also [`ones_mps`](@ref), [`identity_mpo`](@ref).
"""
function ones_mpo(site_inds::Vector{<:Index}, bond_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array)
    length(site_inds) == length(bond_inds) + 1 || throw(
        DimensionMismatch(
            "Site indices length should be +1 longer than bond indices length, got $((length(site_inds), length(bond_inds))).",
        ),
    )
    N = length(site_inds)
    M = MPO(N)
    N == 1 && return MPO([ITensor(1.0, site_inds[1], site_inds[1]')])
    for i in 1:N
        if i == 1
            inds = (site_inds[i], site_inds[i]', bond_inds[i])
        elseif i == N
            inds = (site_inds[i], site_inds[i]', bond_inds[i-1])
        else
            inds = (site_inds[i], site_inds[i]', bond_inds[i-1], bond_inds[i])
        end

        M[i] = adapt(array_type, ITensor(1.0, inds...))
    end

    return M
end

function ones_mpo(site_inds::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    bond_inds = link_indices(fill(bond_dim, length(site_inds) - 1))
    return ones_mpo(site_inds, bond_inds, array_type)
end

ones_mpo(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) = ones_mpo(site_inds, 1, array_type)

function ones_mpo(site_dims::Vector{Int}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    site_inds = site_indices(site_dims)
    bond_inds = link_indices(fill(bond_dim, length(site_inds) - 1))
    return ones_mpo(site_inds, bond_inds, array_type)
end

ones_mpo(site_dims::Vector{Int}, array_type::Type{<:AbstractArray}=Array) = ones_mpo(site_dims, 1, array_type)

function ones_mpo(arch::Architecture, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    ones_mpo(collect(Int, site_dims(arch)), bond_dim, array_type)
end
function ones_mpo(arch::Architecture, array_type::Type{<:AbstractArray}=Array)
    ones_mpo(collect(Int, site_dims(arch)), 1, array_type)
end

"""
    upper_triang_mpo(site_inds::Vector{<:Index}, include_diag::Bool=true; array_type=Array) -> MPO
    lower_triang_mpo(sites::Vector{<:Index}, include_diag::Bool=true; array_type=Array) -> MPO

Return the upper (lower) triangular matrix of ones, with or without its diagonal, as an MPO of bond dimension 2.

The lower triangular matrix of ones is the mask ``P_L^{(1:N)}`` of the FAI construction, whose entry ``(i, j)`` is 1
if ``i \\ge j`` and 0 otherwise, in the order of [`contract_mpo`](@ref) (site 1 most significant). In order to build
it without forming the dense matrix, we use its exact TT representation, Eq. (14) of Torrente et al., *Factorized
approximate inverse preconditioners in the tensor train format*:

```math
P_L^{(1:N)} = \\begin{bmatrix} \\mathbb{I}^{(d_1)} & \\mathbb{J}^{(d_1)} \\end{bmatrix}
\\bowtie_{k=2}^{N-1}
\\begin{bmatrix} \\mathbb{I}^{(d_k)} & \\mathbb{J}^{(d_k)} \\\\ & \\mathbb{O}^{(d_k)} \\end{bmatrix}
\\bowtie \\begin{bmatrix} \\mathbb{L}^{(d_N)} \\\\ \\mathbb{O}^{(d_N)} \\end{bmatrix},
```

where ``\\bowtie`` is the strong Kronecker product, ``\\mathbb{I}`` the identity, ``\\mathbb{O}`` the matrix of ones,
and ``\\mathbb{L}`` and ``\\mathbb{J}`` the lower triangular matrices of ones with and without the diagonal; without
the diagonal, ``\\mathbb{J}`` replaces ``\\mathbb{L}`` in the last core. The decomposition holds for any site
dimensions ``d_k``, so it serves the TT and the QTT formats alike. `upper_triang_mpo` builds the transposed cores, and
`lower_triang_mpo` is [`transpose_mpo`](@ref) of it.

# Arguments

  - `site_inds::Vector{<:Index}`: sites of the operator, at least one.
  - `include_diag::Bool=true`: keep the diagonal.

# Keywords

  - `array_type::Type{<:AbstractArray}=Array`: array backend of the tensors.

# Examples

```jldoctest
julia> sites = DifferenTT.site_indices([2, 3]);

julia> contract_mpo(DifferenTT.lower_triang_mpo(sites)) == [Float64(i >= j) for i in 1:6, j in 1:6]
true
```

See also [`triang_projector_mpo`](@ref), which turns the mask into a projector in the vectorized space.
"""
function upper_triang_mpo(site_inds::Vector{<:Index}, include_diag::Bool=true; array_type::Type{<:AbstractArray}=Array)
    n = length(site_inds)
    if n == 1
        d = dim(only(site_inds))
        mask = include_diag ? triu(ones(Float64, d, d)) : triu(ones(Float64, d, d), 1)
        return MPO([ITensor(adapt(array_type, mask), only(site_inds)', only(site_inds))])
    end
    mpo_itensors = ITensor[]
    mpo_links = [Index(2, "Link,l=$l") for l in 1:(n-1)]

    # Site 1
    d1 = dim(site_inds[1])
    i1 = Matrix{Float64}(I, d1, d1)
    j1 = triu(ones(Float64, d1, d1), 1)

    l_arr = zeros(Float64, d1, d1, 2)
    l_arr[:, :, 1] = i1
    l_arr[:, :, 2] = j1

    l_arr = adapt(array_type, l_arr)
    push!(mpo_itensors, ITensor(l_arr, site_inds[1]', site_inds[1], mpo_links[1]))

    # Bulk sites
    for k in 2:(n-1)
        dk = dim(site_inds[k])
        ik = Matrix{Float64}(I, dk, dk)
        jk = triu(ones(Float64, dk, dk), 1)
        ok = ones(Float64, dk, dk)

        bulk_arr = zeros(Float64, dk, dk, 2, 2)
        bulk_arr[:, :, 1, 1] = ik
        bulk_arr[:, :, 1, 2] = jk
        bulk_arr[:, :, 2, 2] = ok
        bulk_arr = adapt(array_type, bulk_arr)

        push!(mpo_itensors, ITensor(bulk_arr, site_inds[k]', site_inds[k], mpo_links[k-1], mpo_links[k]))
    end

    # Site n
    dn = dim(site_inds[n])
    un = include_diag ? triu(ones(Float64, dn, dn)) : triu(ones(Float64, dn, dn), 1)
    on = ones(Float64, dn, dn)

    r_arr = zeros(Float64, dn, dn, 2)
    r_arr[:, :, 1] = un
    r_arr[:, :, 2] = on
    r_arr = adapt(array_type, r_arr)

    push!(mpo_itensors, ITensor(r_arr, site_inds[n]', site_inds[n], mpo_links[n-1]))

    return MPO(mpo_itensors)
end

"""
    lower_triang_mpo(sites::Vector{<:Index}, include_diag::Bool=true; array_type=Array) -> MPO

Return the lower triangular matrix of ones as an MPO of bond dimension 2, the mask ``P_L`` of Eq. (14) of the FAI
paper; see [`upper_triang_mpo`](@ref).
"""
function lower_triang_mpo(sites::Vector{<:Index}, include_diag::Bool=true; array_type::Type{<:AbstractArray}=Array)
    return transpose_mpo(upper_triang_mpo(sites, include_diag; array_type=array_type))
end

"""
    triang_projector_mpo(sites::Vector{<:Index}, combiners::Vector{ITensor}, which_triang::Symbol=:upper,
                         include_diag::Bool=true; array_type=Array) -> MPO
    triang_projector_mpo(sites::Vector{<:Index}, which_triang::Symbol=:upper, include_diag::Bool=true;
                         array_type=Array) -> MPO

Return the projector ``\\mathbb{P}_L`` that keeps the lower (or upper) triangular entries of a vectorized matrix and
sets the others to zero, as an MPO of bond dimension 2.

In the FAI construction the unknown is a lower triangular matrix ``\\tilde{G}_L``, which we vectorize in order to
solve for it with the linear solvers of the package. Its triangularity then becomes a constraint on the vector, which
we impose by projection (Sec. 3.1 of Torrente et al., *Factorized approximate inverse preconditioners in the tensor
train format*). To this end, we take the mask of [`lower_triang_mpo`](@ref), vectorize it, and place it on the diagonal
of an operator in the vectorized space, by contracting each site with a delta tensor (Eq. (20) and Fig. 4). The
projector therefore keeps the bond dimension 2 of the mask, and its diagonal entries are 1 on the kept entries and 0
on the others.

The matrix ``X`` on `sites` (rows on `sites'`, columns on `sites`) is vectorized site by site: site ``k`` of the vector
carries the pair (column, row) of site ``k`` of ``X``, merged by `combiners[k]`. Every combiner must merge two indices
of dimension `dim(sites[k])`, the first standing for the column and the second for the row; only their dimensions and
their order matter. For example, `fai_dg` vectorizes ``\\tilde{G}_L`` with `combiner(sites[k], helper_sites[k])`, where
the helper sites carry the row index and become the output sites `sites'` of ``\\tilde{G}_L``, so that the projector
with `which_triang = :lower` keeps the entries with row ``\\ge`` column, in the order of [`contract_mpo`](@ref) (site 1
most significant). On a single site of dimension 2, its diagonal is ``(1, 0, 1, 1)``, as in Eq. (20). The second method
builds the combiners `combiner(sites[k], sites[k]')` itself.

# Arguments

  - `sites::Vector{<:Index}`: sites of the matrix being vectorized.
  - `combiners::Vector{ITensor}`: one combiner per site, merging (column, row).
  - `which_triang::Symbol=:upper`: `:lower` or `:upper`.
  - `include_diag::Bool=true`: keep the diagonal entries.

# Keywords

  - `array_type::Type{<:AbstractArray}=Array`: array backend of the tensors.

# Returns

  - `MPO`: the projector, on the combined indices `combinedind.(combiners)` and their primes.

# Throws

  - `DimensionMismatch`: if `sites` and `combiners` have different lengths.
  - `ArgumentError`: if `which_triang` is neither `:lower` nor `:upper`.

See also [`lower_triang_mpo`](@ref), [`fai_preconditioner`](@ref).
"""
function triang_projector_mpo(
    sites::Vector{<:Index},
    combiners::Vector{ITensor},
    which_triang::Symbol=:upper,
    include_diag::Bool=true;
    array_type::Type{<:AbstractArray}=Array,
)
    length(sites) == length(combiners) || throw(
        DimensionMismatch("Number of sites and combiners do not match, got $(length(sites)) and $(length(combiners))"),
    )
    n = length(sites)

    if which_triang == :upper
        operator = upper_triang_mpo(sites, include_diag; array_type=array_type)
    elseif which_triang == :lower
        operator = lower_triang_mpo(sites, include_diag; array_type=array_type)
    else
        throw(ArgumentError("which_triang can either be :upper or :lower, got $which_triang"))
    end

    combiner_inds = combinedind.(combiners)
    helper_combiners = [combiner(sites[i], sites[i]'; tags="csite=$i") for i in 1:n]
    hcombiner_inds = [combinedind(combiner) for combiner in helper_combiners]

    for i in 1:n
        operator[i] = operator[i] * helper_combiners[i]
        operator[i] = operator[i] * delta(hcombiner_inds[i], combiner_inds[i], combiner_inds[i]')
    end
    return operator
end

function triang_projector_mpo(
    sites::Vector{<:Index},
    which_triang::Symbol=:upper,
    include_diag::Bool=true;
    array_type::Type{<:AbstractArray}=Array,
)
    n = length(sites)
    combiners = [combiner(sites[i], sites[i]'; tags="csite=$i") for i in 1:n]
    triang_projector_mpo(sites, combiners, which_triang, include_diag; array_type=array_type)
end

"""
    arrays2mps(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mps_arrays::Vector{<:AbstractArray}) -> MPS
    arrays2mps(site_indices::Vector{<:Index}, mps_arrays::Vector{<:AbstractArray}) -> MPS

Build an MPS from the arrays of its cores, given with the site index first and the link indices after it: the first
array is ``d_1 \\times \\chi_1``, the middle ones ``d_k \\times \\chi_{k-1} \\times \\chi_k``, and the last one
``d_N \\times \\chi_{N-1}``; a single site takes a vector of length ``d_1``. The second method creates link indices
tagged `"Link,l=\$n"`, with the dimensions read from the last axis of each array. The arrays are wrapped, not copied,
so they also fix the array backend.

# Throws

  - `DimensionMismatch`: if the numbers of indices and arrays, or the bond dimensions, do not match.

See also [`arrays2mpo`](@ref), [`vec2mps`](@ref).
"""
function arrays2mps(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mps_arrays::Vector{<:AbstractArray})
    nsite = length(mps_arrays)

    (length(site_indices) == length(mps_arrays) && length(bond_indices) == length(mps_arrays) - 1) ||
        throw(DimensionMismatch("Index set length do not match with array set length"))
    nsite == 1 && return MPS([ITensor(only(mps_arrays), only(site_indices))])
    dim(bond_indices[1]) == size(mps_arrays[1], 2) == size(mps_arrays[2], 2) ||
        throw(DimensionMismatch("bond dimension mismatch at site 1"))

    output = MPS(nsite)

    s = site_indices[1]
    l_right = bond_indices[1]

    output[1] = ITensor(mps_arrays[1], s, l_right)

    for n in 2:(nsite-1)
        dim(bond_indices[n]) == size(mps_arrays[n], 3) == size(mps_arrays[n+1], 2) ||
            throw(DimensionMismatch("bond dimension mismatch at site $n"))

        s = site_indices[n]
        l_left = bond_indices[n-1]
        l_right = bond_indices[n]

        output[n] = ITensor(mps_arrays[n], s, l_left, l_right)
    end

    s = site_indices[nsite]
    l_left = bond_indices[nsite-1]

    output[nsite] = ITensor(mps_arrays[nsite], s, l_left)

    return output
end

function arrays2mps(site_indices::Vector{<:Index}, mpo_arrays::Vector{<:AbstractArray})
    nsite = length(mpo_arrays)

    bond_indices = Vector{Index}(undef, nsite - 1)

    for n in 1:(nsite-1)
        bond_dim = size(mpo_arrays[n], ndims(mpo_arrays[n]))
        bond_indices[n] = Index(bond_dim, "Link,l=$(n)")
    end

    return arrays2mps(site_indices, bond_indices, mpo_arrays)
end

"""
    arrays2mpo(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mpo_arrays::Vector{<:AbstractArray}) -> MPO
    arrays2mpo(site_indices::Vector{<:Index}, mpo_arrays::Vector{<:AbstractArray}) -> MPO

Build an MPO from the arrays of its cores, given with the output (row) index first, the input (column) index second,
and the link indices after them: the first array is ``d_1 \\times d_1 \\times \\chi_1``, the middle ones
``d_k \\times d_k \\times \\chi_{k-1} \\times \\chi_k``, and the last one ``d_N \\times d_N \\times \\chi_{N-1}``; a
single site takes a ``d_1 \\times d_1`` matrix. The output index is `site_indices[k]'` and the input index
`site_indices[k]`. The second method creates link indices tagged `"Link,l=\$n"`, with the dimensions read from the
last axis of each array.

# Throws

  - `DimensionMismatch`: if the numbers of indices and arrays, or the bond dimensions, do not match.

See also [`arrays2mps`](@ref), [`mat2mpo`](@ref).
"""
function arrays2mpo(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mpo_arrays::Vector{<:AbstractArray})
    nsite = length(mpo_arrays)

    (length(site_indices) == length(mpo_arrays) && length(bond_indices) == length(mpo_arrays) - 1) ||
        throw(DimensionMismatch("Index set length do not match with array set length"))

    nsite == 1 && return MPO([ITensor(only(mpo_arrays), only(site_indices)', only(site_indices))])
    output = MPO(nsite)

    dim(bond_indices[1]) == size(mpo_arrays[1], 3) == size(mpo_arrays[2], 3) ||
        throw(DimensionMismatch("bond dimension mismatch at site 1"))

    s_out = site_indices[1]'
    s_in = site_indices[1]
    l_right = bond_indices[1]

    output[1] = ITensor(mpo_arrays[1], s_out, s_in, l_right)

    for n in 2:(nsite-1)
        dim(bond_indices[n]) == size(mpo_arrays[n], 4) == size(mpo_arrays[n+1], 3) ||
            throw(DimensionMismatch("bond dimension mismatch at site $n"))
        s_out = site_indices[n]'
        s_in = site_indices[n]
        l_left = bond_indices[n-1]
        l_right = bond_indices[n]

        output[n] = ITensor(mpo_arrays[n], s_out, s_in, l_left, l_right)
    end

    s_out = site_indices[nsite]'
    s_in = site_indices[nsite]
    l_left = bond_indices[nsite-1]

    output[nsite] = ITensor(mpo_arrays[nsite], s_out, s_in, l_left)

    return output
end

function arrays2mpo(site_indices::Vector{<:Index}, mpo_arrays::Vector{<:AbstractArray})
    nsite = length(mpo_arrays)

    bond_indices = Vector{Index}(undef, nsite - 1)

    for n in 1:(nsite-1)
        bond_dim = size(mpo_arrays[n], ndims(mpo_arrays[n]))
        bond_indices[n] = Index(bond_dim, "Link,l=$(n)")
    end

    return arrays2mpo(site_indices, bond_indices, mpo_arrays)
end

"""
    vec2tensor(v::AbstractVector, leg_dims::Vector{Int}) -> Array

Reshape `v` into an array with one axis per leg, where `leg_dims` lists the leg lengths from the least significant
(fastest) leg to the most significant one, and the returned array has its axes in the reverse order, the most
significant first, as site 1 is in [`contract_mps`](@ref). For sites of dimensions ``d_1, \\dots, d_N``, site 1 the
most significant, pass `leg_dims = [d_N, …, d_1]` to obtain the ``d_1 \\times \\dots \\times d_N`` array of
[`vec2mps`](@ref); the order only matters when the dimensions differ.

# Examples

```jldoctest
julia> vec2tensor(collect(1:6), [2, 3])
3×2 Matrix{Int64}:
 1  2
 3  4
 5  6
```
"""
function vec2tensor(v::AbstractVector, leg_dims::Vector{Int})
    n = length(leg_dims)
    return permutedims(reshape(v, leg_dims...), n:-1:1)
end

"""
    vec2mps(v::AbstractVector, s::Vector{<:Index}; kwargs...) -> MPS

Return the MPS on the sites `s` of the dense vector `v`, the inverse of [`contract_mps`](@ref) (site 1 most
significant). The tensor of [`vec2tensor`](@ref) is factorized by successive SVDs (`ITensorMPS.MPS`), to which the
keywords, e.g. `cutoff` and `maxdim`, are passed. It needs the dense vector, of length ``\\prod_k d_k``, so it is meant
for small grids and tests.
"""
function vec2mps(v::AbstractVector, s::Vector{<:Index}; kwargs...)
    site_dims = dim.(s)
    return MPS(vec2tensor(v, reverse(site_dims)), s; kwargs...) # vec2tensor takes the least significant leg first
end

"""
    mat2tensor(m::AbstractMatrix, leg_dims::Vector{Int}) -> Array

Reshape the square matrix `m` into an array whose first `N = length(leg_dims)` axes split its row index and whose last
`N` axes split its column index. As in [`vec2tensor`](@ref), `leg_dims` lists the leg lengths from the least
significant to the most significant, and each group of axes is returned with the most significant axis first.
"""
function mat2tensor(m::AbstractMatrix, leg_dims::Vector{Int})
    n = length(leg_dims)
    return permutedims(reshape(m, leg_dims..., leg_dims...), [n:-1:1; 2n:-1:(n+1)])
end

"""
    mat2mpo(m::AbstractMatrix, s::Vector{<:Index}; kwargs...) -> MPO

Return the MPO on the sites `s` (input) and `s'` (output) of the dense matrix `m`, the inverse of
[`contract_mpo`](@ref). The tensor of [`mat2tensor`](@ref) is factorized by successive SVDs (`ITensorMPS.MPO`), to
which the keywords, e.g. `cutoff` and `maxdim`, are passed. As [`vec2mps`](@ref), it is meant for small operators.
"""
function mat2mpo(m::AbstractMatrix, s::Vector{<:Index}; kwargs...)
    site_dims = dim.(s)
    itensor_object = itensor(mat2tensor(m, reverse(site_dims)), s'..., s...) # least significant leg first
    return MPO(itensor_object, s; kwargs...)
end
