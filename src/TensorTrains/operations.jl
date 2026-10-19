"""
    transpose_mpo(mpo::MPO) -> MPO

Return the transpose of `mpo`, obtained by swapping the prime levels 0 (input) and 1 (output) of the site index at
every site, so that `contract_mpo(transpose_mpo(mpo)) == transpose(contract_mpo(mpo))`. The bond dimensions do not
change, and the link indices are shared with `mpo`. This is the owned replacement of the former
`Base.transpose(::MPO)` method.
"""
function transpose_mpo(mpo::MPO)
    nsite = length(mpo)
    new_mpo = MPO(nsite)
    for site in 1:nsite
        index = siteind(mpo, site)
        new_mpo[site] = swapinds(mpo[site], index, index')
    end
    return new_mpo
end

"""
    reverse_mpo(mpo::MPO) -> MPO

Return `mpo` with its site order reversed, as needed by a reversed variable of a [`QTTArchitecture`](@ref), whose
first site carries the least significant bit. The site indices move with their tensors, so their tags are no longer
in ascending order, and the link indices are renamed so that bond `k` is tagged `"Link,l=\$k"` again. The tensors are
copies. This is the owned replacement of the former `Base.reverse(::MPO)` method.
"""
function reverse_mpo(mpo::MPO)
    nsite = length(mpo)
    new_mpo = MPO(nsite)

    # Copy tensors to new positions
    for i in 1:nsite
        new_mpo[nsite-i+1] = copy(mpo[i])
    end

    # Reverse link naming -> index system would otherwise have long range connections!
    for k in 1:(nsite-1)
        l_k = commonind(mpo[k], mpo[k+1])
        if !isnothing(l_k)
            new_link = Index(dim(l_k), "Link,l=$(nsite-k)")
            replaceind!(new_mpo[nsite-k], l_k, new_link)
            replaceind!(new_mpo[nsite-k+1], l_k, new_link)
        end
    end
    return new_mpo
end

"""
    _nothing_sum_handler(elements...; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Return the sum of the MPS or MPO among `elements` that are not `nothing` (see [`_sum_elements`](@ref)): `nothing` if
there are none, and the element itself, uncopied, if there is one. The optional terms of an operator, such as the
boundary corrections, are `nothing` when they do not apply.
"""
function _nothing_sum_handler(elements...; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    filtered = filter(!isnothing, elements)

    if isempty(filtered)
        return nothing
    elseif length(filtered) == 1
        return first(filtered)
    else
        return _sum_elements(filtered; cutoff=cutoff, maxdim=maxdim)
    end
end

"""
    _sum_elements(elements; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Return the sum of the MPS or MPO in `elements`: exactly, by direct sum of the bonds, if `cutoff == 0` and `maxdim` is
`typemax(Int)`, and otherwise truncated to `cutoff` and `maxdim` by `ITensorMPS`; one-site tensor trains are summed
tensor by tensor.
"""
function _sum_elements(elements; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    if length(first(elements)) == 1
        return typeof(first(elements))([sum([e[1] for e in elements])])
    end
    if cutoff == 0.0 && maxdim == typemax(Int)
        return sum(collect(elements); alg="directsum")
    else
        return sum(collect(elements); cutoff=cutoff, maxdim=maxdim)
    end
end

"""
    _nothing_apply_handler(elements::Union{MPO,Nothing}...; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Return the product ``M_1 M_2 \\cdots`` of the operators among `elements` that are not `nothing`, applied from left to
right with `apply` and truncated to `cutoff` and `maxdim` after each product; `nothing` if there are none, and the
operator itself, uncopied, if there is one.
"""
function _nothing_apply_handler(elements::Union{MPO,Nothing}...; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    element_vec = collect(elements)
    filtered_elements = element_vec[.!isnothing.(element_vec)]

    if length(filtered_elements) == 0
        return nothing
    elseif length(filtered_elements) == 1
        return filtered_elements[1]
    else
        output_mpo = filtered_elements[1]
        for element in filtered_elements[2:end]
            output_mpo = apply(output_mpo, element; cutoff=cutoff, maxdim=maxdim)
        end
        return output_mpo
    end
end

"""
    replace_mpo_siteinds!(mpo::MPO, sites::Vector{<:Index}) -> MPO
    replace_mpo_siteinds(mpo::MPO, sites::Vector{<:Index}) -> MPO

Replace the site indices of `mpo`, at prime levels 0 (input) and 1 (output), by `sites` and `sites'`. The version with
`!` overwrites the tensors of `mpo` and returns it; the other one works on a copy. For an MPS, use
`ITensorMPS.replace_siteinds!`. These are the owned replacements of the former `replace_siteinds(!)(::MPO, sites)`
methods.
"""
function replace_mpo_siteinds!(M::MPO, sites::Vector{<:Index})
    for j in eachindex(M)
        sj = only(siteinds(M, j; plev=0))
        M[j] = replaceinds(M[j], sj => sites[j], sj' => sites[j]')
    end
    return M
end

"""
    replace_mpo_siteinds(mpo::MPO, sites::Vector{<:Index}) -> MPO

Return a copy of `mpo` with its site indices replaced by `sites` and `sites'`; see [`replace_mpo_siteinds!`](@ref).
"""
replace_mpo_siteinds(M::MPO, sites::Vector{<:Index}) = replace_mpo_siteinds!(copy(M), sites)

"""
    diagmpo(mpo::MPO) -> MPO

Return the diagonal part of the operator `mpo`, ``\\mathrm{Diagonal}(\\mathrm{diag}(M))``, as an MPO on the same site
indices. We contract every site with a delta tensor that keeps only the entries with equal input and output values,
so the bond dimensions do not change.

# Examples

```julia
M = random_mpo(siteinds(2, 3))
contract_mpo(diagmpo(M)) ≈ Diagonal(diag(contract_mpo(M)))  # true
```

See also [`diagmps`](@ref).
"""
function diagmpo(mpo::MPO)
    n = length(mpo)
    out_sites = only.(siteinds(mpo; plev=0))
    placeholder_sites = sim(out_sites)

    deltas = [delta(out_sites[i], out_sites[i]', placeholder_sites[i], placeholder_sites[i]') for i in 1:n]
    out_mpo = replace_mpo_siteinds(mpo, placeholder_sites)

    for i in 1:n
        out_mpo[i] *= deltas[i]
    end
    return out_mpo
end

"""
    diagmps(mpo::MPO) -> MPS

Return the diagonal ``\\mathrm{diag}(M)`` of the operator `mpo` as an MPS on its input sites (prime level 0). As in
[`diagmpo`](@ref), every site is contracted with a delta tensor, so the bond dimensions do not change. The FAI
construction uses it to extract ``\\mathrm{diag}(\\tilde{G}_L)`` before computing ``D`` element-wise (Fig. 6 and
Sec. 3.2 of Torrente et al., *Factorized approximate inverse preconditioners in the tensor train format*).

# Examples

```julia
M = random_mpo(siteinds(2, 3))
contract_mps(diagmps(M)) ≈ diag(contract_mpo(M))  # true
```
"""
function diagmps(mpo::MPO)
    n = length(mpo)
    out_sites = only.(siteinds(mpo; plev=0))
    placeholder_sites = sim(out_sites)

    deltas = [delta(out_sites[i], placeholder_sites[i], placeholder_sites[i]') for i in 1:n]
    out_mpo = replace_mpo_siteinds(mpo, placeholder_sites)

    for i in 1:n
        out_mpo[i] *= deltas[i]
    end
    return MPS([out_mpo...])
end

"""
    concatenate_mpos(site_inds::Vector{<:Index}, mpos::MPO...) -> MPO

Return the Kronecker product ``M_1 \\otimes M_2 \\otimes \\cdots`` of the operators `mpos` as one MPO, whose sites are
those of `mpos[1]`, then those of `mpos[2]`, and so on, renamed to `site_inds`. The operators are joined by link
indices of dimension 1, and their own link indices are replaced by new ones, so the result shares no index with the
inputs except `site_inds`; its array backend is that of `mpos[1]`. Throws a `DimensionMismatch` if `site_inds` does not
have one index per site of `mpos`.

See also [`concatenate_mpss`](@ref), [`onedim_eye_padding`](@ref).
"""
function concatenate_mpos(site_inds::Vector{<:Index}, mpos::MPO...)
    total_len = sum(length, mpos)
    array_type = backend_type(first(mpos))

    length(site_inds) == total_len || throw(
        DimensionMismatch("Number of new site indices $(length(site_inds)) must match total MPO length ($total_len)."),
    )

    combined_tensors = ITensor[]
    current_site_offset = 0

    for mpo in mpos
        new_mpo = adapt(array_type, deepcopy(mpo))
        nsite = length(mpo)

        for s in 1:nsite
            global_idx = current_site_offset + s
            old_site_ind = siteind(new_mpo, s)
            new_site_ind = site_inds[global_idx]
            replaceind!(new_mpo[s], old_site_ind, new_site_ind)
            replaceind!(new_mpo[s], old_site_ind', new_site_ind')
        end

        sim!(linkinds, new_mpo) # This changes the ID's on the linkinds not to confuse ITensors
        push!(combined_tensors, new_mpo...)
        current_site_offset += length(new_mpo)
    end

    nmpo = length(mpos)
    current_idx = 0

    for k in 1:(nmpo-1)
        current_idx += length(mpos[k])
        glue_index = Index(1, "Link,l=$k") # Glue index -> ITensor MPSs & MPOs do not have dummy end legs

        combined_tensors[current_idx] *= adapt(array_type, onehot(glue_index => 1))
        combined_tensors[current_idx+1] *= adapt(array_type, onehot(glue_index => 1))
    end

    return MPO(combined_tensors)
end

"""
    concatenate_mpss(site_inds::Vector{<:Index}, mpss::MPS...) -> MPS

Return the Kronecker product ``\\mathbf{x}_1 \\otimes \\mathbf{x}_2 \\otimes \\cdots`` of the vectors `mpss` as one
MPS, as [`concatenate_mpos`](@ref) does for operators. Throws a `DimensionMismatch` if `site_inds` does not have one
index per site of `mpss`.
"""
function concatenate_mpss(site_inds::Vector{<:Index}, mpss::Vararg{MPS})
    total_len = sum(length, mpss)
    array_type = backend_type(first(mpss))

    length(site_inds) == total_len || throw(
        DimensionMismatch("Number of new site indices $(length(site_inds)) must match total MPO length ($total_len)."),
    )

    combined_tensors = ITensor[]
    current_site_offset = 0

    for mps in mpss
        new_mps = adapt(array_type, deepcopy(mps))
        nsite = length(mps)

        for s in 1:nsite
            global_idx = current_site_offset + s
            old_site_ind = siteind(new_mps, s)
            new_site_ind = site_inds[global_idx]
            replaceind!(new_mps[s], old_site_ind, new_site_ind)
        end

        sim!(linkinds, new_mps) # This changes the ID's on the linkinds not to confuse ITensors
        push!(combined_tensors, new_mps...)
        current_site_offset += length(new_mps)
    end

    nmps = length(mpss)
    current_idx = 0

    for k in 1:(nmps-1)
        current_idx += length(mpss[k])
        glue_index = Index(1, "Link,l=$k") # Glue index -> ITensor MPSs & MPOs do not have dummy end legs

        combined_tensors[current_idx] *= adapt(array_type, onehot(glue_index => 1))
        combined_tensors[current_idx+1] *= adapt(array_type, onehot(glue_index => 1))
    end

    return MPS(combined_tensors)
end

"""
    contract_mpo(mpo::MPO) -> Matrix

Return the dense matrix of `mpo`, with rows indexed by the output sites (prime level 1) and columns by the input sites
(prime level 0). Site 1 is the most significant index of both, so that the MPO of ``A_1 \\otimes A_2 \\otimes \\cdots``,
with ``A_k`` on site ``k``, contracts to `kron(A₁, A₂, …)`; this is the convention of every dense reference in the
package. The element type is that of the tensors. It forms the full ``\\prod_k d_k \\times \\prod_k d_k`` matrix, so it
is meant for tests and small examples.

# Examples

```julia
s = siteinds(2, 2)
A, B = rand(2, 2), rand(2, 2)
M = MPO([ITensor(A, s[1]', s[1]), ITensor(B, s[2]', s[2])])  # bond dimension 1, without link indices
contract_mpo(M) ≈ kron(A, B)  # true
```

See also [`contract_mps`](@ref), [`mat2mpo`](@ref).
"""
function contract_mpo(mpo::MPO)
    # Contract using package-consistent in/out convention
    row_inds = reverse([first(s) for s in siteinds(mpo; plev=1) if !isempty(s)]) # output
    col_inds = reverse([first(s) for s in siteinds(mpo; plev=0) if !isempty(s)]) # input

    row_combiner = combiner(row_inds...)
    col_combiner = combiner(col_inds...)

    T_combined = prod(mpo) * row_combiner * col_combiner
    M_tt = Array(T_combined, combinedind(row_combiner), combinedind(col_combiner))

    return M_tt
end

"""
    contract_mps(mps::MPS) -> Vector

Return the dense vector of `mps`, with site 1 as the most significant index, as in [`contract_mpo`](@ref): the MPS of
``\\mathbf{x}_1 \\otimes \\mathbf{x}_2 \\otimes \\cdots`` contracts to `kron(x₁, x₂, …)`. It forms the full vector of
length ``\\prod_k d_k``, so it is meant for tests and small examples.

See also [`vec2mps`](@ref).
"""
function contract_mps(mps::MPS)
    # Contract using package-consistent in/out convention
    inds = reverse(siteinds(mps))
    combine = combiner(inds...)

    T_combined = prod(mps) * combine
    vector_tt = Array(T_combined, combinedind(combine))

    return vector_tt
end

"""
    ndim_operator_mpo(onedim_op::AbstractMatrix, axis::Int, site_inds::Vector{<:Index}, arch::TTArchitecture{N}) -> MPO

Return the ``N``-dimensional operator that applies the one-dimensional operator `onedim_op` to variable `axis` and the
identity to every other variable, i.e. ``\\mathbb{I} \\otimes \\cdots \\otimes A \\otimes \\cdots \\otimes \\mathbb{I}``
in site order, as an MPO of bond dimension 1. The matrix goes on the site that carries `axis` (`arch.var_map[axis]`),
with its rows on the output index; the array backend is taken from the type of `onedim_op`.

# Arguments

  - `onedim_op::AbstractMatrix`: ``n_{\\text{axis}} \\times n_{\\text{axis}}`` operator on variable `axis`.
  - `axis::Int`: the variable, in `1:N`.
  - `site_inds::Vector{<:Index}`: one site index per variable, in site order (e.g. `siteinds(arch)`).
  - `arch::TTArchitecture{N}`: the layout.

# Throws

  - `DimensionMismatch`: if `site_inds` does not have `N` indices.
  - `ArgumentError`: if `axis` is not in `1:N`.

See also [`ndim_operator_sum_mpo`](@ref), and the method for a [`QTTArchitecture`](@ref).
"""
function ndim_operator_mpo(
    onedim_op::AbstractMatrix, axis::Int, site_inds::Vector{<:Index}, arch::TTArchitecture{N}
) where {N}
    length(site_inds) == length(arch.site_map) ||
        throw(DimensionMismatch("Length of site_inds must match length of site_map"))
    1 <= axis <= N || throw(ArgumentError("axis must lie in 1:$N, got $axis"))

    total_sites = length(site_inds)
    mpo_tensors = Vector{ITensor}(undef, total_sites)

    link_dims = ones(Int, max(0, total_sites - 1))
    links = [Index(1, "Link,l=$i") for i in 1:(total_sites-1)]

    # Determine array type from the matrix type
    array_type = Base.typename(typeof(onedim_op)).wrapper

    for i in 1:total_sites
        if arch.site_map[i] == axis
            s = site_inds[i]
            T = ITensor(array_type(onedim_op), s', s)

            if total_sites > 1
                if i == 1
                    T *= adapt(array_type, onehot(links[1] => 1))
                elseif i == total_sites
                    T *= adapt(array_type, onehot(links[end] => 1))
                else
                    T *= adapt(array_type, onehot(links[i-1] => 1))
                    T *= adapt(array_type, onehot(links[i] => 1))
                end
            end
            mpo_tensors[i] = adapt(array_type, T)
        else
            if total_sites > 1
                if i == 1
                    mpo_tensors[i] = make_identity_site(site_inds[i], links[i], array_type)
                elseif i == total_sites
                    mpo_tensors[i] = make_identity_site(site_inds[i], links[i-1], array_type)
                else
                    mpo_tensors[i] = make_identity_site(site_inds[i], links[i-1], links[i], array_type)
                end
            else
                throw(ArgumentError("Cannot have total_sites == 1 and axis not equal to site_map[1]"))
            end
        end
    end

    return MPO(mpo_tensors)
end

"""
    ndim_operator_sum_mpo(onedim_ops::NTuple{N,AbstractMatrix}, site_inds::Vector{<:Index}, arch::TTArchitecture{N};
                          pair_ops=nothing) -> MPO

Return the Kronecker sum
``\\sum_{c=1}^{N} \\mathbb{I} \\otimes \\cdots \\otimes A_c \\otimes \\cdots \\otimes \\mathbb{I}`` of the
one-dimensional operators `onedim_ops`, one per variable, as an exact MPO of bond dimension 2.

Summing the ``N`` lifted operators of [`ndim_operator_mpo`](@ref) with `+` would give bond dimension ``N`` before
rounding. We instead write the sum directly with the bond-dimension-2 cores of Kazeev & Khoromskij (2012), which behave
as a two-state automaton along the chain: state 1 means that no operator has been placed yet and state 2 that one has,
so the middle cores are ``\\begin{bmatrix} \\mathbb{I} & A_{c(k)} \\\\ & \\mathbb{I} \\end{bmatrix}``, where ``c`` =
`arch.site_map`. In this way [`kronecker_laplacian`](@ref) keeps bond dimension 2 in any number of dimensions, and the
pairwise term gives the mixed derivatives of [`equicorrelated_laplacian`](@ref) with bond dimension 3.

If `pair_ops = (P, Q)` is given, we also add the pairwise term

```math
\\sum_{i < j} P_{c(i)} \\otimes Q_{c(j)},
```

where ``i < j`` runs over the sites, with the identity on every other site. A third state ("``P`` placed, waiting for
``Q``") is needed for it, so the bond dimension becomes 3. `P` and `Q` are indexed by variable, like `onedim_ops`, so
the `P` of the variable on the last site and the `Q` of the variable on the first site are never used. On a single
site the result is `onedim_ops[1]`.

# Arguments

  - `onedim_ops::NTuple{N,AbstractMatrix}`: one operator per variable, ``n_c \\times n_c``.
  - `site_inds::Vector{<:Index}`: one site index per variable, in site order (e.g. `siteinds(arch)`).
  - `arch::TTArchitecture{N}`: the layout.

# Keywords

  - `pair_ops::Union{Nothing,Tuple{NTuple{N,AbstractMatrix},NTuple{N,AbstractMatrix}}}=nothing`: the operators
    `(P, Q)` of the pairwise term.

# Throws

  - `DimensionMismatch`: if `site_inds` does not have `N` indices.

See also [`ndim_operator_mpo`](@ref), and the method for a [`QTTArchitecture`](@ref).
"""
function ndim_operator_sum_mpo(
    onedim_ops::NTuple{N,AbstractMatrix},
    site_inds::Vector{<:Index},
    arch::TTArchitecture{N};
    pair_ops::Union{Nothing,Tuple{NTuple{N,AbstractMatrix},NTuple{N,AbstractMatrix}}}=nothing,
) where {N}
    length(site_inds) == length(arch.site_map) ||
        throw(DimensionMismatch("Length of site_inds must match length of site_map"))

    total_sites = length(site_inds)
    mpo_tensors = Vector{ITensor}(undef, total_sites)

    # States: 1 = nothing placed, 2 = term complete, 3 = P placed and waiting for Q (only with pair_ops)
    has_pairs = !isnothing(pair_ops) && total_sites > 1
    links = [Index(has_pairs ? 3 : 2, "Link,l=$i") for i in 1:(total_sites-1)]

    # Determine array type from the matrix type
    array_type = Base.typename(typeof(onedim_ops[1])).wrapper

    for i in 1:total_sites
        c = arch.site_map[i]
        Oc = onedim_ops[c]
        s = site_inds[i]

        # Base tensors
        T_Oc = ITensor(array_type(Oc), s', s)
        T_I = ITensor(array_type(Matrix{eltype(Oc)}(I, dim(s), dim(s))), s', s)

        if total_sites > 1
            if i == 1
                T = T_I * adapt(array_type, onehot(links[1] => 1)) + T_Oc * adapt(array_type, onehot(links[1] => 2))
            elseif i == total_sites
                T = T_Oc * adapt(array_type, onehot(links[end] => 1)) + T_I * adapt(array_type, onehot(links[end] => 2))
            else
                T =
                    T_I * adapt(array_type, onehot(links[i-1] => 1)) * adapt(array_type, onehot(links[i] => 1)) +
                    T_Oc * adapt(array_type, onehot(links[i-1] => 1)) * adapt(array_type, onehot(links[i] => 2)) +
                    T_I * adapt(array_type, onehot(links[i-1] => 2)) * adapt(array_type, onehot(links[i] => 2))
            end
        else
            T = T_Oc
        end

        if has_pairs
            T_P = ITensor(array_type(pair_ops[1][c]), s', s)
            T_Q = ITensor(array_type(pair_ops[2][c]), s', s)

            if i == 1
                T += T_P * adapt(array_type, onehot(links[1] => 3))
            elseif i == total_sites
                T += T_Q * adapt(array_type, onehot(links[end] => 3))
            else
                T +=
                    T_P * adapt(array_type, onehot(links[i-1] => 1)) * adapt(array_type, onehot(links[i] => 3)) +
                    T_I * adapt(array_type, onehot(links[i-1] => 3)) * adapt(array_type, onehot(links[i] => 3)) +
                    T_Q * adapt(array_type, onehot(links[i-1] => 3)) * adapt(array_type, onehot(links[i] => 2))
            end
        end
        mpo_tensors[i] = adapt(array_type, T)
    end

    return MPO(mpo_tensors)
end

"""
    onedim_eye_padding(mpo::MPO, site_inds::Vector{<:Index}, side::Symbol=:left) -> MPO

Extend the operator `mpo` to the longer chain `site_inds` with identities on the extra sites, placed before it
(`side = :left`, giving ``\\mathbb{I} \\otimes M``) or after it (`:right`, ``M \\otimes \\mathbb{I}``), by
[`concatenate_mpos`](@ref). Returns `mpo` itself if no site is missing, and throws an `ArgumentError` if `site_inds` is
shorter than `mpo` or if `side` is unknown.
"""
function onedim_eye_padding(mpo::MPO, site_inds::Vector{<:Index}, side::Symbol=:left)
    nsite = length(site_inds)
    mpo_len = length(mpo)
    pad_len = nsite - mpo_len
    array_type = backend_type(mpo)

    if pad_len < 0
        throw(ArgumentError("Length of provided site_inds is less than that of the MPO"))
    elseif pad_len == 0
        return mpo
    end
    eye_sites = side == :left ? site_inds[1:pad_len] : site_inds[(nsite-pad_len+1):nsite]

    eyes = identity_mpo(eye_sites, array_type)

    if side == :left
        return concatenate_mpos(site_inds, eyes, mpo)
    elseif side == :right
        return concatenate_mpos(site_inds, mpo, eyes)
    else
        throw(ArgumentError("Unknown side symbol: $side. Choose either :left or :right."))
    end
end
