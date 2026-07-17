"""
    ITensorMPS.replace_siteinds(mpo::MPO, sites::Vector{I}) where I <: index

Update the siteinds with plev=0, 1 with a new set of siteinds, mathcing the prime levels.
"""
function ITensorMPS.replace_siteinds!(mpo::MPO, new_sites::Vector{I}) where I <: Index
    sites = only.(siteinds(mpo; plev=0))
    nsite = length(mpo)
    for j in 1:nsite
        old_s = sites[j]
        new_s = new_sites[j]
        replaceinds!(mpo[j], old_s => new_s, old_s' => new_s')
    end
end

"""
    Base.transpose(mpo::MPO)

Transpose an MPO by swapping the prime levels of site indices at each site.
"""
Base.transpose(mpo::MPO) = swapprime!(copy(mpo), 0, 1)
    

"""
    Base.reverse(mpo::MPO)

Reverse the site order of an MPO. The output site indices are also reversed, so the tags will
not be in ascending order!
"""
function Base.reverse(mpo::MPO)
    nsite = length(mpo)
    new_mpo = MPO(nsite)

    # Copy tensors to new positions
    for i in 1:nsite
        new_mpo[nsite-i+1] = copy(mpo[i])
    end

    # Reverse link naming -> index system would otherwise have long range connections!
    for k in 1:nsite-1

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
    _nothing_sum_handler(elements::Vararg{<:Any}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Helper to sum a vector of elements where some might be `nothing`.
"""
function _nothing_sum_handler(elements::Vararg{Union{T, Nothing}}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where T <: AbstractMPS
    element_vec = collect(elements)
    filtered_elements = element_vec[.!isnothing.(element_vec)]

    if length(filtered_elements) == 0
        return nothing
    elseif length(filtered_elements) == 1
        return filtered_elements[1]
    else
        return sum(filtered_elements; cutoff=cutoff, maxdim=maxdim)
    end
end

# TODO Test this function
"""
    _nothing_apply_handler(elements::Vector{<:Any}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))

Helper to apply (multiply) a sequence of operators where some might be `nothing`.
"""
function _nothing_apply_handler(elements::Vararg{Union{T, Nothing}}; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) where T <: MPO
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

function replace_siteinds!(M::MPO, sites::Vector{<:Index})
    for j in eachindex(M)
        sj = only(siteinds(M, j; plev=0))
        M[j] = replaceinds(M[j], sj => sites[j], sj' => sites[j]')
    end
    return M
end

replace_siteinds(M::MPO, sites) = replace_siteinds!(copy(M), sites)

function diagmpo(mpo::MPO)
    n = length(mpo)
    out_sites = only.(siteinds(mpo; plev=0))
    placeholder_sites = sim(out_sites)

    deltas = [delta(out_sites[i], out_sites[i]', placeholder_sites[i], placeholder_sites[i]') for i in 1:n]
    out_mpo = replace_siteinds(mpo, placeholder_sites)

    for i in 1:n
        out_mpo[i] *= deltas[i]
    end
    return out_mpo
end

function diagmps(mpo::MPO)
    n = length(mpo)
    out_sites = only.(siteinds(mpo; plev=0))
    placeholder_sites = sim(out_sites)

    deltas = [delta(out_sites[i], placeholder_sites[i], placeholder_sites[i]') for i in 1:n]
    out_mpo = replace_siteinds(mpo, placeholder_sites)

    for i in 1:n
        out_mpo[i] *= deltas[i]
    end
    return MPS([out_mpo...])
end

"""
    concatenate_mpos(site_inds::Vector{<:Index}, mpos::MPO...)

Concatenate multiple MPOs into a single larger MPO along the site dimension.
Glue indices are inserted between the MPOs to connect them.

# Arguments
- `site_inds::Vector{<:Index}`: The site indices for the combined MPO.
- `mpos::MPO...`: Variable number of MPOs to concatenate.
"""
function concatenate_mpos(site_inds::Vector{<:Index}, mpos::MPO...)
    total_len = sum(length, mpos)
    array_type = array_backend(first(mpos))

    @assert length(site_inds) == total_len "Number of new site indices $(length(site_inds)) must match total MPO length ($total_len)."

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

    for k in 1:nmpo-1
        current_idx += length(mpos[k])
        glue_index = Index(1, "Link,l=$k") # Glue index -> ITensor MPSs & MPOs do not have dummy end legs

        combined_tensors[current_idx] *= adapt(array_type, onehot(glue_index => 1))
        combined_tensors[current_idx+1] *= adapt(array_type, onehot(glue_index => 1))
    end

    return MPO(combined_tensors)
end

"""
    concatenate_mpss(site_inds::Vector{<:Index}, mpss::MPS...)

Concatenate multiple MPSs into a single larger MPS along the site dimension.

# Arguments
- `site_inds::Vector{<:Index}`: The site indices for the combined MPS.
- `mpss::MPS...`: Variable number of MPSs to concatenate.
"""
function concatenate_mpss(site_inds::Vector{<:Index}, mpss::Vararg{MPS})
    total_len = sum(length, mpss)
    array_type = array_backend(first(mpss))

    @assert length(site_inds) == total_len "Number of new site indices $(length(site_inds)) must match total MPO length ($total_len)."

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

    for k in 1:nmps-1
        current_idx += length(mpss[k])
        glue_index = Index(1, "Link,l=$k") # Glue index -> ITensor MPSs & MPOs do not have dummy end legs

        combined_tensors[current_idx] *= adapt(array_type, onehot(glue_index => 1))
        combined_tensors[current_idx+1] *= adapt(array_type, onehot(glue_index => 1))
    end

    return MPS(combined_tensors)
end

"""
    contract_mpo(mpo::MPO)

Fully contract an MPO into a dense matrix, reordering from the TT basis to the standard linear basis via bit-reversal permutation.

# Returns
- `Matrix{Float64}`: The dense matrix representation.
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
    contract_mps(mps::MPS)

Fully contract an MPS into a dense vector, reordering from the TT basis to the standard linear basis via bit-reversal permutation.

# Returns
- `Vector{Float64}`: The dense vector representation.
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
    ndim_operator_mpo(onedim_op::AbstractMatrix, axis::Int, site_inds::Vector{<:Index}, arch::TTArchitecture{N}) where N

Lift a 1D operator matrix into an N-dimensional MPO acting on the `axis`-th dimension.
Identities are placed on all other sites.

# Arguments
- `onedim_op::AbstractMatrix`: The 1D operator matrix.
- `axis::Int`: The logical dimension (1 to N) this operator acts on.
- `site_inds::Vector{<:Index}`: The full set of site indices for the N-dim system.
- `arch::TTArchitecture{N}`: The architecture defining the variable mapping.
"""
function ndim_operator_mpo(onedim_op::AbstractMatrix, axis::Int, site_inds::Vector{<:Index}, arch::TTArchitecture{N}) where N
    @assert length(site_inds) == length(arch.site_map) "Length of site_inds must match length of site_map"
    
    total_sites = length(site_inds)
    mpo_tensors = Vector{ITensor}(undef, total_sites)
    
    link_dims = ones(Int, total_sites - 1)
    links = [Index(1, "Link,l=$i") for i in 1:total_sites-1]

    # Determine array type from the matrix type
    array_type = Base.typename(typeof(onedim_op)).wrapper

    for i in 1:total_sites
        if arch.site_map[i] == axis
            s = site_inds[i]
            T = ITensor(array_type(onedim_op), s', s)
            
            if i == 1
                T *= adapt(array_type, onehot(links[1] => 1))
            elseif i == total_sites
                T *= adapt(array_type, onehot(links[end] => 1))
            else
                T *= adapt(array_type, onehot(links[i-1] => 1))
                T *= adapt(array_type, onehot(links[i] => 1))
            end
            mpo_tensors[i] = adapt(array_type, T)
        else
            if i == 1
                mpo_tensors[i] = make_identity_site(site_inds[i], links[i], array_type)
            elseif i == total_sites
                mpo_tensors[i] = make_identity_site(site_inds[i], links[i-1], array_type)
            else
                mpo_tensors[i] = make_identity_site(site_inds[i], links[i-1], links[i], array_type)
            end
        end
    end
    
    return MPO(mpo_tensors)
end

function onedim_eye_padding(mpo::MPO, site_inds::Vector{<:Index}, side::Symbol=:left)
    nsite = length(site_inds)
    mpo_len = length(mpo)
    pad_len = nsite - mpo_len
    array_type = array_backend(mpo)

    if pad_len < 0
        throw(ArgumentError("Length of provided site_inds is less than that of the MPO"))
    elseif pad_len == 0
        return mpo
    end
    eye_sites = side == :left ? site_inds[1:pad_len] : site_inds[pad_len+1:nsite]

    eyes = identity_mpo(eye_sites, array_type)

    if side == :left
        return concatenate_mpos(site_inds, eyes, mpo)
    elseif side == :right
        return concatenate_mpos(site_inds, mpo, eyes)
    else
        throw(ArgumentError("Unknown side symbol: $side. Choose either :left or :right."))
    end
end
