# TODO This needs heavy polishing
"""
    ndim_operator_mpo(onedim_mpo::MPO, axis::Int, site_inds::Vector{<:Index}, arch::Architecture{N}) where N

Lift a 1D QTT operator in MPO format into an N-dimensional MPO operator acting on the `axis`-th dimension, respecting the qubitization mapping defined by `arch`.

# Arguments

  - `onedim_mpo::MPO`: The 1D operator MPO.
  - `axis::Int`: The logical dimension (1 to N) this operator acts on.
  - `site_inds::Vector{<:Index}`: The full set of site indices for the N-dim system.
  - `arch::Architecture{N}`: The architecture defining the variable mapping.
"""
function ndim_operator_mpo(onedim_mpo::MPO, axis::Int, site_inds::Vector{<:Index}, arch::QTTArchitecture{N}) where {N}
    length(site_inds) == length(arch.site_map) ||
        throw(DimensionMismatch("Length of site_inds must match length of site_map"))

    active_vars = count(==(axis), arch.site_map)
    length(onedim_mpo) == active_vars || throw(
        DimensionMismatch(
            "One-dim MPO length $(length(onedim_mpo)) does not match number of sites for variable $axis in site_map ($active_vars)",
        ),
    )

    array_type = backend_type(onedim_mpo)
    # Handle reversal if needed (negative arch value), reverse already implements a deepcopy operation
    working_mpo = arch.var_nsites[axis] < 0 ? reverse_mpo(onedim_mpo) : deepcopy(onedim_mpo)

    active_indices = findall(==(axis), arch.site_map)
    total_sites = length(site_inds)

    mpo_tensors = Vector{ITensor}(undef, total_sites)

    # --- 1. Handle bond indices and dimensions first ---
    link_dims = ones(Int, total_sites - 1)
    # Propagate MPO bond dimensions through all links between consecutive active sites,
    # so identity sites in interleaved architectures can thread the full bond through.
    for i in 1:(length(active_indices)-1)
        from_idx = active_indices[i]
        to_idx = active_indices[i+1]
        bond_dim = linkdims(working_mpo)[i]
        for j in from_idx:(to_idx-1)
            link_dims[j] = bond_dim
        end
    end

    links = Vector{Index}(undef, total_sites - 1)
    for i in 1:(total_sites-1)
        links[i] = Index(link_dims[i], "Link,l=$i")
    end

    onedim_links = linkinds(working_mpo)
    onedim_sites = [siteind(working_mpo, j) for j in 1:active_vars]
    # --- 2. Handle the first site ---
    first_active = isempty(active_indices) ? total_sites + 1 : active_indices[1]
    if first_active > 1
        mpo_tensors[1] = make_identity_site(site_inds[1], links[1], array_type)
        counter = 1
    else
        old_s = onedim_sites[1]
        new_s = site_inds[1]
        old_right_link = onedim_links[1]
        new_right_link = links[1]

        replaceinds!(working_mpo[1], [old_s, old_s', old_right_link], [new_s, new_s', new_right_link])
        mpo_tensors[1] = working_mpo[1]
        counter = 2
    end

    # --- 3. Handle the middle sites ---
    for i in 2:(total_sites-1)
        if i in active_indices
            # We need to handle left-right bond legs of the first and last tensors in the onedim_mpo separately
            if counter == 1
                old_s = onedim_sites[counter]
                new_s = site_inds[i]

                old_right_link = onedim_links[counter]
                new_right_link = links[i]

                replaceinds!(working_mpo[counter], [old_s, old_s', old_right_link], [new_s, new_s', new_right_link])
                working_mpo[counter] *= adapt(array_type, onehot(links[i-1] => 1)) # Add the left dummy leg
                mpo_tensors[i] = working_mpo[counter]

            elseif counter == active_vars
                old_s = onedim_sites[counter]
                new_s = site_inds[i]

                old_left_link = onedim_links[counter-1]
                new_left_link = links[i-1]

                replaceinds!(working_mpo[counter], [old_s, old_s', old_left_link], [new_s, new_s', new_left_link])
                working_mpo[counter] *= adapt(array_type, onehot(links[i] => 1)) # Add the right dummyleg
                mpo_tensors[i] = working_mpo[counter]

            else
                old_s = onedim_sites[counter]
                new_s = site_inds[i]

                old_left_link = onedim_links[counter-1]
                new_left_link = links[i-1]

                old_right_link = onedim_links[counter]
                new_right_link = links[i]

                replaceinds!(
                    working_mpo[counter],
                    [old_s, old_s', old_left_link, old_right_link],
                    [new_s, new_s', new_left_link, new_right_link],
                )
                mpo_tensors[i] = working_mpo[counter] # No need to add legs, just to update them
            end
            counter += 1
        else
            pass_dim = dim(links[i-1])
            mpo_tensors[i] = make_pass_identity_site(site_inds[i], links[i-1], links[i], pass_dim, array_type)
        end
    end

    # --- 4. Handle the last site ---
    last_active = isempty(active_indices) ? 0 : active_indices[end]
    if last_active < total_sites
        mpo_tensors[total_sites] = make_identity_site(site_inds[total_sites], links[total_sites-1], array_type)

    else
        old_s = onedim_sites[active_vars]
        new_s = last(site_inds)

        old_left_link = onedim_links[active_vars-1]
        new_left_link = links[total_sites-1]

        replaceinds!(working_mpo[active_vars], [old_s, old_s', old_left_link], [new_s, new_s', new_left_link])
        mpo_tensors[total_sites] = working_mpo[active_vars]
    end
    return MPO(mpo_tensors)
end

"""
    ndim_operator_sum_mpo(onedim_ops::NTuple{N, Any}, site_inds::Vector{<:Index}, arch::QTTArchitecture{N}) where N

Construct the exact sum of N 1D operators lifted to an N-dimensional MPO acting on a QTT architecture.
This uses a finite-state machine to construct the MPO directly, guaranteeing an optimal bond dimension.

# Arguments

  - `onedim_ops::NTuple{N, Any}`: The 1D operator MPOs for each dimension.
  - `site_inds::Vector{<:Index}`: The full set of site indices for the N-dim system.
  - `arch::QTTArchitecture{N}`: The architecture defining the variable mapping.
"""
function ndim_operator_sum_mpo(
    onedim_ops::NTuple{N,Any}, site_inds::Vector{<:Index}, arch::QTTArchitecture{N}
) where {N}
    length(site_inds) == length(arch.site_map) ||
        throw(DimensionMismatch("Length of site_inds must match length of site_map"))
    total_sites = length(site_inds)
    mpo_tensors = Vector{ITensor}(undef, total_sites)

    # Handle reversal if needed
    working_mpos = ntuple(i -> arch.var_nsites[i] < 0 ? reverse_mpo(onedim_ops[i]) : deepcopy(onedim_ops[i]), N)
    array_type = backend_type(working_mpos[1])
    active_indices = [findall(==(i), arch.site_map) for i in 1:N]

    D = zeros(Int, max(0, total_sites - 1))
    offsets = [Dict{Int,Int}() for _ in 1:(total_sites-1)]

    # 1. Compute link dimensions and offsets
    for k in 1:(total_sites-1)
        current_dim = 3 # 1: Id, 2: Finished. Active states start at 3.
        for i in 1:N
            p = active_indices[i]
            L_i = length(p)
            if L_i > 1 && p[1] <= k < p[end]
                m = searchsortedlast(p, k)
                chi = dim(commonind(working_mpos[i][m], working_mpos[i][m+1]))
                offsets[k][i] = current_dim
                current_dim += chi
            end
        end
        D[k] = current_dim - 1
    end

    links = [Index(D[k], "Link,l=$k") for k in 1:(total_sites-1)]

    function make_projector(old_ind, new_ind, offset)
        chi = dim(old_ind)
        d_new = dim(new_ind)
        mat = zeros(Float64, chi, d_new)
        for i in 1:chi
            mat[i, offset+i-1] = 1.0
        end
        return adapt(array_type, ITensor(mat, old_ind, new_ind))
    end

    function make_pass_projector(l_ind, r_ind, off_l, off_r, chi)
        mat = zeros(Float64, dim(l_ind), dim(r_ind))
        for i in 1:chi
            mat[off_l+i-1, off_r+i-1] = 1.0
        end
        return adapt(array_type, ITensor(mat, l_ind, r_ind))
    end

    # 2. Construct the tensors
    for k in 1:total_sites
        c = arch.site_map[k]
        a = searchsortedlast(active_indices[c], k)
        L_c = length(active_indices[c])

        old_s = only(siteinds(working_mpos[c], a; plev=0))
        new_s = site_inds[k]
        T_c = replaceinds(working_mpos[c][a], old_s => new_s, old_s' => new_s')

        # Build embedded T_c
        T_embedded = T_c
        if a > 1
            l_old = commonind(working_mpos[c][a-1], working_mpos[c][a])
            P_L = make_projector(l_old, links[k-1], offsets[k-1][c])
            T_embedded *= P_L
        else
            if k > 1
                T_embedded *= adapt(array_type, onehot(links[k-1] => 1))
            end
        end

        if a < L_c
            r_old = commonind(working_mpos[c][a], working_mpos[c][a+1])
            P_R = make_projector(r_old, links[k], offsets[k][c])
            T_embedded *= P_R
        else
            if k < total_sites
                T_embedded *= adapt(array_type, onehot(links[k] => 2))
            end
        end

        # Build Identity and passing terms
        I_mat = Matrix{Float64}(I, dim(new_s), dim(new_s))
        I_site = adapt(array_type, ITensor(I_mat, new_s', new_s))

        W_k = T_embedded

        # Id -> Id
        if k < total_sites
            if k == 1
                W_k += I_site * adapt(array_type, onehot(links[1] => 1))
            else
                W_k += I_site * adapt(array_type, onehot(links[k-1] => 1)) * adapt(array_type, onehot(links[k] => 1))
            end
        end

        # Fin -> Fin
        if k > 1
            if k == total_sites
                W_k += I_site * adapt(array_type, onehot(links[end] => 2))
            else
                W_k += I_site * adapt(array_type, onehot(links[k-1] => 2)) * adapt(array_type, onehot(links[k] => 2))
            end
        end

        # Active passes for j != c
        if 1 < k < total_sites
            for j in 1:N
                if j != c
                    p = active_indices[j]
                    L_j = length(p)
                    if L_j > 1 && p[1] <= k - 1 < p[end]
                        m = searchsortedlast(p, k - 1)
                        chi = dim(commonind(working_mpos[j][m], working_mpos[j][m+1]))
                        P = make_pass_projector(links[k-1], links[k], offsets[k-1][j], offsets[k][j], chi)
                        W_k += P * I_site
                    end
                end
            end
        end

        mpo_tensors[k] = W_k
    end

    return MPO(mpo_tensors)
end

function zeros_padding(mpo::MPO, site_inds::Vector{<:Index}, corner::Symbol=:upleft)
    totlen = length(site_inds)
    mpo_len = length(mpo)
    pad_len = totlen - mpo_len

    array_type = backend_type(mpo)

    if pad_len < 0
        throw(ArgumentError("Target total length must be greater than or equal to input MPO length."))
    elseif pad_len == 0
        return mpo
    end

    raw_matrix = if corner == :upleft # "ul": |0><0|
        [1.0 0.0; 0.0 0.0]
    elseif corner == :upright         # "ur": |0><1|
        [0.0 1.0; 0.0 0.0]
    elseif corner == :bottomleft      # "ll": |1><0|
        [0.0 0.0; 1.0 0.0]
    elseif corner == :bottomright     # "lr": |1><1|
        [0.0 0.0; 0.0 1.0]
    else
        throw(ArgumentError("Unknown corner symbol: $corner"))
    end

    new_mpo = MPO(totlen)

    pad_links = [Index(1, "Link,Pad,$n") for n in 1:pad_len]

    for n in 1:pad_len
        s = site_inds[n]

        T = adapt(array_type, ITensor(raw_matrix, s', s))

        if n == 1
            # First site: Only has a Right link
            T *= onehot(pad_links[n] => 1)
        else
            # Middle pad sites: Left link from (n-1) and Right link to (n)
            T *= onehot(pad_links[n-1] => 1)
            T *= onehot(pad_links[n] => 1)
        end
        new_mpo[n] = T
    end

    join_link = pad_links[end] # This is the link coming out of the padding section

    for k in 1:mpo_len
        old_node = mpo[k]

        s_old = siteinds(mpo, k) # This returns the non-primmed / primmed tuple of site indicies at site k
        s_new = [site_inds[pad_len+k]', site_inds[pad_len+k]]

        T_shifted = adapt(array_type, replaceinds(old_node, s_old => s_new))

        if k == 1
            T_shifted *= adapt(array_type, onehot(join_link => 1))
        end

        new_mpo[pad_len+k] = T_shifted
    end
    return new_mpo
end
