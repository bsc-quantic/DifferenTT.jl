function make_identity_site(site::Index, link::Index, array_type)
    T = ITensor(array_type, site, site', link)
    for i in 1:dim(site)
        T[site => i, site' => i, link => 1] = 1.0
    end
    return T
end

function make_identity_site(site::Index, left_link::Index, right_link::Index, array_type)
    T = ITensor(array_type, site, site', left_link, right_link)
    for i in 1:dim(site)
        T[site => i, site' => i, left_link => 1, right_link => 1] = 1.0
    end
    return T
end

function identity_mpo(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array)
    nsite = length(site_inds)

    link_inds = linkinds(fill(1, nsite-1))
    tensor_list = ITensor[]

    push!(tensor_list, make_identity_site(site_inds[1], link_inds[1], array_type))
    for i in 2:nsite-1
        push!(tensor_list, make_identity_site(site_inds[i], link_inds[i-1], link_inds[i], array_type))
    end
    push!(tensor_list, make_identity_site(site_inds[nsite], link_inds[nsite-1], array_type))
    return MPO(tensor_list)
end

function identity_mpo(site_dims::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array)
    site_inds = siteinds(site_dims)
    return identity_mpo(site_inds, array_type)
end

function identity_mpo(arch::Architecture, array_type::Type{<:AbstractArray}=Array)
    return identity_mpo(collect(Int, site_dims(arch)), array_type)
end

# TODO Test this function
# TODO Should we bring to zero all the tensors, as done now, or only do 0.0 * mpo (sets to zero the ortho center)
"""
    zeros_mpo(sites::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)

Construct an all-zeros MPO with the given site indices and bond dimension.
"""
function zeros_mpo(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)
    mpo = adapt(array_type, random_mpo(site_inds, linkdims))
    for i in 1:length(site_inds)
        mpo[i] *= 0.0
    end
    return mpo
end

function zeros_mpo(site_dims::Vector{Int}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)
    site_inds = siteinds(site_dims)
    return zeros_mpo(site_inds, array_type; linkdims=linkdims)
end

function zeros_mpo(arch::Architecture, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)
    return zeros_mpo(collect(Int, site_dims(arch)), array_type; linkdims=linkdims)
end


# TODO Test this function
# TODO Should we bring to zero all the tensors, as done now, or only do 0.0 * mps (sets to zero the ortho center)
"""
    zeros_mps(sites::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)

Construct an all-zeros MPS with the given site indices and bond dimension.
"""
function zeros_mps(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)
    mps = adapt(array_type, random_mps(site_inds; linkdims=linkdims))
    for i in 1:length(site_inds)
        mps[i] *= 0.0
    end
    return mps
end

function zeros_mps(site_dims::Vector{Int}, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)
    site_inds = siteinds(site_dims)
    return zeros_mps(site_inds, array_type; linkdims=linkdims)
end

function zeros_mps(arch::Architecture, array_type::Type{<:AbstractArray}=Array; linkdims::Int=1)
    return zeros_mps(collect(Int, site_dims(arch)), array_type; linkdims=linkdims)
end

# TODO Test these functions
"""
    ones_mps(site_indices::Vector{<:Index}, bond_dims::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array)

Construct an MPS whose every tensor entry is 1.0, with per-bond dimensions given by `bond_dims`.
"""
function ones_mps(site_inds::Vector{<:Index}, bond_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array)
    @assert length(site_inds) == length(bond_inds) + 1 "Site indices length should be +1 longer than bond indices length, got $((length(site_inds), length(bond_inds)))."
    N = length(site_inds)
    M = MPS(N)
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

"""
    ones_mps(site_indices::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)

Construct an MPS whose every tensor entry is 1.0, with uniform bond dimension `bond_dim`.
"""
function ones_mps(site_inds::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    bond_inds = linkinds(fill(bond_dim, length(site_inds) - 1))
    return ones_mps(site_inds, bond_inds, array_type)
end

# TODO This assumes physical legs to always have dimension 2!
function ones_mps(site_dims::Vector{Int}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    site_inds = siteinds(site_dims)
    bond_inds = linkinds(fill(bond_dim, length(site_inds) - 1))
    return ones_mps(site_inds, bond_inds, array_type)
end

function ones_mps(arch::Architecture, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    return ones_mps(collect(Int, site_dims(arch)), bond_dim, array_type)
end

# TODO Test these functions
"""
    ones_mpo(site_indices::Vector{<:Index}, bond_dims::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array)

Construct an MPO whose every tensor entry is 1.0, with per-bond dimensions given by `bond_dims`.
"""
function ones_mpo(site_inds::Vector{<:Index}, bond_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array)
    @assert length(site_inds) == length(bond_inds) + 1 "Site indices length should be +1 longer than bond indices length, got $((length(site_inds), length(bond_inds)))."
    N = length(site_inds)
    M = MPO(N)
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

"""
    ones_mpo(site_indices::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)

Construct an MPS whose every tensor entry is 1.0, with uniform bond dimension `bond_dim`.
"""
function ones_mpo(site_inds::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    bond_inds = linkinds(fill(bond_dim, length(site_inds) - 1))
    return ones_mpo(site_inds, bond_inds, array_type)
end

# TODO This assumes physical legs to always have dimension 2!
function ones_mpo(site_dims::Vector{Int}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    site_inds = siteinds(site_dims)
    bond_inds = linkinds(fill(bond_dim, length(site_inds) - 1))
    return ones_mpo(site_inds, bond_inds, array_type)
end

function ones_mpo(arch::Architecture, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    return ones_mpo(collect(Int, site_dims(arch)), bond_dim, array_type)
end

"""
    arrays2mps(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mps_arrays::Vector{<:AbstractArray})

Construct an MPS from a list of arrays, using provided site and bond indices.

# Arguments
- `site_indices`: Vector of site indices for the MPS.
- `bond_indices`: Vector of bond indices (links) between MPS tensors.
- `mps_arrays`: Vector of arrays containing the tensor data.
"""
function arrays2mps(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mps_arrays::Vector{<:AbstractArray})
    nsite = length(mps_arrays)

    @assert length(site_indices) == length(mps_arrays) && length(bond_indices) == length(mps_arrays) - 1 "Index set length do not match with array set length"
    @assert dim(bond_indices[1]) == size(mps_arrays[1], 2) == size(mps_arrays[2], 2) "bond dimension mismatch at site 1"

    output = MPS(nsite)

    s = site_indices[1]
    l_right = bond_indices[1]

    output[1] = ITensor(mps_arrays[1], s, l_right)

    for n in 2:nsite-1
        @assert dim(bond_indices[n]) == size(mps_arrays[n], 3) == size(mps_arrays[n+1], 2) "bond dimension mismatch at site $n"

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

"""
    arrays2mpo(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mpo_arrays::Vector{<:AbstractArray})

Construct an MPO from a list of arrays, using provided site and bond indices.

# Arguments
- `site_indices`: Vector of site indices (will be primed for output, unprimed for input).
- `bond_indices`: Vector of bond indices (links).
- `mpo_arrays`: Vector of arrays containing the tensor data.
"""
function arrays2mpo(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mpo_arrays::Vector{<:AbstractArray})
    nsite = length(mpo_arrays)

    @assert length(site_indices) == length(mpo_arrays) && length(bond_indices) == length(mpo_arrays) - 1 "Index set length do not match with array set length"

    output = MPO(nsite)

    @assert dim(bond_indices[1]) == size(mpo_arrays[1], 3) == size(mpo_arrays[2], 3) "bond dimension mismatch at site 1"

    s_out = site_indices[1]'
    s_in = site_indices[1]
    l_right = bond_indices[1]

    output[1] = ITensor(mpo_arrays[1], s_out, s_in, l_right)

    for n in 2:nsite-1
        @assert dim(bond_indices[n]) == size(mpo_arrays[n], 4) == size(mpo_arrays[n+1], 3) "bond dimension mismatch at site $n"
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

"""
    arrays2mpo(site_indices::Vector{<:Index}, mpo_arrays::Vector{<:AbstractArray})

Construct an MPO from a list of arrays and site indices, automatically creating link indices.

# Arguments
- `site_indices`: Vector of site indices.
- `mpo_arrays`: Vector of arrays containing tensor data.
"""
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
    vec2tensor(v::AbstractVector)

Reshape a vector `v` into a high-dimensional tensor with dimensions given by leg_dims, suitable for TT/MPS operations.

# Arguments
- `v::AbstractVector`: Input vector to be reshaped.
"""
function vec2tensor(v::AbstractVector, leg_dims::Vector{Int})
    n = length(leg_dims)
    return permutedims(reshape(v, leg_dims...), n:-1:1)
end

"""
    vec2mps(v::AbstractVector, s::Vector{<:Index}; kwargs...)

Convert a vector `v` directly into an MPS using site indices `s`.

# Arguments
- `v::AbstractVector`: Input vector to be converted.
- `s::Vector{<:Index}`: Site indices for the MPS.
"""
function vec2mps(v::AbstractVector, s::Vector{<:Index}; kwargs...)
    site_dims = dim.(s)
    return MPS(vec2tensor(v, site_dims), s; kwargs...)
end