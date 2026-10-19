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

function make_pass_identity_site(site::Index, left_link::Index, right_link::Index, pass_dim::Int, array_type)
    T = ITensor(Float64, site, site', left_link, right_link)
    for s in 1:dim(site)
        for d in 1:pass_dim
            T[site=>s, site'=>s, left_link=>d, right_link=>d] = 1.0
        end
    end
    return adapt(array_type, T)
end

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

# TODO Test this function
# TODO Should we bring to zero all the tensors, as done now, or only do 0.0 * mpo (sets to zero the ortho center)
"""
    zeros_mpo(sites::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array)

Construct an all-zeros MPO  of bond dimension 1 with the given site indices and bond dimension.
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
    site_inds = site_indices(site_dims)
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

"""
    ones_mps(site_indices::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)

Construct an MPS whose every tensor entry is 1.0, with uniform bond dimension `bond_dim`.
"""
function ones_mps(site_inds::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    bond_inds = link_indices(fill(bond_dim, length(site_inds) - 1))
    return ones_mps(site_inds, bond_inds, array_type)
end

ones_mps(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) = ones_mps(site_inds, 1, array_type)

# TODO This assumes physical legs to always have dimension 2!
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

# TODO Test these functions
"""
    ones_mpo(site_indices::Vector{<:Index}, bond_dims::Vector{<:Int}, array_type::Type{<:AbstractArray}=Array)

Construct an MPO whose every tensor entry is 1.0, with per-bond dimensions given by `bond_dims`.
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

"""
    ones_mpo(site_indices::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)

Construct an MPS whose every tensor entry is 1.0, with uniform bond dimension `bond_dim`.
"""
function ones_mpo(site_inds::Vector{<:Index}, bond_dim::Int, array_type::Type{<:AbstractArray}=Array)
    bond_inds = link_indices(fill(bond_dim, length(site_inds) - 1))
    return ones_mpo(site_inds, bond_inds, array_type)
end

ones_mpo(site_inds::Vector{<:Index}, array_type::Type{<:AbstractArray}=Array) = ones_mpo(site_inds, 1, array_type)

# TODO This assumes physical legs to always have dimension 2!
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

function lower_triang_mpo(sites::Vector{<:Index}, include_diag::Bool=true; array_type::Type{<:AbstractArray}=Array)
    return transpose_mpo(upper_triang_mpo(sites, include_diag; array_type=array_type))
end

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
    arrays2mps(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mps_arrays::Vector{<:AbstractArray})

Construct an MPS from a list of arrays, using provided site and bond indices.

# Arguments

  - `site_indices`: Vector of site indices for the MPS.
  - `bond_indices`: Vector of bond indices (links) between MPS tensors.
  - `mps_arrays`: Vector of arrays containing the tensor data.
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

"""
    arrays2mps(site_indices::Vector{<:Index}, mps_arrays::Vector{<:AbstractArray})

Construct an MPS from a list of arrays and site indices, automatically creating link indices.

# Arguments

  - `site_indices`: Vector of site indices.
  - `mps_arrays`: Vector of arrays containing tensor data.
"""
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
    arrays2mpo(site_indices::Vector{<:Index}, bond_indices::Vector{<:Index}, mpo_arrays::Vector{<:AbstractArray})

Construct an MPO from a list of arrays, using provided site and bond indices.

# Arguments

  - `site_indices`: Vector of site indices (will be primed for output, unprimed for input).
  - `bond_indices`: Vector of bond indices (links).
  - `mpo_arrays`: Vector of arrays containing the tensor data.
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
    return MPS(vec2tensor(v, reverse(site_dims)), s; kwargs...) # vec2tensor takes the least significant leg first
end

"""
    mat2tensor(m::AbstractMatrix)

Reshape a matrix `m` into a high-dimensional tensor with dimensions given by leg_dims. It is assumed that the matrix
is square, in the sense that the the input-output leg pairs will be set to have leg_dims.

# Arguments

  - `m::AbstractMatrix`: Input matrix to be reshaped.
"""
function mat2tensor(m::AbstractMatrix, leg_dims::Vector{Int})
    n = length(leg_dims)
    return permutedims(reshape(m, leg_dims..., leg_dims...), [n:-1:1; 2n:-1:(n+1)])
end

"""
    mat2mpo(v::AbstractMatrix, s::Vector{<:Index}; kwargs...)

Convert a matrix `m` directly into an MPO using site indices `s` and `s'`.

# Arguments

  - `v::AbstractVector`: Input vector to be converted.
  - `s::Vector{<:Index}`: Input site indices for the MPO.
"""
function mat2mpo(m::AbstractMatrix, s::Vector{<:Index}; kwargs...)
    site_dims = dim.(s)
    itensor_object = itensor(mat2tensor(m, reverse(site_dims)), s'..., s...) # least significant leg first
    return MPO(itensor_object, s; kwargs...)
end
