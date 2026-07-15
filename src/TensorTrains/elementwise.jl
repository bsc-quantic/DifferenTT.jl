using ITensors
using ITensors: ITensors, Arrow, Index, TagSets, id, plev, primestring, space
using ITensors.NDTensors: matrix, dim
using LinearAlgebra
using ITensors.NDTensors.BackendSelection: @Algorithm_str, Algorithm
using Printf

# TODO Test all of this
function ITensors.Index(
    pivs::Vector{Vector{P}}; dir::Arrow=ITensors.Out, tags="", plev=0
) where {P<:Pair}
    return Index(rand(ITensors.index_id_rng(), ITensors.IDType), pivs, dir, tags, plev)
end

const PivotIndex = Index{Vector{Vector{P}}} where {P<:Pair}

ITensors.NDTensors.dim(i::PivotIndex) = length(space(i))

ITensors.hasqns(i::PivotIndex) = false

haspivots(i::PivotIndex) = true
haspivots(i::Index) = false

function Base.show(io::IO, i::PivotIndex)
    idstr = "$(id(i) % 1000)"
    if length(tags(i)) > 0
        print(
        io,
        "(PI|dim=$(dim(i))|id=$(idstr)|\"$(TagSets.tagstring(tags(i)))\")$(primestring(plev(i)))",
        )
    else
        print(io, "(PI|dim=$(dim(i))|id=$(idstr))$(primestring(plev(i)))")
    end
end


function eye(Elt, R::Int, C::Int)
    M = zeros(Elt, R, C)
    for j in 1:min(R, C)
        M[j, j] = 1.0
    end
    return M
end

eye(R::Int, C::Int) = eye(Float64, R, C)

"""
    backsolveL(L::AbstractMatrix)

Compute the inverse of a lower-triangular matrix L whose
diagonal entries are all equal to 1.0 using a stable
back-solving algorithm.
"""
function backsolveL(L::AbstractMatrix)
    N = size(L, 1)
    (size(L, 2) == N) || error("backsolveL and backsolveU only supported for square matrices")
    iL = eye(eltype(L), N, N)
    for j in 1:(N - 1), i in (j + 1):N
        iL[i, j] = -L[i, j]
        for m in (j + 1):(i - 1)
            iL[i, j] -= L[i, m] * iL[m, j]
        end
    end
    return iL
end

"""
    backsolveU(U::AbstractMatrix)

Compute the inverse of an upper-triangular matrix U whose
diagonal entries are all equal to 1.0 using a stable
back-solving algorithm.
"""
backsolveU(U::Matrix) = transpose(backsolveL(transpose(U)))

"""
    prrldu(M::Matrix; cutoff, maxdim)

Compute the pivoted, rank-revealing LDU decomposition of an
arbitrary matrix M (M can be non-invertible and/or rectangular).
Returns L,d,U and permutations pr and pc such that
L and U are lower- and upper-triangular matrices with diagonal
values equal to 1 and L[pr,:]*Diagonal(d)*U[:,pc] ≈ M. The entries of
d decrease in absolute value such that 
norm(L[pr,:]*Diagonal(d)*U[:,pc]-M,Inf) <= abs(d[k])
where k=length(d). (Note that this inequality uses the infinity norm.)
The value of k is determined dynamically such that `k <= maxdim`
and `abs(d[k]) < cutoff`.
"""
function prrldu(M_::Matrix; cutoff::Real=0.0, maxdim::Int=typemax(Int), mindim::Int=1)
    @assert maxdim > 0
    @assert mindim > 0
    mindim = min(maxdim, mindim)
    Elt = eltype(M_)
    M = copy(M_)
    Nr, Nc = size(M)
    k = min(Nr, Nc)

    # Determine pivots
    rps = collect(1:Nr)
    cps = collect(1:Nc)

    inf_error = 0.0
    for s in 1:k
        Mabs_max, piv = findmax(abs, M)
        if Mabs_max < cutoff
            inf_error = Mabs_max
            break
        end
        Base.swaprows!(M, 1, piv[1])
        Base.swapcols!(M, 1, piv[2])
        M = M[2:end, 2:end] - M[2:end, 1] * transpose(M[1, 2:end]) / M[1, 1] # Schur complement
        rps[s], rps[piv[1] + s - 1] = rps[piv[1] + s - 1], rps[s]
        cps[s], cps[piv[2] + s - 1] = cps[piv[2] + s - 1], cps[s]
    end
    M = M_[rps, cps] # Pivoted matrix

    # Once pivots are found by successive Schur complement minimization, LDU are built
    L = eye(Elt, Nr, k)
    d = zeros(Elt, k)
    U = eye(Elt, k, Nc)
    rank = 0
    for s in 1:min(k, maxdim)
        P = M[s, s]
        d[s] = P

        if rank < mindim
        # then proceed
        elseif (iszero(P) || (abs(P) < cutoff && rank + 1 > mindim))
            break
        end
        iszero(P) && (P = one(Elt))
        rank += 1

        piv_col = M[(s + 1):end, s]
        L[(s + 1):end, s] = piv_col / P

        piv_row = M[s, (s + 1):end]
        U[s, (s + 1):end] = piv_row / P

        if s < k
            M[(s + 1):end, (s + 1):end] = M[(s + 1):end, (s + 1):end] - piv_col * transpose(piv_row) / P
        end
    end
    L = L[:, 1:rank]
    d = d[1:rank]
    U = U[1:rank, :]

    return L, d, U, invperm(rps), invperm(cps), inf_error
end

"""
    interpolative(M::Matrix; cutoff, maxdim)

Returns `C, Z, piv_cols, inf_error` where
C and Z are matrices such that `C*Z ≈ M`.
The matrix C consists of columns of M, and
which column is given by the integer entries of
the array `piv_cols`. The number of columns of
C is controlled by the approximate (SVD) rank of M,
which is controlled by the parameters `cutoff`
and `maxdim`.

"""
function interpolative(::Algorithm"prrldu", M::Matrix; cols=nothing, kws...)
    if !isnothing(cols)
        error("user-specified cols not yet supported for prrldu algorithm")
    end
    # Compute interpolative decomposition (ID) from PRRLDU
    L, d, U, pr, pc, inf_error = prrldu(M; kws...)
    U11 = U[:, 1:length(d)]
    iU11 = backsolveU(U11) # Inverses of a triangular matrix are easy and stable to compute
    ZjJ = iU11 * U
    CIj = L * LinearAlgebra.Diagonal(d) * U11
    C = CIj[pr, :] #Revert pivoting
    Z = ZjJ[:, pc]
    # Compute mapping of pivot columns to column indices
    piv_cols = invperm(pc)[1:length(d)]
    return C, Z, piv_cols, inf_error
end

function interpolative(
  ::Algorithm"nuclear", M::Matrix; cols=nothing, cutoff=0.0, maxdim=min(size(M)...), kws...
)
    error = Inf
    use_pinv = get(kws, :use_pinv, false)
    if isnothing(cols)
        maxdim = min(maxdim, size(M)...)
        cols = Int[]
        K = M' * M
        m = size(K, 1)
        Kt = K
        error = 0.0
        #num_below_cutoff = 0
        for t in 1:maxdim
        Kt2 = Kt * Kt
        l = argmax(p -> (p ∈ cols ? -Inf : Kt2[p, p] / Kt[p, p]), 1:m)
        max_err2 = Kt2[l, l] / Kt[l, l]
        #@show max_err2 
        #@show sqrt(max_err2)
        #(max_err2 < eps(eltype(M)) || num_below_cutoff >= 2) && break
        error = sqrt(abs(max_err2))
        if (error < cutoff) 
            break
            #num_below_cutoff += 1
        end
        push!(cols, l)
        Kt = K - K[:, cols] * (K[cols, cols] \ K[cols, :]) #Schur complement
        end
    end

    C = M[:, cols]
    X = use_pinv ? pinv(C) * M : C \ M
    for w in 1:length(cols), r in 1:length(cols)
        X[r, cols[w]] = (r == w) ? 1.0 : 0.0
    end
    return C, X, cols, error
end

"""
Compute the interpolative decomposition M≈C*Z of a
matrix M such that C consists of columns of M and
Z is an "interpolating matrix".

Optional keyword arguments:
* cols::Int[] (default: nothing) - array of columns to use for C. Other column or rank selection parameters (such as cutoff or maxdim) will be ignored.
* maxdim::Int - maximum number of columns to keep in factorization
* mindim::Int - minimum number of columns to keep in factorization
* cutoff::Float64 - keep only as many columns such that the value of the infinity (max) norm difference from the original tensor is below this value
* tags="Link" - tags to use for the Index connecting `C` to `Z`
"""
function interpolative(M::Matrix; algorithm="prrldu", kws...)
    return interpolative(Algorithm(algorithm), M; kws...)
end

"""
    interpolative(M::ITensor, col_inds; kws...)

Compute the interpolative decomposition of an ITensor, treated as a 
matrix with column indices given by the collection `col_inds`. 

Return a tuple of the following:
* C - ITensor containing specific columns of `M`. Indices are the
      row indices of `M` and index connecting to `Z`.
* Z - "interpolating" ITensor having `col_inds` and such that `C*Z ≈ M`
* error - estimate of error between `C*Z` and `M`

Internally uses the pivoted, rank-revealing LDU matrix decomposition.

Optional keyword arguments:
* maxdim::Int - maximum number of columns to keep in factorization
* mindim::Int - minimum number of columns to keep in factorization
* cutoff::Float64 - keep only as many columns such that the value of the infinity (max) norm difference from the original tensor is below this value
* tags="Link" - tags to use for the Index connecting `C` to `Z`
"""
function interpolative(
    T::ITensor, col_inds, local_vert; pivot_index::Union{Nothing,Index}=nothing, tags="Link", kws...
  )
    cols = nothing
    if !isnothing(pivot_index)
        pivots = space(pivot_index)
        # Remove pivot data involving the current vertex (local_vert)
        local_pivots = [only(filter(p->(p.first == local_vert),piv_set)).second for piv_set in pivots]
        other_pivots = [filter(p->(p.first != local_vert),piv_set) for piv_set in pivots]
        if isempty(other_pivots) || isempty(first(other_pivots))
            # then we only have a single row index - the local one
            cols = local_pivots
        else
        @assert length(col_inds) == 2
        cols = zeros(Int,length(local_pivots))
        for j = 1:length(local_pivots)
            l = local_pivots[j]
            if haspivots(col_inds[1])
                o = findfirst(==(other_pivots[j]),space(col_inds[1]))
                cols[j] = o + dim(col_inds[1])*(l-1)
            else
                o = findfirst(==(other_pivots[j]),space(col_inds[2]))
                cols[j] = l + dim(col_inds[1])*(o-1)
            end
        end
        end
    end

    # Matricize T
    row_inds = setdiff(inds(T), col_inds)
    Cmb_row, Cmb_col = combiner(row_inds), combiner(col_inds)
    cr, cc = combinedind(Cmb_row), combinedind(Cmb_col)
    T_matrix = matrix(Cmb_row * T * Cmb_col, cr, cc)

    # Interpolative decomp of t matrix
    if isnothing(cols)
        C_matrix, Z_matrix, piv_cols, error = interpolative(T_matrix; cols, kws...)
    else
        # Must use nuclear algorithm for now if cols != nothing
        C_matrix, Z_matrix, piv_cols, error = interpolative(T_matrix; algorithm="nuclear", cols, kws...)
    end 

    rank = length(piv_cols)

    # Compute mapping of pivot columns to column indices
    col_ranges = [1:dim(i) for i in col_inds]
    col_pivs = [zeros(Int, length(col_ranges)) for c in 1:rank]
    for (col, vals) in enumerate(Iterators.product(col_ranges...))
        loc = findfirst(==(col), piv_cols)
        if !isnothing(loc)
            @assert length(vals) == length(col_inds)
            col_pivs[loc] = collect(vals)
        end
    end

    # Make connecting index "b" with pivot info
    pivot_eltype = Pair{typeof(local_vert),Int}
    pivs = Vector{Vector{pivot_eltype}}(undef, rank)
    function get_pivs(r, c)
        i, ip = col_inds[c], col_pivs[r][c]
        return haspivots(i) ? space(i)[ip] : [local_vert => ip]
    end
    pivs = [vcat([get_pivs(r, c) for c in 1:length(col_inds)]...) for r in 1:rank]

    # TODO: debugging code to check if reconstructed pivot index matches
    check_pivot_index = Index(pivs; tags)
    @assert dim(check_pivot_index) == rank
    if isnothing(pivot_index)
        pivot_index = check_pivot_index
    else
        if space(pivot_index) != space(check_pivot_index)
            @show space(pivot_index)
            @show space(check_pivot_index)
            error("reconstructed pivot index space does not matched user provided pivot index space")
        end
    end
  #end

  # Make ITensors from C and Z matrices
    C = ITensor(C_matrix, cr, pivot_index) * dag(Cmb_row)
    Z = ITensor(Z_matrix, pivot_index, cc) * dag(Cmb_col)

    return C, Z, error
end

function sketch(M::AbstractMPS, region_verts, k::Index)
    (length(region_verts) == 0) && return ITensor(1.)
    sites = siteinds(M)
    S = ITensor()
    for p=1:dim(k)
        P = prod([random_itensor(sites[v])*M[v] for v in reverse(region_verts)])
        S += P*onehot(k=>p)
    end
    return S
end

# 1. Define the core method that unpacks the ITensor into an NDTensors.Tensor
function Random.rand!(rng::AbstractRNG, T::ITensor)
    # Mutate the underlying data container directly using Julia's base rand!
    rand!(rng, ITensors.tensor(T)) 
    return T
end

# 2. Define the convenience wrapper matching the default Julia syntax
function Random.rand!(T::ITensor)
    return rand!(Random.default_rng(), T)
end

function uniform_itensor(inds::Vararg{Index})
    tens = ITensor(undef, inds)
    rand!(tens)
    return tens
end

function positive_sketch(M::AbstractMPS, region_verts, k::Index)
    (length(region_verts) == 0) && return ITensor(1.)
    sites = siteinds(M)
    S = ITensor()
    for p=1:dim(k)
        P = prod([uniform_itensor(sites[v])*M[v] for v in reverse(region_verts)])
        S += P*onehot(k=>p)
    end
    return S
end

"""
    tt_mul(mps1::MPS, mps2::MPS)
    tt_mul(mpo1::MPO, mpo2::MPO)
    tt_mul(mps::MPS, mpo::MPO)
    tt_mul(mpo::MPO, mps::MPS)

Element-wise (Hadamard) multiplication of tensor-train objects.

Supported combinations are MPS-MPS, MPO-MPO, and mixed MPS-MPO/MPO-MPS products.
Inputs must share the same physical site indexing and array backend.

# Keywords
- `cutoff::Float64=1e-15`: Truncation cutoff used in compression steps.
- `maxdim::Int=typemax(Int64)`: Maximum allowed bond dimension.

# Returns
- `MPS` for MPS-MPS products.
- `MPO` for MPO-MPO and mixed MPS-MPO/MPO-MPS products.
"""

function _direct_tt_mul(mps1::MPS, mps2::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    @assert siteinds(mps1) == siteinds(mps2) "Site indices between the two MPS must match"
    @assert array_backend(mps1) == array_backend(mps2) "Array backend mismatch between multiplied MPSs"

    array_type = array_backend(mps1)

    mpo_tensors = ITensor[] # tensors for the mpo resulting of mps1*deltas
    out_sites = siteinds(mps1)

    new_sites = sim(out_sites) # new sites to be substituted into the mpss in order to perform the product and have out_sites in the output mps

    nsite = length(mps1)

    for i in 1:nsite
        t1 = mps1[i]
        new_site = new_sites[i]
        out_site = out_sites[i]
        
        t1 = replaceind(t1, out_site, new_site)
        delta_tensor = adapt(array_type, delta(new_site, out_site, out_site'))

        push!(mpo_tensors, t1 * delta_tensor)
    end

    mpsdelta_mpo = MPO(mpo_tensors)
    output_mps = apply(mpsdelta_mpo, mps2; cutoff=cutoff, maxdim=maxdim)
    return output_mps
end

function _rsi_tt_mul(mps1::MPS, mps2::MPS; cutoff=0.0, nsite=2, maxdim=typemax(Int), oversample=15, outputlevel=0, kws...)
  nsite ≥ 2 || error("nsite must be ≥ 2 in multiply")
  mps1, mps2 = copy(mps1), copy(mps2)

  siteinds(mps1) != siteinds(mps2) && error("mps sites must match")

  N = length(mps1)
  sites = siteinds(mps1)

  max_χ = max(maxlinkdim(mps1),maxlinkdim(mps2))
  nsketch = oversample+max_χ

  G = copy(mps1)
  x = nothing

  ranks = Int[]
  for j=1:(N-nsite+1)

    region = j:(j+nsite-1)
    v_first = region[1]
    v_second = region[2]
    s_first = sites[v_first]

    #
    # Sketch mps1 and mps2
    #
    k = Index(nsketch,"sketch,j=$j")

    S1 = sketch(mps1, last(region)+1:N, k)
    T1 = prod([mps1[v] for v in region])

    S2 = sketch(mps2, last(region)+1:N, k)
    T2 = prod([mps2[v] for v in region])

    #
    # Hadamard product of sketched functions
    #
    H = hadamard_product(T1*S1,T2*S2)

    #
    # Compute pivots of G and interpolating matrix X
    #
    row_inds = isnothing(x) ? [s_first] : [x,s_first] 
    R, X, err = interpolative(H,row_inds, v_first; maxdim, cutoff)
    x = commonind(X,R)
    G[v_first] = X
    push!(ranks,dim(x))

    # Some finalizer code
    if last(region) == N
        for v in region[2:end-1]
            row_inds = [x,only(sites[v])]
            R, X, err = interpolative(R,row_inds, v; maxdim, cutoff)
            x = commonind(X,R)
            push!(ranks,dim(x))
            G[v] = X
        end
        G[end] = R
        break
    end

    #
    # Re-decompose mps1 and mps2 on pivots of g
    #
    T1 = mps1[v_first] * mps1[v_second]
    R1, X1, _ = interpolative(T1,row_inds,v_first; pivot_index=x, maxdim, cutoff)
    mps1[v_first] = X1
    mps1[v_second] = R1

    T2 = mps2[v_first] * mps2[v_second]
    R2, X2, _ = interpolative(T2,row_inds,v_first; pivot_index=x, maxdim, cutoff)
    mps2[v_first] = X2
    mps2[v_second] = R2
  end

  (outputlevel ≥ 1) && @printf("Ranks of product = %s\n",ranks)

  return G
end

# using rsi if possible -> It is less robust that direct method and can fail, so fallback is needed
function tt_mul(mps1::MPS, mps2::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    try
        return _rsi_tt_mul(mps1, mps2; cutoff=cutoff, maxdim=maxdim) 
    catch e # In case ID fails, when masking for example
        @warn "Using slow tt_mul"
        return _direct_tt_mul(mps1, mps2; cutoff=cutoff, maxdim=maxdim)
    end
end

function tt_mul(mpo1::MPO, mpo2::MPO; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    @assert siteinds(mpo1; plev=0) == siteinds(mpo2; plev=0) && siteinds(mpo1; plev=1) == siteinds(mpo2; plev=1) "Site indices between the two MPO must match"
    @assert array_backend(mpo1) == array_backend(mpo2) "Array backend mismatch between multiplied MPSs"

    array_type = array_backend(mpo1)

    out_sites = first.(siteinds(mpo1; plev=0))
    linkinds1 = linkinds(mpo1)
    linkinds2 = linkinds(mpo2)

    out_ITensors = ITensor[]

    for (i, site) in enumerate(out_sites)
        t1 = mpo1[i]
        new_site1 = sim(site)
        t1 = replaceinds(t1, [site, site'], [new_site1, new_site1'])

        t2 = mpo2[i]
        new_site2 = sim(site)
        t2 = replaceinds(t2, [site, site'], [new_site2, new_site2'])

        delta_in = adapt(array_type, delta(new_site1, new_site2, site))
        delta_out = adapt(array_type, delta(new_site1', new_site2', site'))

        out_ITensor = t1 * t2 * delta_in * delta_out
        push!(out_ITensors, out_ITensor)

        if i > 1
            left_combiner = adapt(array_type, combiner(linkinds1[i-1], linkinds2[i-1]; tags="Link,l=$(i-1)"))
            out_ITensors[i-1] *= left_combiner # Combine the right legs of the tensor to the left if i > 1
            out_ITensors[i] *= left_combiner # Combine the left legs from the current tensor
        end
    end
    full_mpo = MPO(out_ITensors)
    compressed_mpo = truncate!(full_mpo; cutoff=cutoff, maxdim=maxdim)
    return compressed_mpo
end

function tt_mul(mps::MPS, mpo::MPO; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    @assert all(siteinds(mps) .== first.(siteinds(mpo; plev=0))) "Site indices between the MPS and the MPO must match"
    @assert array_backend(mps) == array_backend(mpo) "Array backend mismatch between multiplied MPSs"

    array_type = array_backend(mps)

    mpo_tensors = ITensor[] # tensors for the mpo resulting of mps1*deltas
    out_sites = siteinds(mps)

    new_sites = sim(out_sites) # new sites to be substituted into the mpss in order to perform the product and have out_sites in the output mps

    nsite = length(mps)

    for i in 1:nsite
        t1 = mps[i]
        new_site = new_sites[i]
        out_site = out_sites[i]
        
        t1 = replaceind(t1, out_site, new_site)
        delta_tensor = adapt(array_type, delta(new_site, out_site, out_site'))

        push!(mpo_tensors, t1 * delta_tensor)
    end

    mpsdelta_mpo = MPO(mpo_tensors)
    output_mpo = apply(mpsdelta_mpo, mpo; cutoff=cutoff, maxdim=maxdim)
    return output_mpo
end

function tt_mul(mpo::MPO, mps::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    return tt_mul(mps, mpo; cutoff=cutoff, maxdim=maxdim)
end

function elementwise_function(func::Function, mps::MPS, positive_domain::Bool=true; cutoff=0.0, nsite=2, maxdim=typemax(Int), oversample=15, outputlevel=0, kws...)
    nsite ≥ 2 || error("nsite must be ≥ 2 in order to apply elementwise functions")
    mps1 = copy(mps)

    N = length(mps1)
    sites = siteinds(mps1)

    max_χ = maxlinkdim(mps1)
    nsketch = oversample+max_χ

    G = copy(mps1)
    x = nothing

    # TODO Check if the QR inverse fails due to using Uniformly sampled sketches instead of Gaussian sketches
    use_pinv = positive_domain

    ranks = Int[]
    for j=1:(N-nsite+1)

        region = j:(j+nsite-1)
        v_first = region[1]
        v_second = region[2]
        s_first = sites[v_first]

        k = Index(nsketch,"sketch,j=$j")

        S1 = positive_domain ? positive_sketch(mps1, last(region)+1:N, k) : sketch(mps1, last(region)+1:N, k)
        T1 = prod([mps1[v] for v in region])
        
        #
        # Apply the function to the sketched array
        #
        mapped = map(func, T1*S1)

        #
        # Compute pivots of G and interpolating matrix X
        #
        row_inds = isnothing(x) ? [s_first] : [x,s_first] 
        R, X, err = interpolative(mapped, row_inds, v_first; maxdim, cutoff)
        x = commonind(X,R)
        G[v_first] = X
        push!(ranks,dim(x))

        # Some finalizer code
        if last(region) == N
            for v in region[2:end-1]
                row_inds = [x,only(sites[v])]
                R, X, err = interpolative(R, row_inds, v; maxdim, cutoff)
                x = commonind(X,R)
                push!(ranks,dim(x))
                G[v] = X
            end
            G[end] = R
            break
        end

        
        T1 = mps1[v_first] * mps1[v_second]
        R1, X1, _ = interpolative(T1, row_inds, v_first; pivot_index=x, maxdim, cutoff, use_pinv=use_pinv)
        mps1[v_first] = X1
        mps1[v_second] = R1
    end

    (outputlevel ≥ 1) && @printf("Ranks of product = %s\n",ranks)
    return G
end
