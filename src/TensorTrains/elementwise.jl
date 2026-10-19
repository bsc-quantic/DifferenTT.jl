"""
    PivotSpace{P<:Pair}

Space of an index whose values are labelled by pivots, as built by the cross interpolation of `interpolative`: entry
`r` of `pivots` lists the `site => value` pairs that select the `r`-th pivot. Wrapping the pivots in an owned type
lets DifferenTT define `dim`, `hasqns` and `show` for these indices without type piracy.
"""
struct PivotSpace{P<:Pair}
    pivots::Vector{Vector{P}}
end

Base.:(==)(a::PivotSpace, b::PivotSpace) = a.pivots == b.pivots
Base.length(s::PivotSpace) = length(s.pivots)
Base.copy(s::PivotSpace) = PivotSpace(copy(s.pivots))

const PivotIndex = Index{<:PivotSpace}

"""
    pivot_index(pivs::Vector{Vector{<:Pair}}; dir=ITensors.Out, tags="", plev=0) -> Index{PivotSpace}

Return a new index of dimension `length(pivs)` that carries the pivots `pivs` in its space.
"""
function pivot_index(pivs::Vector{Vector{P}}; dir::Arrow=ITensors.Out, tags="", plev=0) where {P<:Pair}
    return Index(rand(ITensors.index_id_rng(), ITensors.IDType), PivotSpace(pivs), dir, tags, plev)
end

"""
Return the pivots stored in the space of the pivot index `i`.
"""
index_pivots(i::PivotIndex) = space(i).pivots

ITensors.NDTensors.dim(i::PivotIndex) = length(space(i))

ITensors.hasqns(i::PivotIndex) = false

"""
Return whether the index `i` carries pivots, i.e. whether it is an `Index{PivotSpace}`.
"""
haspivots(i::PivotIndex) = true
haspivots(i::Index) = false

function Base.show(io::IO, i::PivotIndex)
    idstr = "$(id(i) % 1000)"
    if length(tags(i)) > 0
        print(io, "(PI|dim=$(dim(i))|id=$(idstr)|\"$(TagSets.tagstring(tags(i)))\")$(primestring(plev(i)))")
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
    for j in 1:(N-1), i in (j+1):N
        iL[i, j] = -L[i, j]
        for m in (j+1):(i-1)
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
    maxdim > 0 || throw(ArgumentError("`maxdim` must be positive"))
    mindim > 0 || throw(ArgumentError("`mindim` must be positive"))
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
        rps[s], rps[piv[1]+s-1] = rps[piv[1]+s-1], rps[s]
        cps[s], cps[piv[2]+s-1] = cps[piv[2]+s-1], cps[s]
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

        piv_col = M[(s+1):end, s]
        L[(s+1):end, s] = piv_col / P

        piv_row = M[s, (s+1):end]
        U[s, (s+1):end] = piv_row / P

        if s < k
            M[(s+1):end, (s+1):end] = M[(s+1):end, (s+1):end] - piv_col * transpose(piv_row) / P
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

function interpolative(::Algorithm"nuclear", M::Matrix; cols=nothing, cutoff=0.0, maxdim=min(size(M)...), kws...)
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

  - cols::Int[] (default: nothing) - array of columns to use for C. Other column or rank selection parameters (such as cutoff or maxdim) will be ignored.
  - maxdim::Int - maximum number of columns to keep in factorization
  - mindim::Int - minimum number of columns to keep in factorization
  - cutoff::Float64 - keep only as many columns such that the value of the infinity (max) norm difference from the original tensor is below this value
  - tags="Link" - tags to use for the Index connecting `C` to `Z`
"""
function interpolative(M::Matrix; algorithm="prrldu", kws...)
    return interpolative(Algorithm(algorithm), M; kws...)
end

"""
    interpolative(M::ITensor, col_inds; kws...)

Compute the interpolative decomposition of an ITensor, treated as a
matrix with column indices given by the collection `col_inds`.

Return a tuple of the following:

  - C - ITensor containing specific columns of `M`. Indices are the
    row indices of `M` and index connecting to `Z`.
  - Z - "interpolating" ITensor having `col_inds` and such that `C*Z ≈ M`
  - error - estimate of error between `C*Z` and `M`

Internally uses the pivoted, rank-revealing LDU matrix decomposition.

Optional keyword arguments:

  - maxdim::Int - maximum number of columns to keep in factorization
  - mindim::Int - minimum number of columns to keep in factorization
  - cutoff::Float64 - keep only as many columns such that the value of the infinity (max) norm difference from the original tensor is below this value
  - tags="Link" - tags to use for the Index connecting `C` to `Z`
"""
function interpolative(T::ITensor, col_inds, local_vert; pivot_index::Union{Nothing,Index}=nothing, tags="Link", kws...)
    cols = nothing
    if !isnothing(pivot_index)
        pivots = index_pivots(pivot_index)
        # Remove pivot data involving the current vertex (local_vert)
        local_pivots = [only(filter(p->(p.first == local_vert), piv_set)).second for piv_set in pivots]
        other_pivots = [filter(p->(p.first != local_vert), piv_set) for piv_set in pivots]
        if isempty(other_pivots) || isempty(first(other_pivots))
            # then we only have a single row index - the local one
            cols = local_pivots
        else
            @assert length(col_inds) == 2
            cols = zeros(Int, length(local_pivots))
            for j in 1:length(local_pivots)
                l = local_pivots[j]
                if haspivots(col_inds[1])
                    o = findfirst(==(other_pivots[j]), index_pivots(col_inds[1]))
                    cols[j] = o + dim(col_inds[1])*(l-1)
                else
                    o = findfirst(==(other_pivots[j]), index_pivots(col_inds[2]))
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
        return haspivots(i) ? index_pivots(i)[ip] : [local_vert => ip]
    end
    pivs = [vcat([get_pivs(r, c) for c in 1:length(col_inds)]...) for r in 1:rank]

    # TODO: debugging code to check if reconstructed pivot index matches
    check_pivot_index = DifferenTT.pivot_index(pivs; tags) # qualified: the keyword `pivot_index` shadows it here
    @assert dim(check_pivot_index) == rank
    if isnothing(pivot_index)
        pivot_index = check_pivot_index
    else
        if index_pivots(pivot_index) != index_pivots(check_pivot_index)
            throw(
                ArgumentError(
                    "reconstructed pivot index space does not match the user-provided pivot index space: " *
                    "got $(index_pivots(check_pivot_index)), expected $(index_pivots(pivot_index))",
                ),
            )
        end
    end
    #end

    # Make ITensors from C and Z matrices
    C = ITensor(C_matrix, cr, pivot_index) * dag(Cmb_row)
    Z = ITensor(Z_matrix, pivot_index, cc) * dag(Cmb_col)

    return C, Z, error
end

function sketch(M::AbstractMPS, region_verts, k::Index; rng::AbstractRNG=Random.default_rng())
    (length(region_verts) == 0) && return ITensor(1.0)
    sites = siteinds(M)
    S = ITensor()
    for p in 1:dim(k)
        P = prod([random_itensor(rng, sites[v])*M[v] for v in reverse(region_verts)])
        S += P*onehot(k=>p)
    end
    return S
end

"""
    suffix_sketches(M::AbstractMPS, k::Index; rng=Random.default_rng())

Random sketches of every suffix of `M`, accumulated in a single right-to-left pass.

`S[v]` is the sketch of the sites `v:length(M)`, i.e. what `sketch(M, v:length(M), k)` builds,
and `S[length(M)+1]` is the empty (scalar) sketch. Building each suffix independently costs
`O(N^2 k χ^2)`, while accumulating them once costs `O(N k χ^2)` with the same (Gaussian) sketch
distribution. The cross in `_rsi_tt_mul` only ever rewrites tensors at or to the left of its
current region, so the suffixes it needs are always the ones of the untouched input MPS.
"""
function suffix_sketches(M::AbstractMPS, k::Index; rng::AbstractRNG=Random.default_rng())
    N = length(M)
    sites = siteinds(M)
    S = Vector{ITensor}(undef, N + 1)
    S[N+1] = ITensor(1.0)
    # One running partial contraction per sketch column, extended site by site.
    cols = [ITensor(1.0) for _ in 1:dim(k)]
    for v in N:-1:1
        Sv = ITensor()
        for p in 1:dim(k)
            cols[p] = random_itensor(rng, sites[v]) * M[v] * cols[p]
            Sv += cols[p] * onehot(k => p)
        end
        S[v] = Sv
    end
    return S
end

"""
    pivot_cutoff(T::ITensor, rtol::Real)

Pivot threshold for the interpolative decomposition of `T`, relative to its largest entry.

`prrldu` compares pivots against an absolute threshold and divides by every pivot it keeps, so
an absolute cutoff behaves differently on operands of different magnitude (`p*dt`, RK4
increments, a zero velocity component): it either discards every pivot, and the cross then
throws, or it keeps - and inverts - round-off pivots. The `floatmin` floor keeps the threshold
strictly positive, so the pivot search of an all-zero block stops instead of dividing 0/0.
"""
function pivot_cutoff(T::ITensor, rtol::Real)
    d = ITensors.NDTensors.data(ITensors.tensor(T))
    scale = isempty(d) ? zero(real(eltype(T))) : maximum(abs, d)
    return max(rtol * scale, floatmin(Float64))
end

"""
    allfinite(M::AbstractMPS)

Whether every entry of every tensor of `M` is finite.
"""
allfinite(M::AbstractMPS) = all(t -> all(isfinite, ITensors.NDTensors.data(ITensors.tensor(t))), M)

"""
    iszero_tt(M::AbstractMPS)

Whether `M` is identically zero, detected from a single all-zero tensor. Cheap (`O(Nχ²d)`) and
sufficient for the zero fields the pipeline builds explicitly, unlike a full `norm`.
"""
iszero_tt(M::AbstractMPS) = any(t -> iszero(norm(t)), M)

# Number of times `tt_mul` had to fall back from the randomized cross to the direct product.
const TT_MUL_FALLBACKS = Ref(0)

"""
    tt_mul_fallbacks()

Number of `tt_mul` calls that fell back from the randomized cross (`_rsi_tt_mul`) to the direct
product since the last [`reset_tt_mul_fallbacks!`](@ref). A run that quietly switches to the slow,
`O(χ1 χ2)` path is otherwise invisible, since the fallback is only logged at `@debug` level.
"""
tt_mul_fallbacks() = TT_MUL_FALLBACKS[]

"""
    reset_tt_mul_fallbacks!()

Reset the counter reported by [`tt_mul_fallbacks`](@ref).
"""
reset_tt_mul_fallbacks!() = (TT_MUL_FALLBACKS[]=0; nothing)

"""
Fill the dense ITensor `T` in place with independent uniform draws in [0, 1) from `rng`, and return it.
"""
function _rand_uniform!(rng::AbstractRNG, T::ITensor)
    # Mutate the underlying NDTensors.Tensor directly with Base's rand!
    rand!(rng, ITensors.tensor(T))
    return T
end

"""
    uniform_itensor([rng::AbstractRNG,] inds::Index...) -> ITensor

Return a dense ITensor with indices `inds` and independent entries uniform in [0, 1), drawn from `rng`
(`Random.default_rng()` by default). Its entries are positive, which is what [`positive_sketch`](@ref) needs.
"""
uniform_itensor(rng::AbstractRNG, inds::Index...) = _rand_uniform!(rng, ITensor(undef, inds))
uniform_itensor(inds::Index...) = uniform_itensor(Random.default_rng(), inds...)

function positive_sketch(M::AbstractMPS, region_verts, k::Index; rng::AbstractRNG=Random.default_rng())
    (length(region_verts) == 0) && return ITensor(1.0)
    sites = siteinds(M)
    S = ITensor()
    for p in 1:dim(k)
        P = prod([uniform_itensor(rng, sites[v])*M[v] for v in reverse(region_verts)])
        S += P*onehot(k=>p)
    end
    return S
end

"""
    _direct_tt_mul(mps1::MPS, mps2::MPS; cutoff=1e-15, maxdim=typemax(Int))

Element-wise (Hadamard) product of two MPS, by applying `mps1` as a diagonal MPO to `mps2` and
compressing the result with the relative Frobenius `cutoff` and `maxdim`.

Exact up to that compression, but `O(χ1^3 χ2^3)`, so roughly 10x slower than the randomized cross
of [`_rsi_tt_mul`](@ref). It is the fallback of [`tt_mul`](@ref).
"""
function _direct_tt_mul(mps1::MPS, mps2::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    siteinds(mps1) == siteinds(mps2) || throw(ArgumentError("Site indices between the two MPS must match"))
    backend_type(mps1) == backend_type(mps2) || throw(ArgumentError("Array backend mismatch between multiplied MPSs"))

    array_type = backend_type(mps1)

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

#TODO Update docstring with new default oversample value of 25 instead of 40
"""
    _rsi_tt_mul(mps1::MPS, mps2::MPS; oversample=40, target_rank=0, pivot_rtol=1e-10, nsite=2)

Element-wise (Hadamard) product of two MPS by a randomized sketch followed by an interpolative
(pivoted, rank-revealing LDU) cross. About 10x faster than [`_direct_tt_mul`](@ref), but less
robust: it can throw, so callers need a fallback.

The cross is deliberately run *without* a rank cap. Capping it truncates on greedily chosen
interpolation pivots instead of on singular values, which costs two to three orders of magnitude
in accuracy at a given rank; `tt_mul` therefore caps the result once, by SVD, afterwards. The
raw rank stays bounded by the sketch size anyway.

# Keywords

  - `oversample::Int=25`: sketch columns on top of the input ranks. This is the accuracy knob of
    the uncapped cross: the sketch spans the face-splitting product of the two factors, which is
    larger than the range of their element-wise product, so the sketch size sets an error floor
    (about 1e-4 at `oversample=5`, 1e-5 at 15, 1e-6 at 40, at +30-100% runtime).
  - `target_rank::Int=0`: expected rank of the *product*, which can exceed both input ranks. The
    sketch is sized as `max(χ1, χ2, target_rank) + oversample`. Pass the rank the same product had
    at the previous step when it is known.
  - `pivot_rtol::Real=1e-10`: pivot threshold of the cross, relative to the block being
    decomposed (see [`pivot_cutoff`](@ref)).
  - `nsite::Int=2`: number of sites decomposed at a time.
  - `rng::AbstractRNG=Random.default_rng()`: random number generator of the Gaussian sketches.
"""
function _rsi_tt_mul(
    mps1::MPS,
    mps2::MPS;
    nsite=2,
    oversample=25,
    target_rank::Int=0,
    pivot_rtol::Real=1e-10,
    outputlevel=0,
    rng::AbstractRNG=Random.default_rng(),
    kws...,
)
    nsite ≥ 2 || error("nsite must be ≥ 2 in multiply")
    mps1, mps2 = copy(mps1), copy(mps2)

    siteinds(mps1) != siteinds(mps2) && error("mps sites must match")

    N = length(mps1)
    sites = siteinds(mps1)

    max_χ = max(maxlinkdim(mps1), maxlinkdim(mps2), target_rank)
    nsketch = oversample+max_χ

    # One sketch index for the whole sweep, and one right-to-left pass to build every suffix
    # sketch of both inputs, instead of rebuilding each suffix from scratch at every bond.
    k = Index(nsketch, "sketch")
    sketches1 = suffix_sketches(mps1, k; rng)
    sketches2 = suffix_sketches(mps2, k; rng)

    G = copy(mps1)
    # `copy` carries over the orthogonality limits of mps1, but none of its tensors survive.
    ITensorMPS.set_ortho_lims!(G, 1:N)
    x = nothing

    ranks = Int[]
    for j in 1:(N-nsite+1)
        region = j:(j+nsite-1)
        v_first = region[1]
        v_second = region[2]
        s_first = sites[v_first]

        #
        # Sketch mps1 and mps2
        #
        S1 = sketches1[last(region)+1]
        T1 = prod([mps1[v] for v in region])

        S2 = sketches2[last(region)+1]
        T2 = prod([mps2[v] for v in region])

        #
        # Hadamard product of sketched functions
        #
        H = hadamard_product(T1*S1, T2*S2)

        #
        # Compute pivots of G and interpolating matrix X
        #
        row_inds = isnothing(x) ? [s_first] : [x, s_first]
        R, X, err = interpolative(H, row_inds, v_first; cutoff=pivot_cutoff(H, pivot_rtol))
        x = commonind(X, R)
        G[v_first] = X
        push!(ranks, dim(x))

        # Some finalizer code
        if last(region) == N
            for v in region[2:(end-1)]
                row_inds = [x, only(sites[v])]
                R, X, err = interpolative(R, row_inds, v; cutoff=pivot_cutoff(R, pivot_rtol))
                x = commonind(X, R)
                push!(ranks, dim(x))
                G[v] = X
            end
            G[end] = R
            break
        end

        #
        # Re-decompose mps1 and mps2 on pivots of g
        # (the pivot columns are fixed here, so no cutoff or maxdim applies)
        #
        T1 = mps1[v_first] * mps1[v_second]
        R1, X1, _ = interpolative(T1, row_inds, v_first; pivot_index=x)
        mps1[v_first] = X1
        mps1[v_second] = R1

        T2 = mps2[v_first] * mps2[v_second]
        R2, X2, _ = interpolative(T2, row_inds, v_first; pivot_index=x)
        mps2[v_first] = X2
        mps2[v_second] = R2
    end

    (outputlevel ≥ 1) && @printf("Ranks of product = %s\n", ranks)

    return G
end

"""
    tt_mul(mps1::MPS, mps2::MPS; cutoff=1e-15, maxdim=typemax(Int), rng=Random.default_rng(), rsi_kwargs...)
    tt_mul(mpo1::MPO, mpo2::MPO; cutoff=1e-15, maxdim=typemax(Int))
    tt_mul(mps::MPS, mpo::MPO; cutoff=1e-15, maxdim=typemax(Int))
    tt_mul(mpo::MPO, mps::MPS; cutoff=1e-15, maxdim=typemax(Int))

Element-wise (Hadamard) multiplication of tensor-train objects.

Supported combinations are MPS-MPS, MPO-MPO, and mixed MPS-MPO/MPO-MPS products.
Inputs must share the same physical site indexing and array backend.

The MPS-MPS product is computed by the randomized cross [`_rsi_tt_mul`](@ref), which is run
uncapped and whose raw, non-orthogonal output is then compressed once by SVD. Both matter:
truncating inside the cross is far from optimal at a given rank, and the raw output carries two
to four times the rank the product needs, which every downstream `apply`, `sum` and time-stepper
stage would then pay for. `_direct_tt_mul` is used instead whenever the cross throws or returns a
non-finite result; such fallbacks are counted by [`tt_mul_fallbacks`](@ref).

# Keywords

  - `cutoff::Float64=1e-15`: relative Frobenius truncation cutoff of the final compression.
  - `maxdim::Int=typemax(Int64)`: maximum allowed bond dimension of the result.
  - `rsi_kwargs...` (MPS-MPS only): forwarded to [`_rsi_tt_mul`](@ref), e.g. `oversample`,
    `target_rank` and `pivot_rtol`.
  - `rng::AbstractRNG=Random.default_rng()` (MPS-MPS only): random number generator of the sketches.

# Returns

  - `MPS` for MPS-MPS products.
  - `MPO` for MPO-MPO and mixed MPS-MPO/MPO-MPS products.
"""
function tt_mul(
    mps1::MPS,
    mps2::MPS;
    cutoff::Float64=1e-15,
    maxdim::Int=typemax(Int64),
    rng::AbstractRNG=Random.default_rng(),
    rsi_kwargs...,
)
    # A zero operand makes the product zero, and the relative pivot cutoff of the cross carries
    # no information on it, so short-circuit instead of sketching noise.
    (iszero_tt(mps1) || iszero_tt(mps2)) && return zeros_mps(siteinds(mps1), backend_type(mps1))

    G = nothing
    try
        G = _rsi_tt_mul(mps1, mps2; rng, rsi_kwargs...)
    catch e # In case ID fails, when masking for example
        @debug "RSI tt_mul threw; using slow tt_mul" exception=e
    end

    # A non-finite result raises no exception of its own, so it has to be checked for explicitly:
    # otherwise a NaN from an inverted tiny pivot propagates into the fields silently.
    if isnothing(G) || !allfinite(G)
        isnothing(G) || @debug "RSI tt_mul returned a non-finite result; using slow tt_mul"
        TT_MUL_FALLBACKS[] += 1
        return _direct_tt_mul(mps1, mps2; cutoff=cutoff, maxdim=maxdim)
    end

    return truncate!(G; cutoff=cutoff, maxdim=maxdim)
end

function tt_mul(mpo1::MPO, mpo2::MPO; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    (siteinds(mpo1; plev=0) == siteinds(mpo2; plev=0) && siteinds(mpo1; plev=1) == siteinds(mpo2; plev=1)) ||
        throw(ArgumentError("Site indices between the two MPO must match"))
    backend_type(mpo1) == backend_type(mpo2) ||
        throw(ArgumentError("Array backend mismatch between the multiplied MPOs"))

    array_type = backend_type(mpo1)

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
    all(siteinds(mps) .== first.(siteinds(mpo; plev=0))) ||
        throw(ArgumentError("Site indices between the MPS and the MPO must match"))
    backend_type(mps) == backend_type(mpo) || throw(ArgumentError("Array backend mismatch between the MPS and the MPO"))

    array_type = backend_type(mps)

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

function elementwise_function(
    func::Function,
    mps::MPS,
    positive_domain::Bool=true;
    cutoff=0.0,
    nsite=2,
    maxdim=typemax(Int),
    oversample=15,
    outputlevel=0,
    rng::AbstractRNG=Random.default_rng(),
    kws...,
)
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
    for j in 1:(N-nsite+1)
        region = j:(j+nsite-1)
        v_first = region[1]
        v_second = region[2]
        s_first = sites[v_first]

        k = Index(nsketch, "sketch,j=$j")

        S1 = if positive_domain
            positive_sketch(mps1, (last(region)+1):N, k; rng)
        else
            sketch(mps1, (last(region)+1):N, k; rng)
        end
        T1 = prod([mps1[v] for v in region])

        #
        # Apply the function to the sketched array
        #
        mapped = map(func, T1*S1)

        #
        # Compute pivots of G and interpolating matrix X
        #
        row_inds = isnothing(x) ? [s_first] : [x, s_first]
        R, X, err = interpolative(mapped, row_inds, v_first; maxdim, cutoff)
        x = commonind(X, R)
        G[v_first] = X
        push!(ranks, dim(x))

        # Some finalizer code
        if last(region) == N
            for v in region[2:(end-1)]
                row_inds = [x, only(sites[v])]
                R, X, err = interpolative(R, row_inds, v; maxdim, cutoff)
                x = commonind(X, R)
                push!(ranks, dim(x))
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

    (outputlevel ≥ 1) && @printf("Ranks of product = %s\n", ranks)
    return G
end
