"""
    PivotSpace{P<:Pair}

Space of an index whose values are labelled by pivots, as built by the cross interpolation of
[`interpolative`](@ref): entry `r` of `pivots` lists the `site => value` pairs that select the `r`-th pivot. Wrapping
the pivots in an owned type lets DifferenTT define `dim`, `hasqns` and `show` for these indices without type piracy.

# Fields

  - `pivots::Vector{Vector{P}}`: the `site => value` pairs of every pivot.
"""
struct PivotSpace{P<:Pair}
    pivots::Vector{Vector{P}}
end

Base.:(==)(a::PivotSpace, b::PivotSpace) = a.pivots == b.pivots
Base.length(s::PivotSpace) = length(s.pivots)
Base.copy(s::PivotSpace) = PivotSpace(copy(s.pivots))

"""
    PivotIndex

Alias of `Index{<:PivotSpace}`, the type of the link indices that carry the pivots of a cross ([`pivot_index`](@ref)).
"""
const PivotIndex = Index{<:PivotSpace}

"""
    pivot_index(pivs::Vector{Vector{<:Pair}}; dir=ITensors.Out, tags="", plev=0) -> Index{PivotSpace}

Return a new index of dimension `length(pivs)` that carries the pivots `pivs` in its space.
"""
function pivot_index(pivs::Vector{Vector{P}}; dir::Arrow=ITensors.Out, tags="", plev=0) where {P<:Pair}
    return Index(rand(ITensors.index_id_rng(), ITensors.IDType), PivotSpace(pivs), dir, tags, plev)
end

"""
    index_pivots(i::PivotIndex) -> Vector{Vector{<:Pair}}

Return the pivots stored in the space of the pivot index `i`.
"""
index_pivots(i::PivotIndex) = space(i).pivots

ITensors.NDTensors.dim(i::PivotIndex) = length(space(i))

ITensors.hasqns(i::PivotIndex) = false

"""
    haspivots(i::Index) -> Bool

Return whether the index `i` carries pivots, i.e. whether it is a [`PivotIndex`](@ref).
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

"""
    eye(Elt, R::Int, C::Int) -> Matrix{Elt}
    eye(R::Int, C::Int) -> Matrix{Float64}

Return the ``R \\times C`` matrix with ones on its main diagonal and zeros elsewhere, the starting point of the unit
triangular factors of [`prrldu`](@ref).
"""
function eye(Elt, R::Int, C::Int)
    M = zeros(Elt, R, C)
    for j in 1:min(R, C)
        M[j, j] = 1.0
    end
    return M
end

eye(R::Int, C::Int) = eye(Float64, R, C)

"""
    backsolveL(L::AbstractMatrix) -> Matrix

Return the inverse of the square, unit lower triangular matrix `L` by forward substitution, column by column, in
``\\mathcal{O}(n^3)`` operations. Only the strictly lower triangular entries of `L` are read: its diagonal is taken to
be 1. Throws a `DimensionMismatch` if `L` is not square.

This routine came with the implementation of the randomized sketch-and-interpolate (RSI) algorithm of Meng, Khoo, Li &
Stoudenmire (arXiv:2602.17974) that its authors shared with us by private communication, and that they agreed to have
published.
"""
function backsolveL(L::AbstractMatrix)
    N = size(L, 1)
    (size(L, 2) == N) ||
        throw(DimensionMismatch("backsolveL and backsolveU only support square matrices, got $(size(L))"))
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
    backsolveU(U::Matrix) -> Matrix

Return the inverse of the square, unit upper triangular matrix `U`, as the transpose of [`backsolveL`](@ref) of its
transpose.

This routine came with the implementation of the randomized sketch-and-interpolate (RSI) algorithm of Meng, Khoo, Li &
Stoudenmire (arXiv:2602.17974) that its authors shared with us by private communication, and that they agreed to have
published.
"""
backsolveU(U::Matrix) = transpose(backsolveL(transpose(U)))

"""
    prrldu(M::Matrix; cutoff::Real=0.0, maxdim::Int=typemax(Int), mindim::Int=1) -> (L, d, U, pr, pc, inf_error)

Return the pivoted, rank-revealing LDU decomposition ``M \\approx L D U`` of any matrix `M`, rectangular or rank
deficient, with `L[pr, :] * Diagonal(d) * U[:, pc] ≈ M`.

The pivots are chosen by complete pivoting: at every step we take the entry of largest magnitude of the current Schur
complement, and stop when it falls below `cutoff` or after ``\\min(n_r, n_c)`` steps. The approximation error is then
the Schur complement that is left, whose largest entry is returned as `inf_error` (0 if the search did not stop on
`cutoff`). We then build the factors along this pivot order, keeping at most `maxdim` pivots and at least `mindim`,
even if they are zero, and stopping at the first pivot below `cutoff` after those. Note that `cutoff` is an
absolute threshold on the entries, not a relative one, and that `inf_error` does not account for pivots dropped by
`maxdim`.

# Returns

  - `L`: ``n_r \\times r`` unit lower triangular factor, in pivoted row order.
  - `d`: the ``r`` pivots.
  - `U`: ``r \\times n_c`` unit upper triangular factor, in pivoted column order.
  - `pr`, `pc`: the permutations that undo the pivoting of the rows and columns.
  - `inf_error`: the largest entry of the Schur complement where the pivot search stopped.

# Throws

  - `ArgumentError`: if `maxdim` or `mindim` is not positive.

This routine came with the implementation of the randomized sketch-and-interpolate (RSI) algorithm of Meng, Khoo, Li &
Stoudenmire (arXiv:2602.17974) that its authors shared with us by private communication, and that they agreed to have
published.
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
    interpolative(::Algorithm"prrldu", M::Matrix; cols=nothing, kws...) -> (C, Z, piv_cols, inf_error)

Return the interpolative decomposition ``M \\approx C Z`` computed from the [`prrldu`](@ref) of `M`, to which `kws`
(`cutoff`, `maxdim`, `mindim`) are passed. ``C`` holds the pivot columns `M[:, piv_cols]` and ``Z = U_{11}^{-1} U``,
where ``U_{11}`` is the leading square block of the unit upper factor, so that `Z[:, piv_cols]` is the identity. The
error is that of [`prrldu`](@ref). Prescribed columns are not supported: `cols` must be `nothing`.

This routine came with the implementation of the randomized sketch-and-interpolate (RSI) algorithm of Meng, Khoo, Li &
Stoudenmire (arXiv:2602.17974) that its authors shared with us by private communication, and that they agreed to have
published.
"""
function interpolative(::Algorithm"prrldu", M::Matrix; cols=nothing, kws...)
    if !isnothing(cols)
        throw(ArgumentError("user-specified cols not yet supported for prrldu algorithm; use algorithm=\"nuclear\""))
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

"""
    interpolative(::Algorithm"nuclear", M::Matrix; cols=nothing, cutoff=0.0, maxdim=min(size(M)...), kws...)
        -> (C, Z, cols, error)

Return the interpolative decomposition ``M \\approx C Z`` with ``C = M_{:, \\text{cols}}``. If `cols` is `nothing`,
the columns are chosen greedily on the Gram matrix ``K = M^\\top M``: at every step we add the column that most
reduces the residual ``K_t = K - K_{:,c} K_{c,c}^{-1} K_{c,:}``, measured by ``(K_t^2)_{pp} / (K_t)_{pp}``, until the
square root of that measure, returned as `error`, falls below `cutoff` or `maxdim` columns are taken. ``Z`` is the
least-squares solution of ``C Z = M``, by `C \\ M`, or by the pseudo-inverse if `use_pinv = true` is passed in `kws`,
and its pivot columns are then set to the identity exactly. With prescribed `cols`, `error` is `Inf`.

This routine came with the implementation of the randomized sketch-and-interpolate (RSI) algorithm of Meng, Khoo, Li &
Stoudenmire (arXiv:2602.17974) that its authors shared with us by private communication, and that they agreed to have
published.
"""
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
        for t in 1:maxdim
            Kt2 = Kt * Kt
            l = argmax(p -> (p ∈ cols ? -Inf : Kt2[p, p] / Kt[p, p]), 1:m)
            max_err2 = Kt2[l, l] / Kt[l, l]
            error = sqrt(abs(max_err2))
            if (error < cutoff)
                break
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
    interpolative(M::Matrix; algorithm="prrldu", kws...) -> (C, Z, piv_cols, error)

Return the interpolative decomposition ``M \\approx C Z`` of the matrix `M`, where ``C`` consists of columns of `M`
(those listed in `piv_cols`) and ``Z`` is an interpolating matrix, equal to the identity on those columns. The rank is
set by `cutoff`, `maxdim` and `mindim`. An interpolative decomposition keeps actual entries of `M` in its factor
``C``, which is what lets RSI rebuild a tensor train from sketched entries (Sec. 3.2 and Fig. 5 of Torrente et al.,
*Factorized approximate inverse preconditioners in the tensor train format*).

# Keywords

  - `algorithm="prrldu"`: `"prrldu"` (pivoted LDU, [`prrldu`](@ref)) or `"nuclear"` (greedy Gram-matrix selection,
    needed when `cols` is given).
  - `cols=nothing`: columns to use for ``C``; then the rank parameters are ignored.
  - `cutoff`, `maxdim`, `mindim`: rank selection; see the method of each algorithm.

This routine came with the implementation of the randomized sketch-and-interpolate (RSI) algorithm of Meng, Khoo, Li &
Stoudenmire (arXiv:2602.17974) that its authors shared with us by private communication, and that they agreed to have
published.
"""
function interpolative(M::Matrix; algorithm="prrldu", kws...)
    return interpolative(Algorithm(algorithm), M; kws...)
end

"""
    interpolative(T::ITensor, col_inds, local_vert; pivot_index::Union{Nothing,Index}=nothing, tags="Link", kws...)
        -> (C, Z, error)

Return the interpolative decomposition ``T \\approx C Z`` of the ITensor `T`, seen as a matrix whose columns are the
indices `col_inds` and whose rows are the others. ``C`` holds the pivot columns of `T`, with the row indices of `T`,
and ``Z`` interpolates them, with the indices `col_inds`; the two share a new [`PivotIndex`](@ref) that records, for
every pivot, the values of `col_inds` that select it, with the values of the vertex `local_vert` written as
`local_vert => value` and pivot indices among `col_inds` expanded into their own pivots. This is how the crosses of
RSI keep track of the multi-indices behind each bond. The pivot columns are chosen by the [`prrldu`](@ref) algorithm,
with `kws` (`cutoff`, `maxdim`, `mindim`); if `pivot_index` is given, its pivots fix the columns instead, which are
then fitted by the `"nuclear"` algorithm, and the pivots are checked to be those of `pivot_index`.

# Throws

  - `ArgumentError`: if the pivots of `pivot_index` do not match the columns they select.

This routine came with the implementation of the randomized sketch-and-interpolate (RSI) algorithm of Meng, Khoo, Li &
Stoudenmire (arXiv:2602.17974) that its authors shared with us by private communication, and that they agreed to have
published.
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

    # Check that the reconstructed pivot index matches the one passed by the caller, if any
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

    # Make ITensors from C and Z matrices
    C = ITensor(C_matrix, cr, pivot_index) * dag(Cmb_row)
    Z = ITensor(Z_matrix, pivot_index, cc) * dag(Cmb_col)

    return C, Z, error
end

"""
    sketch(M::AbstractMPS, region_verts, k::Index; rng::AbstractRNG=Random.default_rng()) -> ITensor
    positive_sketch(M::AbstractMPS, region_verts, k::Index; rng::AbstractRNG=Random.default_rng()) -> ITensor

Return the random sketch of the sites `region_verts` of `M`: for each of the `dim(k)` values of the sketch index `k`,
we contract every site of the region with an independent random vector, Gaussian for `sketch` and uniform in
``[0, 1)`` for `positive_sketch`, so that the sketch of an entrywise-positive tensor train stays positive. The result
keeps the link indices at the ends of the region and `k`; an empty region gives the scalar 1. Each region costs
``\\mathcal{O}(|\\text{region}| \\, k \\, d \\chi^2)``. The crosses use [`suffix_sketches`](@ref) instead, which
builds the sketches of all suffixes in one pass; these single-region versions stay as the references the tests compare
it with.
"""
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
    suffix_sketches(M::AbstractMPS, k::Index; positive::Bool=false, rng::AbstractRNG=Random.default_rng())
        -> Vector{ITensor}

Return the random sketches of every suffix of `M`, accumulated in a single right-to-left pass: `S[v]` is the sketch of
the sites `v:length(M)`, distributed as [`sketch`](@ref)`(M, v:length(M), k)` builds it, or as [`positive_sketch`](@ref)
with `positive = true` (uniform, positive sketch vectors instead of Gaussian ones), and `S[length(M)+1]` is the empty,
scalar sketch. Building every suffix on its own would cost ``\\mathcal{O}(N^2 k d \\chi^2)``; extending one running
contraction per sketch column costs ``\\mathcal{O}(N k d \\chi^2)``. Note that the sketches of different suffixes share
their random vectors, which a sketch built per suffix would not. Both crosses, of [`_rsi_tt_mul`](@ref) and of
[`elementwise_function`](@ref), use it: they only rewrite tensors at or to the left of their current region, so the
suffixes they need are always those of the untouched input.
"""
function suffix_sketches(M::AbstractMPS, k::Index; positive::Bool=false, rng::AbstractRNG=Random.default_rng())
    N = length(M)
    sites = siteinds(M)
    S = Vector{ITensor}(undef, N + 1)
    S[N+1] = ITensor(1.0)
    # One running partial contraction per sketch column, extended site by site.
    cols = [ITensor(1.0) for _ in 1:dim(k)]
    for v in N:-1:1
        Sv = ITensor()
        for p in 1:dim(k)
            r = positive ? uniform_itensor(rng, sites[v]) : random_itensor(rng, sites[v])
            cols[p] = r * M[v] * cols[p]
            Sv += cols[p] * onehot(k => p)
        end
        S[v] = Sv
    end
    return S
end

"""
    pivot_cutoff(T::ITensor, rtol::Real) -> Float64

Return the pivot threshold `max(rtol * maximum(abs, T), floatmin(Float64))` for the interpolative decomposition of
`T`, relative to its largest entry. [`prrldu`](@ref) compares pivots with an absolute threshold and divides by every
pivot it keeps, so an absolute cutoff behaves differently on operands of different magnitude (`p*dt`, RK4 increments,
a zero velocity component): it either discards every pivot, and the cross then throws, or it keeps, and inverts,
round-off pivots. The `floatmin` floor keeps the threshold strictly positive, so that the pivot search of an all-zero
block stops instead of dividing 0 by 0.
"""
function pivot_cutoff(T::ITensor, rtol::Real)
    d = ITensors.NDTensors.data(ITensors.tensor(T))
    scale = isempty(d) ? zero(real(eltype(T))) : maximum(abs, d)
    return max(rtol * scale, floatmin(Float64))
end

"""
    allfinite(M::AbstractMPS) -> Bool

Return whether every entry of every tensor of `M` is finite; [`tt_mul`](@ref) checks it to detect a failed cross.
"""
allfinite(M::AbstractMPS) = all(t -> all(isfinite, ITensors.NDTensors.data(ITensors.tensor(t))), M)

"""
    iszero_tt(M::AbstractMPS) -> Bool

Return whether one of the tensors of `M` is zero, which makes `M` zero. It costs ``\\mathcal{O}(N d \\chi^2)``, less
than a full `norm`, and detects the zero fields that the package builds explicitly ([`zeros_mps`](@ref)), but it
returns `false` for a tensor train that is zero only through cancellations between its tensors.
"""
iszero_tt(M::AbstractMPS) = any(t -> iszero(norm(t)), M)

"""
    TT_MUL_FALLBACKS

Atomic counter of the [`tt_mul`](@ref) calls that fell back from the randomized cross to the direct product; read it
with [`tt_mul_fallbacks`](@ref).
"""
const TT_MUL_FALLBACKS = Threads.Atomic{Int}(0)

"""
    tt_mul_fallbacks() -> Int

Return the number of MPS-MPS [`tt_mul`](@ref) calls that fell back from the randomized cross
([`_rsi_tt_mul`](@ref)) to the direct product ([`_direct_tt_mul`](@ref)) since the last
[`reset_tt_mul_fallbacks!`](@ref). The fallback is only logged at `@debug` level, so this counter is how a run reports
that its products did not go through the cross. The counter is shared by all threads.
"""
tt_mul_fallbacks() = TT_MUL_FALLBACKS[]

"""
    reset_tt_mul_fallbacks!() -> Nothing

Reset the counter reported by [`tt_mul_fallbacks`](@ref) to zero.
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
    _direct_tt_mul(mps1::MPS, mps2::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) -> MPS

Return the element-wise (Hadamard) product of two MPS, by applying `mps1` as a diagonal MPO to `mps2` with
`ITensorMPS.apply`, truncated to the relative Frobenius `cutoff` and `maxdim`. It is exact up to that truncation, and
its cost grows with the product ``\\chi_1 \\chi_2`` of the bond dimensions. It is the fallback of [`tt_mul`](@ref).

# Throws

  - `ArgumentError`: if the site indices or the array backends of the two MPS differ.
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

"""
    _rsi_tt_mul(mps1::MPS, mps2::MPS; nsite=2, oversample=25, target_rank::Int=0, pivot_rtol::Real=1e-10,
                outputlevel=0, rng::AbstractRNG=Random.default_rng(), kws...) -> MPS

Return the element-wise (Hadamard) product of two MPS by randomized sketch-and-interpolate (RSI): at every bond, from
left to right, we sketch the suffixes of both factors ([`suffix_sketches`](@ref)), multiply the two sketched blocks
of `nsite` sites element-wise, and take the interpolative decomposition ([`interpolative`](@ref)) of the result,
whose pivots give the next core of the product; both factors are then re-expressed on the same pivots before moving
on (Fig. 5 of Torrente et al., *Factorized approximate inverse preconditioners in the tensor train format*).
The output is not orthogonalized, and it can throw, so callers need a fallback ([`tt_mul`](@ref)). With fewer than
`nsite` sites the product of the full tensors is returned instead.

The cross is deliberately run *without* a rank cap. Capping it truncates on greedily chosen interpolation pivots
instead of on singular values, which costs two to three orders of magnitude in accuracy at a given rank, so
[`tt_mul`](@ref) caps the result once, by SVD, afterwards. The raw rank stays bounded by the sketch size of
`max(χ₁, χ₂, target_rank) + oversample` columns. This bound is also the main limitation: the product of two MPS can
need a rank up to ``\\chi_1 \\chi_2``, and when it exceeds the sketch size the result is not merely truncated but can
be far off. For two random MPS of bond dimension 16 on 16 sites, whose product has rank 256, the relative error was
45 with the default sketch, 5 with `target_rank = 64`, and at round-off level from `target_rank = 128`. On the
low-rank products of smooth fields it agrees with the direct product to the accuracy of the inputs.

# Keywords

  - `nsite=2`: number of sites decomposed at a time.
  - `oversample=25`: sketch columns on top of the largest rank. This is the accuracy knob of the uncapped cross: the
    sketch spans the face-splitting product of the two factors, which is larger than the range of their element-wise
    product, so the sketch size sets an error floor (about 1e-4 at `oversample = 5`, 1e-5 at 15 and 1e-6 at 40, for
    30 to 100 % more runtime).
  - `target_rank::Int=0`: expected rank of the product, which can exceed both input ranks. Pass the rank that the same
    product had at the previous step when it is known.
  - `pivot_rtol::Real=1e-10`: pivot threshold of the cross, relative to the block being decomposed
    ([`pivot_cutoff`](@ref)).
  - `outputlevel=0`: print the ranks of the product if at least 1.
  - `rng::AbstractRNG=Random.default_rng()`: random number generator of the Gaussian sketches.

Other keywords are ignored.

The randomized sketch-and-interpolate (RSI) algorithm is that of Meng, Khoo, Li & Stoudenmire, arXiv:2602.17974.
Our implementation started from a rough implementation that its authors shared with us by private communication, and
that they agreed to have published.
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
    nsite ≥ 2 || throw(ArgumentError("nsite must be ≥ 2 in multiply, got $nsite"))
    mps1, mps2 = copy(mps1), copy(mps2)

    siteinds(mps1) != siteinds(mps2) && throw(ArgumentError("mps sites must match"))

    N = length(mps1)
    sites = siteinds(mps1)
    # With fewer sites than one region, the sweep below would not run; the full tensors are small, so multiply them
    N < nsite && return MPS(hadamard_product(prod(mps1), prod(mps2)), sites)

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
                row_inds = [x, sites[v]]
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
    tt_mul(mps1::MPS, mps2::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64),
           rng::AbstractRNG=Random.default_rng(), rsi_kwargs...) -> MPS
    tt_mul(mpo1::MPO, mpo2::MPO; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) -> MPO
    tt_mul(mps::MPS, mpo::MPO; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) -> MPO
    tt_mul(mpo::MPO, mps::MPS; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)) -> MPO

Return the element-wise (Hadamard) product of two tensor trains on the same sites and array backend, truncated to the
relative Frobenius `cutoff` and to `maxdim`.

Element-wise products appear wherever a nonlinear or variable-coefficient term is evaluated in the TT format, such as
the advection term of the Navier–Stokes equations. Their exact bond dimension is the product ``\\chi_1 \\chi_2`` of
those of the factors, although the result usually needs far less. For two MPS we therefore use the randomized
sketch-and-interpolate (RSI) cross of [`_rsi_tt_mul`](@ref), whose raw rank is bounded by its sketch size, and
compress its non-orthogonal output once by SVD. Both steps matter: truncating inside the cross is far from optimal at
a given rank, and the raw output carries two to four times the rank that the product needs, which every later `apply`,
`sum` and time-stepper stage would pay for. If either factor has a zero tensor ([`iszero_tt`](@ref)), the zero MPS is
returned. If the cross throws or returns a non-finite result, the product is computed by [`_direct_tt_mul`](@ref)
instead, and the fallback is counted by [`tt_mul_fallbacks`](@ref).

Note the limits of this choice. The cross is accurate only if the rank of the product fits in its sketch: when it does
not, the result can be far off without any error being raised (see [`_rsi_tt_mul`](@ref), and pass `target_rank`).
Nor is it always faster: on ``2^{18}`` to ``2^{20}`` points, with smooth factors of bond dimension 5 to 10, or with
one factor of bond dimension 2 and the other of 16 to 64, the direct product was 5 to 12 times faster in our tests.

The products that involve an MPO are computed exactly and then truncated. For two MPO, every site is contracted with
delta tensors that identify the indices of both, which gives the entrywise product
``(A \\circ B)_{ij} = A_{ij} B_{ij}``. For an MPS ``\\mathbf{x}`` and an MPO ``M``, in either order, the result is
``\\mathrm{Diagonal}(\\mathbf{x}) M``, i.e. row ``i`` of ``M`` scaled by ``x_i``.

# Keywords

  - `cutoff::Float64=1e-15`: relative Frobenius cutoff of the final truncation.
  - `maxdim::Int=typemax(Int64)`: maximum bond dimension of the result.
  - `rng::AbstractRNG=Random.default_rng()` (MPS-MPS only): random number generator of the sketches.
  - `rsi_kwargs...` (MPS-MPS only): passed to [`_rsi_tt_mul`](@ref), e.g. `oversample`, `target_rank` and `pivot_rtol`.

# Throws

  - `ArgumentError` (products with an MPO): if the site indices or the array backends differ.

The randomized sketch-and-interpolate (RSI) algorithm is that of Meng, Khoo, Li & Stoudenmire, arXiv:2602.17974.
Our implementation started from a rough implementation that its authors shared with us by private communication, and
that they agreed to have published.
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
        Threads.atomic_add!(TT_MUL_FALLBACKS, 1)
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

"""
    elementwise_function(func::Function, mps::MPS, positive_domain::Bool=true; cutoff=0.0, nsite=2,
                         maxdim=typemax(Int), oversample=15, outputlevel=0, rng::AbstractRNG=Random.default_rng(),
                         kws...) -> MPS

Return an MPS approximation of the element-wise map `func.(x)` of the vector ``\\mathbf{x}`` represented by `mps`,
built by randomized sketch-and-interpolate (RSI), as in Sec. 3.2 of Torrente et al., *Factorized approximate inverse
preconditioners in the tensor train format*.

A nonlinear function cannot be applied to a tensor train core by core, and forming the dense vector costs
``\\mathcal{O}(d^N)``. RSI avoids both: at every bond, from left to right, it contracts the sites to the right of the
current block of `nsite` sites with random vectors (a sketch), which leaves a small array of size
``\\chi \\times d^{\\text{nsite}} \\times k`` on which `func` can be applied entry by entry. An interpolative
decomposition of the mapped array ([`interpolative`](@ref)) then gives the next core of the result, and `mps` is
re-expressed on the same pivots before moving on (Fig. 5). The sketches of every suffix are built once, in one
right-to-left pass, as in [`_rsi_tt_mul`](@ref) ([`suffix_sketches`](@ref)), so the sketching costs
``\\mathcal{O}(N k d \\chi^2)`` for ``k`` = `oversample` ``+ \\chi`` sketch columns, and each block then costs
``\\mathcal{O}(d^2 \\chi^2 k)`` to contract and map and ``\\mathcal{O}(d^3 \\chi k \\min(\\chi, k))`` to decompose. With
fewer than `nsite` sites, `func` is applied to the full tensor instead.

The original RSI draws Gaussian sketches, whose contraction with a positive tensor train can have entries of any sign.
For functions defined only on positive arguments, such as the ``x^{-1/2}`` that gives the rescaling ``D`` of the
FAI, we draw uniform sketches in ``[0, 1)`` instead (`positive_domain = true`), so that the sketched entries of an
entrywise-positive `mps` stay positive (Sec. 3.2). In that case the pivot columns are also fitted with a
pseudo-inverse. With `positive_domain = false` they are fitted by `C \\ M`, which throws a `SingularException` for a
part of the Gaussian draws (about half of them for `exp` of a signed function on
``2^8`` points in our tests); the result is approximate in both cases, with an accuracy set by the sketch size.

# Arguments

  - `func::Function`: scalar function applied to every entry.
  - `mps::MPS`: the vector ``\\mathbf{x}``; it must be entrywise positive if `positive_domain` is `true` and `func` is
    only defined for positive arguments.
  - `positive_domain::Bool=true`: draw uniform, positive sketches instead of Gaussian ones.

# Keywords

  - `cutoff=0.0`: absolute pivot threshold of the interpolative decompositions ([`prrldu`](@ref)).
  - `nsite=2`: number of sites decomposed at a time.
  - `maxdim=typemax(Int)`: maximum number of pivots, i.e. bond dimension, of every decomposition.
  - `oversample=15`: sketch columns on top of the bond dimension of `mps`.
  - `outputlevel=0`: print the ranks of the result if at least 1.
  - `rng::AbstractRNG=Random.default_rng()`: random number generator of the sketches.

Other keywords are ignored.

# Throws

  - `ArgumentError`: if `nsite < 2`.

# Examples

```julia
using StableRNGs
D = elementwise_function(x -> inv(sqrt(x)), diag_mps, true; cutoff=1e-12, rng=StableRNG(1))
```

The randomized sketch-and-interpolate (RSI) algorithm is that of Meng, Khoo, Li & Stoudenmire, arXiv:2602.17974.
Our implementation started from a rough implementation that its authors shared with us by private communication, and
that they agreed to have published.
"""
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
    nsite ≥ 2 || throw(ArgumentError("nsite must be ≥ 2 in order to apply elementwise functions, got $nsite"))
    mps1 = copy(mps)

    N = length(mps1)
    sites = siteinds(mps1)
    # With fewer sites than one region, the sweep below would not run; the full tensor is small, so map it directly
    N < nsite && return MPS(map(func, prod(mps1)), sites; maxdim)

    max_χ = maxlinkdim(mps1)
    nsketch = oversample+max_χ

    G = copy(mps1)
    x = nothing

    # TODO(release): find out why the least-squares fit fails (SingularException) for about half of the Gaussian
    # sketches, and whether the pseudo-inverse used with the uniform sketches would fix it
    use_pinv = positive_domain

    # One sketch index for the whole sweep, and one right-to-left pass to build every suffix sketch of the input, as
    # in `_rsi_tt_mul`: the sweep only rewrites sites at or to the left of `last(region)`, so the suffix
    # `last(region)+1:N` it sketches at every bond is always a suffix of the untouched input.
    k = Index(nsketch, "sketch")
    sketches = suffix_sketches(mps1, k; positive=positive_domain, rng)

    ranks = Int[]
    for j in 1:(N-nsite+1)
        region = j:(j+nsite-1)
        v_first = region[1]
        v_second = region[2]
        s_first = sites[v_first]

        S1 = sketches[last(region)+1]
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
                row_inds = [x, sites[v]]
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
