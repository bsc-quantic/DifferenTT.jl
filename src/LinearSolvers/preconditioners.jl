"""
    _regularized_inv(mat; rtol=eps(Float64)) -> Matrix

Return ``(\\texttt{mat} + \\varepsilon I)^{-1}`` with ``\\varepsilon = \\texttt{rtol} \\max(\\|\\texttt{mat}\\|_2, 1)``,
which avoids a `SingularException` when a truncation leaves a nearly singular local factor. Throws a `DimensionMismatch`
if `mat` is not square.
"""
function _regularized_inv(mat::AbstractMatrix; rtol::Float64=eps(Float64))
    n = size(mat, 1)
    size(mat, 1) == size(mat, 2) || throw(DimensionMismatch("Local rank-1 tensor must be square"))
    ε = rtol * max(opnorm(mat), 1.0)
    return inv(mat + ε * I(n))
end

"""
    rank1_preconditioner(operator::MPO) -> Preconditioner

Build the preconditioner of bond dimension 1 obtained by truncating `operator` to bond dimension 1 and inverting each
of its local factors.

An operator of bond dimension 1 is a Kronecker product ``A_1 \\otimes \\dots \\otimes A_N``, whose inverse is the
Kronecker product of the inverses of its ``d \\times d`` factors. We therefore truncate `operator` to bond dimension 1
by SVD, and invert each factor with a Tikhonov regularization ``(A_k + \\varepsilon I)^{-1}``,
``\\varepsilon = \\epsilon \\max(\\|A_k\\|_2, 1)``, at a cost of ``\\mathcal{O}(N d^3)`` after the truncation. The
result is exact, up to ``\\varepsilon``, for an operator of bond dimension 1, and a cheap but rough approximation
otherwise: for the Laplacian, whose bond dimension is 2 or 3, ``\\|I - M^{-1} A\\|_2`` is about 1.5 on a ``2^4``-point
QTT grid. Note that the truncated factors need not be SPD, so neither does the result. We use it to precondition the
construction of other preconditioners (the `preconditioning` keyword of [`fai_preconditioner`](@ref) and
[`linsolve_preconditioner`](@ref)) and as the starting guess of [`newtoniter_preconditioner`](@ref).

# Arguments

  - `operator::MPO`: operator with site indices at prime levels 0 (input) and 1 (output).

# Returns

  - `Preconditioner`: the approximate inverse, of bond dimension 1.
"""
function rank1_preconditioner(operator::MPO)
    rank1_operator = deepcopy(operator)
    truncate!(rank1_operator; maxdim=1)

    preconditioner_tensors = ITensor[]
    # TODO(release): this way of dropping the link indices of dimension 1 assumes the index order (s', s, links...)
    for tensor in rank1_operator
        mat = order(tensor) == 3 ? dropdims(array(tensor); dims=3) : dropdims(array(tensor); dims=(3, 4))
        invmat = _regularized_inv(mat)
        push!(preconditioner_tensors, ITensor(reshape(invmat, dims(tensor)), inds(tensor)...))
    end
    return Preconditioner(MPO(preconditioner_tensors))
end

"""
    newtoniter_preconditioner(operator::MPO, init_preconditioner::MPO, niter::Int; cutoff=1e-15, maxdim=typemax(Int))
        -> Preconditioner
    newtoniter_preconditioner(operator::MPO, niter::Int; cutoff=1e-15, maxdim=typemax(Int)) -> Preconditioner

Refine an approximate inverse of `operator` by `niter` Newton–Schulz iterations
``X_{k+1} = X_k (2I - A X_k)``, starting from `init_preconditioner`, or from 0.01 times
[`rank1_preconditioner`](@ref)`(operator)` in the second method.

The iteration needs no linear solve, and its residual satisfies ``I - X_{k+1} A = (I - X_k A)^2``, so it converges
quadratically once ``\\|I - X_0 A\\| < 1``, but not otherwise. The bond dimension grows with every step, since each one
multiplies three MPOs; we truncate the products and sums with `cutoff` and `maxdim`, which also limits the attainable
accuracy. From the default starting guess the convergence can be slow: on a ``2^4``-point QTT Laplacian, three steps
reduce ``\\|I - X A\\|_2`` only from 0.9998 to 0.9984. As for any truncated approximate inverse, the result is not
guaranteed to be SPD.

# Throws

  - `ArgumentError`: if `init_preconditioner` and `operator` do not have the same site indices.
"""
function newtoniter_preconditioner(
    operator::MPO, init_preconditioner::MPO, niter::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64)
)
    siteinds(operator) != siteinds(init_preconditioner) &&
        throw(ArgumentError("Index mismatch between operator and init_preconditioner"))

    preconditioner = init_preconditioner
    for _ in 1:niter
        xax = apply(preconditioner, apply(operator, preconditioner); cutoff=cutoff, maxdim=maxdim)
        preconditioner = +(2 * preconditioner, (-1.0) * xax; cutoff=cutoff, maxdim=maxdim)
    end
    return Preconditioner(preconditioner)
end

function newtoniter_preconditioner(operator::MPO, niter::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    init_preconditioner = 0.01 * rank1_preconditioner(operator).operator
    return newtoniter_preconditioner(operator, init_preconditioner, niter; cutoff=cutoff, maxdim=maxdim)
end

"""
    linsolve_preconditioner(operator::MPO, solver::AbstractLinearSolver, symmetrize::Bool=true; linsolve_x0=nothing,
                            preconditioning=false, residual_tol=nothing, rng=Random.default_rng(), kwargs...)
        -> Preconditioner

Build an approximate inverse ``X \\approx A^{-1}`` of `operator` by solving, in the TT format, the matrix equation
``AX + XA = 2I`` (`symmetrize = true`) or ``AX = I``, as in the matrix inversion of Oseledets & Dolgov (2012).

We vectorize the unknown ``X`` by fusing, at each site, its row and column indices into one index of dimension
``d^2``, which turns the matrix equation into a linear system for an MPS with ``N`` sites, and solve it with
`solver`. Both equations have the solution ``A^{-1}``; the symmetric form keeps the solution symmetric for a
symmetric ``A``. Note that the result is truncated to the bond dimension of `solver`, so it controls the norm error
but not the sign of the eigenvalues: for an SPD ``A`` it can be indefinite (Fig. 3 of Torrente et al.,
*Factorized approximate inverse preconditioners in the tensor train format*, solid lines), which is the motivation for
[`fai_preconditioner`](@ref).

# Arguments

  - `operator::MPO`: the operator ``A``.
  - `solver::AbstractLinearSolver`: solver of the vectorized system.
  - `symmetrize::Bool=true`: solve ``AX + XA = 2I`` instead of ``AX = I``.

# Keywords

  - `linsolve_x0=nothing`: initial guess for ``X``, an MPO on the sites of `operator`; by default ``10^{-12}`` times a
    random MPO of bond dimension 1, drawn from `rng`.
  - `preconditioning::Bool=false`: precondition the vectorized system with [`rank1_preconditioner`](@ref).
  - `residual_tol::Union{Float64,Nothing}=nothing`: if set, stop the sweeps once the relative residual of the
    vectorized system is below it (through a [`ResidualObserver`](@ref) used as `sweep_observer!`).
  - `rng::AbstractRNG=Random.default_rng()`: random number generator of the default initial guess.
  - `kwargs...`: passed to [`linearsolve`](@ref).

# Returns

  - `Preconditioner`: the approximate inverse ``X``.
"""
function linsolve_preconditioner(
    operator::MPO,
    solver::AbstractLinearSolver,
    symmetrize::Bool=true;
    linsolve_x0=nothing,
    preconditioning::Bool=false,
    residual_tol::Union{Float64,Nothing}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...,
)
    sites = only.(siteinds(operator; plev=0))
    helper_sites = sim(sites) # Helper sites to extend the space

    # The unknown X is vectorized on (helper_sites, sites): the helper sites carry its row index and become the output
    # (primed) sites of X at the end, and the original sites carry its column index. AX acts on the row index with A,
    # and XA, i.e. (XA)_ij = Σ_k X_ik A_kj, acts on the column index with Aᵀ.
    op1 = deepcopy(operator) # To construct AX
    replace_mpo_siteinds!(op1, helper_sites) # A, for AX, lives in the helper space

    if symmetrize
        op2 = transpose_mpo(operator) # To construct XA, on the original sites
    else
        op2 = nothing
    end

    linsolve_b = identity_mpo(sites, backend_type(operator)) # Construct the right hand side
    linsolve_b = symmetrize ? 2 * linsolve_b : linsolve_b # Construct the right hand side

    N = length(op1)
    combiners = [combiner(sites[i], helper_sites[i]; tags="csite=$i") for i in 1:N] # Combiners that join a pair of sites coming from the two spaces into a single index

    for i in 1:N
        op1[i] = op1[i] * delta(sites[i], sites[i]') # Identity on the column index for AX
        op1[i] = op1[i] * combiners[i] * combiners[i]' # Join op and identity into a single leg for both operators

        if symmetrize
            op2[i] = op2[i] * delta(helper_sites[i], helper_sites[i]') # Identity on the row index for XA
            op2[i] = op2[i] * combiners[i] * combiners[i]'
        end

        replaceind!(linsolve_b[i], sites[i]' => helper_sites[i])
        linsolve_b[i] = linsolve_b[i] * combiners[i]
    end

    linsolve_A = symmetrize ? +(op1, op2; alg="directsum") : op1  # Construct the operator AX + XA
    linsolve_b = MPS([linsolve_b...]) # Convert the still MPO object into an MPS

    if isnothing(linsolve_x0)
        _linsolve_x0 = MPS([1e-12 * adapt(backend_type(operator), random_mpo(rng, sites, 1))...])
        for i in 1:N
            replaceind!(_linsolve_x0[i], sites[i]' => helper_sites[i])
            _linsolve_x0[i] = _linsolve_x0[i] * combiners[i]
        end
    else
        _linsolve_x0 = deepcopy(linsolve_x0)
        _linsolve_x0 = MPS([_linsolve_x0...])
        for i in 1:N
            replaceind!(_linsolve_x0[i], sites[i]' => helper_sites[i])
            _linsolve_x0[i] = _linsolve_x0[i] * combiners[i]
        end
    end

    @debug "Linsolve preconditioner with preconditioning set to $preconditioning"

    ls_kwargs = Dict{Symbol,Any}(kwargs)
    if !isnothing(residual_tol)
        ls_kwargs[:sweep_observer!] = ResidualObserver(; A=linsolve_A, b=linsolve_b, rtol=residual_tol)
    end

    if preconditioning
        preconditioner = rank1_preconditioner(linsolve_A)
        preconditioner_mps = linearsolve(solver, linsolve_A, linsolve_b, preconditioner, _linsolve_x0; ls_kwargs...)
    else
        preconditioner_mps = linearsolve(solver, linsolve_A, linsolve_b, _linsolve_x0; ls_kwargs...)
    end
    for i in 1:N
        preconditioner_mps[i] = preconditioner_mps[i] * dag(combiners[i]) # Unfold the legs 
        replaceind!(preconditioner_mps[i], helper_sites[i] => sites[i]') # Map the helper legs into the original primed space
    end
    return Preconditioner(MPO([preconditioner_mps...]))
end

# TODO(release): test the densitymatrix and zipup algorithms for the products below, with maxdim and cutoff
"""
    fai_dg(operator::MPO, μ::Real, g_solver::AbstractLinearSolver, D_kwargs; preconditioning=false,
           residual_tol=nothing, compute_residual=false, rng=Random.default_rng(), kwargs...) -> (G̃_L::MPO, D::MPO)

Internal. Return the two factors of the FAI of the SPD `operator` ``A``: the lower-triangular approximate inverse
``\\tilde{G}_L`` and the diagonal rescaling ``D``, which [`fai_preconditioner`](@ref) multiplies into
``G_L = D \\tilde{G}_L``. We keep the two factors apart so that the effects of truncating ``\\tilde{G}_L`` and ``D`` can
be studied separately when prototyping; call it qualified, `DifferenTT.fai_dg`.

We vectorize ``X = \\tilde{G}_L`` by fusing, at each site, its row index (a copy of the sites, the output of
``\\tilde{G}_L``) with its column index into one index of dimension ``d^2``, and solve with `g_solver` the projected
system of Eq. (22) of Torrente et al., *Factorized approximate inverse preconditioners in the tensor train format*,

```math
\\left[\\mathbb{P}_L \\mathcal{T}_A \\mathbb{P}_L + \\mu (\\mathbb{I} - \\mathbb{P}_L)\\right]
\\mathrm{vec}(\\tilde{G}_L) = \\mathrm{vec}(\\mathbf{I}),
```

where ``\\mathcal{T}_A \\mathrm{vec}(X) = \\mathrm{vec}(XA)`` is the operator written ``\\mathbf{I} \\otimes A`` in Eq.
(19) for a symmetric ``A``, ``\\mathbb{P}_L`` = [`triang_projector_mpo`](@ref) keeps the lower-triangular entries of
``X`` (Eqs. (14)–(20)), and ``\\mu > 0`` weights the equations ``\\mu X_{ij} = 0`` of the entries above the diagonal.
Its exact solution satisfies Eq. (10), ``(\\tilde{G}_L A)_{ij} = \\delta_{ij}`` for ``i \\ge j``, and does not depend on
``\\mu`` (Appendix B.1). The initial guess is ``10^{-3}`` times the lower-triangular matrix of ones. We then extract
``\\mathrm{diag}(\\tilde{G}_L)`` as an MPS with delta tensors ([`diagmps`](@ref)), and evaluate
``D = [\\mathrm{diag}(\\tilde{G}_L)]^{-1/2}`` (Eq. (23)) element-wise by RSI with uniform sketches
([`elementwise_function`](@ref), Sec. 3.2). The inverse square root is evaluated as ``1/\\sqrt{\\max(x, \\epsilon)}``,
so a non-positive entry, which Lemma 1 rules out unless its whole row is lost, gives a large finite entry of ``D``
rather than an error.

`D_kwargs` may hold the `cutoff` (default `1e-15`), `maxdim` (default `typemax(Int)`) and `oversample` (default `10`)
of [`elementwise_function`](@ref). With `preconditioning = true`, the system is preconditioned by
``\\mathbb{P}_L R \\mathbb{P}_L + \\mu^{-1}(\\mathbb{I} - \\mathbb{P}_L)``, where ``R`` is the
[`rank1_preconditioner`](@ref) of its operator and ``\\mu^{-1}`` inverts the complement block. Note that this does not
speed the build up: on a ``2^6``-point QTT Laplacian with `ResidualALS`, it multiplied the local CG iterations by 3.1
for ``\\mu = 1``, 1.04 for ``\\mu = 10^3`` and 7.8 for ``\\mu = 10^7``, for a preconditioner of the same quality;
`residual_tol` adds a [`ResidualObserver`](@ref) on the system as `sweep_observer!`; `compute_residual = true` logs the
relative residual of the system after the solve, at the cost of two more sandwiches with its operator; the other
`kwargs` go to [`linearsolve`](@ref).
"""
function fai_dg(
    operator::MPO,
    μ::Real,
    g_solver::AbstractLinearSolver,
    D_kwargs;
    preconditioning::Bool=false,
    residual_tol::Union{Float64,Nothing}=nothing,
    compute_residual::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...,
)
    array_type = backend_type(operator)

    sites = only.(siteinds(operator; plev=0))
    n = length(sites)

    # The unknown X = G̃_L is vectorized on (helper_sites, sites): the helper sites carry its row index and become the
    # output (primed) sites of G̃_L at the end, and the original sites carry its column index. Eq. (10) needs X A, i.e.
    # (XA)_ij = Σ_k X_ik A_kj, so the operator acts on the column index (the original sites) as Aᵀ, with the identity
    # on the row index. (Acting on the helper sites instead would solve AX, which gives the transposed FSAI.)
    A = transpose_mpo(operator)
    helper_sites = sim(sites) # Helper sites to expand the space and vectorize the linear system

    combiners = [combiner(sites[i], helper_sites[i]; tags="csite=$i") for i in 1:n] # Combiners that join a pair of sites coming from the two spaces into a single index
    combiner_inds = [combinedind(combiner) for combiner in combiners]

    projector = triang_projector_mpo(sites, combiners, :lower; array_type=array_type)
    projector_complement = +(identity_mpo(combiner_inds, array_type), -1.0 * projector; alg="directsum")

    linsolve_b = identity_mpo(sites, backend_type(operator))
    linsolve_x0 = 1e-3 * lower_triang_mpo(sites, true; array_type=array_type)

    for i in 1:n
        A[i] = A[i] * delta(helper_sites[i], helper_sites[i]') # Identity on the row index of X
        A[i] = A[i] * combiners[i] * combiners[i]'

        replaceind!(linsolve_b[i], sites[i]' => helper_sites[i])
        linsolve_b[i] = linsolve_b[i] * combiners[i]

        replaceind!(linsolve_x0[i], sites[i]' => helper_sites[i])
        linsolve_x0[i] = linsolve_x0[i] * combiners[i]
    end

    projected_A = +(
        apply(projector, apply(A, projector; alg="naive"); alg="naive"), μ * projector_complement; alg="directsum"
    )
    projected_b = MPS([linsolve_b...])

    projected_x0 = MPS([linsolve_x0...])
    ls_kwargs = Dict{Symbol,Any}(kwargs)
    if !isnothing(residual_tol)
        ls_kwargs[:sweep_observer!] = ResidualObserver(; A=projected_A, b=projected_b, rtol=residual_tol)
    end

    if preconditioning
        preconditioner = rank1_preconditioner(projected_A)
        projected_preconditioner = Preconditioner(
            +(
                apply(projector, apply(preconditioner.operator, projector; alg="naive"); alg="naive"),
                inv(μ) * projector_complement; # (μ(I - P_L))⁻¹ = (I - P_L)/μ on the complement
                alg="directsum",
            ),
        )
        unscaledG = linearsolve(
            g_solver, projected_A, projected_b, projected_preconditioner, projected_x0; ls_kwargs...
        )
    else
        unscaledG = linearsolve(g_solver, projected_A, projected_b, projected_x0; ls_kwargs...)
    end

    # The residual of the projected system costs two more sandwiches with projected_A, so it is opt-in
    if compute_residual
        term1 = ITensorMPS.inner(projected_A, unscaledG, projected_A, unscaledG)
        term2 = 2 * real(ITensorMPS.inner(projected_b', projected_A, unscaledG))
        term3 = ITensorMPS.inner(projected_b, projected_b)
        rel_error = sqrt(max(0.0, term1 - term2 + term3)) / norm(projected_b)
        @info "Preconditioner found with rel residual = $rel_error"
    end

    # We need to go back to MPO form from the vectorized solution by reverting the combiners!
    for i in 1:n
        unscaledG[i] = unscaledG[i] * dag(combiners[i]) # Unfold the legs 
        replaceind!(unscaledG[i], helper_sites[i] => sites[i]') # Map the helper legs into the original primed space
    end

    unscaledG = MPO([unscaledG...])
    unscaledGdiag_mps = diagmps(unscaledG)

    f(x) = inv(sqrt(max(x, eps(Float64))))
    cutoff = get(D_kwargs, :cutoff, 1e-15)
    maxdim = get(D_kwargs, :maxdim, typemax(Int))
    oversample = get(D_kwargs, :oversample, 10)

    D_mps = elementwise_function(
        f, unscaledGdiag_mps, true; cutoff=cutoff, nsite=2, maxdim=maxdim, oversample=oversample, outputlevel=0, rng=rng
    )

    placeholder_sites = sim.(sites)
    ITensorMPS.replace_siteinds!(D_mps, placeholder_sites)
    for i in 1:n
        D_mps[i] = D_mps[i] * delta(placeholder_sites[i], sites[i], sites[i]')
    end
    return unscaledG, MPO([D_mps...])
end

"""
    fai_preconditioner(A::MPO; μ::Real, solver::AbstractLinearSolver, totalmaxdim::Int=5, D_cutoff::Real=1e-15,
                       D_maxdim::Int=typemax(Int), D_oversample::Int=10, preconditioning::Bool=false,
                       residual_tol=nothing, compute_residual::Bool=false, rng::AbstractRNG=Random.default_rng(),
                       kwargs...) -> FactorizedPreconditioner

Build a factorized approximate inverse (FAI) preconditioner ``M^{-1} = G_L^\\top G_L \\approx A^{-1}`` of the SPD
operator `A` in the TT format.

Truncated TT approximate inverses control the norm error but not the sign of their eigenvalues, so they can become
indefinite and break CG (Sec. 3 and Fig. 3 of Torrente et al., *Factorized approximate inverse preconditioners in the
tensor train format*). We therefore build the inverse in factorized form, as in the FSAI of Kolotilina & Yeremin
(1993): ``G_L^\\top G_L`` is positive semidefinite by construction, and definite whenever ``G_L`` is non-singular,
whatever the truncation of ``G_L``. Since ``G_L`` approximates the inverse ``L_A^{-1}`` of the Cholesky factor
``A = L_A L_A^\\top``, which we cannot compute in the TT format, we proceed in two steps. We first solve the projected
system of Eq. (22) with `solver` for the lower-triangular approximate inverse ``\\tilde{G}_L``, defined by
``(\\tilde{G}_L A)_{ij} = \\delta_{ij}`` for ``i \\ge j`` (Eq. (10)); this needs no Cholesky factor. We then set
``G_L = D \\tilde{G}_L``, truncated to `totalmaxdim`, where ``D = [\\mathrm{diag}(\\tilde{G}_L)]^{-1/2}`` (Eq. (23)) is
evaluated element-wise by RSI with uniform sketches ([`elementwise_function`](@ref), Sec. 3.2). For an exact
``\\tilde{G}_L`` this gives ``G_L = L_A^{-1}`` and ``M^{-1} = A^{-1}``. The bond dimension of ``G_L`` plays the role of
the sparsity pattern of the classical FSAI: it sets the cost of storing and applying the preconditioner, at the expense
of its quality.

The cost is dominated by the solve for ``\\tilde{G}_L``, an MPS with ``N`` sites of dimension ``d^2`` whose operator
contains ``A`` and the projector ``\\mathbb{P}_L`` of bond dimension 2 (Eq. (14)) twice. In the paper's experiments we
use ``\\mu = 10^7`` (the results do not change for ``\\mu`` between ``10^5`` and ``10^8``, and the exact solution does
not depend on ``\\mu``, Appendix B.1), build ``\\tilde{G}_L`` with MALS ([`ResidualALS`](@ref)) at a local tolerance
between ``10^{-5}`` and ``10^{-8}``, stop once the residual of Eq. (22) is below ``10^{-1}`` (`residual_tol = 0.1`),
and keep ``\\chi_{G_L}`` between 4 and 10. The FAI pays off when it is reused, e.g. across the time steps of the
Chorin projection, and when the local dimension is small (Sec. 5.2); with a large local dimension ``d^2`` the build
becomes expensive (Sec. 5.1). Note two limitations. The shortcut of Eq. (23) for ``D`` holds only if ``\\tilde{G}_L``
solves Eq. (10) exactly; Lemma 1 bounds the error of each diagonal entry by the relative energy error of its row, but
on strongly non-uniform grids or with strongly varying coefficients this shortcut can lose accuracy (Appendix B.6).

The value of ``\\mu`` sets the conditioning of the build, although not its exact solution. Since ``\\mathbb{P}_L``
acts on each row of ``\\tilde{G}_L`` separately, the operator of Eq. (22) is block diagonal, with the leading principal
submatrices of ``A``, whose eigenvalues lie in ``[\\lambda_{\\min}(A), \\lambda_{\\max}(A)]``, and ``\\mu`` times the
identity on the entries above the diagonal, so its condition number is
``\\max(\\mu, \\lambda_{\\max}(A)) / \\min(\\mu, \\lambda_{\\min}(A))``. A ``\\mu`` far above ``\\lambda_{\\max}(A)``,
such as ``10^7`` on a coarse grid, multiplies it by ``\\mu / \\lambda_{\\max}(A)``: the build converges more slowly and,
with a capped number of sweeps, can return a worse preconditioner (on a ``2^5 \\times 2^5`` QTT pressure operator with
``\\lambda_{\\max}(A) \\approx 8 \\cdot 10^3``, ``\\chi_{G_L} = 4`` and 30 MALS sweeps, ``\\mu = 10^7`` gave
``\\kappa(G_L^\\top G_L A) = 146`` and ``\\mu = 10^3`` gave 5.3). Since ``\\lambda_{\\max}(A)`` grows as ``h^{-2}``, the
paper's ``\\mu = 10^7`` is close to the top of the spectrum only on its finest grids. A value at the top of the
spectrum, ``\\mu \\approx \\lambda_{\\max}(A)``, is a reasonable choice: it keeps the condition number of the build at
``\\kappa(A)`` and weights the entries above the diagonal at least as strongly as any eigenvalue of ``A``. A rough
estimate suffices, e.g. from the stencil (at most ``4/h^2`` per axis for the second-order Laplacian) or from a few
Lanczos iterations.

# Arguments

  - `A::MPO`: SPD operator, with site indices at prime levels 0 (input) and 1 (output).

# Keywords

  - `μ::Real`: weight of the equations of the entries above the diagonal of ``\\tilde{G}_L`` in Eq. (22), e.g. `1e7`.
  - `solver::AbstractLinearSolver`: solver of Eq. (22), e.g. [`ResidualALS`](@ref) with `ishermitian` and `isposdef`
    set, since the system is SPD.
  - `totalmaxdim::Int=5`: maximum bond dimension ``\\chi_{G_L}`` of ``G_L = D \\tilde{G}_L``.
  - `D_cutoff::Real=1e-15`, `D_maxdim::Int=typemax(Int)`, `D_oversample::Int=10`: truncation cutoff, maximum bond
    dimension and sketch oversampling of the element-wise map that computes ``D``.
  - `preconditioning::Bool=false`: precondition the solve for ``\\tilde{G}_L`` with [`rank1_preconditioner`](@ref).
    In our tests this did not speed the build up (see the internal `DifferenTT.fai_dg`).
  - `residual_tol::Union{Float64,Nothing}=nothing`: if set, stop the sweeps of `solver` once the relative residual of
    Eq. (22) is below it (through a [`ResidualObserver`](@ref) used as `sweep_observer!`).
  - `compute_residual::Bool=false`: compute and log the relative residual of Eq. (22) after the solve.
  - `rng::AbstractRNG=Random.default_rng()`: random number generator of the sketches of ``D``.
  - `kwargs...`: passed to [`linearsolve`](@ref) for Eq. (22).

# Returns

  - `FactorizedPreconditioner`: wraps ``G_L``; pass it to [`linearsolve`](@ref).

# Examples

```julia
A, sites = …                                  # SPD MPO, e.g. -1.0 * symmetrized_laplacian(field, grid)
build = ResidualALS(;
    sweep_kwargs=SweepingArgs(; nsweeps=10, maxdim=8),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-5),
)
M⁻¹ = fai_preconditioner(A; μ=1e7, solver=build, totalmaxdim=8, residual_tol=0.1)
x = linearsolve(ResidualALS(), A, b, M⁻¹, nothing)
```

See also [`FactorizedPreconditioner`](@ref), [`Preconditioner`](@ref).
"""
function fai_preconditioner(
    operator::MPO;
    μ::Real,
    solver::AbstractLinearSolver,
    totalmaxdim::Int=5,
    D_cutoff::Real=1e-15,
    D_maxdim::Int=typemax(Int),
    D_oversample::Int=10,
    preconditioning::Bool=false,
    residual_tol::Union{Float64,Nothing}=nothing,
    compute_residual::Bool=false,
    rng::AbstractRNG=Random.default_rng(),
    kwargs...,
)
    D_kwargs = (; cutoff=D_cutoff, maxdim=D_maxdim, oversample=D_oversample)
    unscaledG, D = fai_dg(
        operator, μ, solver, D_kwargs; preconditioning, residual_tol, compute_residual, rng, kwargs...
    )
    Gl = apply(D, unscaledG; maxdim=totalmaxdim)
    return FactorizedPreconditioner(Gl)
end

"""
    fai_preconditioner(A::MPO, μ::Real, g_solver::AbstractLinearSolver, D_kwargs; kwargs...)

Deprecated positional form of [`fai_preconditioner`](@ref), kept so that existing scripts keep running; `D_kwargs`
maps to `D_cutoff`, `D_maxdim` and `D_oversample`.
"""
function fai_preconditioner(operator::MPO, μ::Real, g_solver::AbstractLinearSolver, D_kwargs; kwargs...)
    Base.depwarn(
        "`fai_preconditioner(A, μ, solver, D_kwargs; kwargs...)` is deprecated, use " *
        "`fai_preconditioner(A; μ, solver, D_cutoff, D_maxdim, D_oversample, kwargs...)` instead.",
        :fai_preconditioner,
    )
    return fai_preconditioner(
        operator;
        μ=μ,
        solver=g_solver,
        D_cutoff=get(D_kwargs, :cutoff, 1e-15),
        D_maxdim=get(D_kwargs, :maxdim, typemax(Int)),
        D_oversample=get(D_kwargs, :oversample, 10),
        kwargs...,
    )
end

"""
    _spectral_lower_bound(mat::AbstractMatrix) -> Float64

Return the smallest eigenvalue of a symmetric `mat`, or the smallest real part of its eigenvalues otherwise.
"""
function _spectral_lower_bound(mat::AbstractMatrix)
    dense = Matrix(mat)
    return ishermitian(dense) ? eigmin(Hermitian(dense)) : minimum(real, eigvals(dense))
end

"""
    sincquadrature_preconditioner(arch::TTArchitecture{N}, sites::Vector{<:Index}, A_list::Vector{<:AbstractMatrix},
                                  M::Int; array_type::Type{<:AbstractArray}=Array,
                                  normalize_spectrum::Bool=true) where {N}
        -> Preconditioner

Build the Sinc-quadrature (exponential-sum) preconditioner ``B_M \\approx L_0^{-1}`` of the Kronecker sum
``L_0 = \\sum_\\ell A_\\ell`` of the one-dimensional SPD operators in `A_list` (Khoromskij, 2009; Eq. (26) of Torrente
et al., *Factorized approximate inverse preconditioners in the tensor train format*).

The inverse of a Kronecker sum has no low-rank form in general, but its exponential does, since
``e^{-tL_0} = \\bigotimes_\\ell e^{-tA_\\ell}``. We therefore approximate ``1/x`` on ``[1, \\infty)`` by the Sinc
quadrature ``\\sum_{k=-M}^{M} c_k e^{-t_k x}`` with ``t_k = e^{kh}``, ``c_k = h t_k`` and ``h = \\pi/\\sqrt{M}``, and
obtain ``B_M`` as a sum of ``2M + 1`` Kronecker products, of bond dimension at most ``2M + 1``. The quadrature is
accurate on spectra normalized to ``[1, R]``, so with `normalize_spectrum = true` we rescale ``L_0`` by
``\\lambda_{\\min}(L_0) = \\sum_\\ell \\lambda_{\\min}(A_\\ell)``, i.e.
``B_M = \\lambda_{\\min}^{-1} \\sum_k c_k \\bigotimes_\\ell e^{-t_k A_\\ell / \\lambda_{\\min}}``, which makes ``B_M``
independent of the scaling of `A_list` (e.g. a ``1/h^2`` factor). `normalize_spectrum = false` applies the quadrature
to the unscaled operators. Note that ``B_M`` is built for the separable part ``L_0`` only: for an operator with
couplings between the dimensions, such as the anisotropic ``\\mathcal{L}(\\rho)`` of Eq. (28), its quality degrades as
the operator becomes less separable (Sec. 5.1).

# Arguments

  - `arch::TTArchitecture{N}`: TT architecture with one site per dimension; `arch.site_map` assigns `A_list` to sites.
  - `sites::Vector{<:Index}`: site indices of the operator.
  - `A_list::Vector{<:AbstractMatrix}`: the ``N`` one-dimensional SPD operators, of sizes `arch.var_dims`.
  - `M::Int`: number of quadrature points on each side; the result has ``2M + 1`` terms.

# Keywords

  - `array_type::Type{<:AbstractArray}=Array`: array backend of the result.
  - `normalize_spectrum::Bool=true`: rescale ``L_0`` by ``\\lambda_{\\min}(L_0)`` before the quadrature.

# Returns

  - `Preconditioner`: ``B_M``.

# Throws

  - `DimensionMismatch`: if a matrix of `A_list` is not square, their number differs from ``N``, or their sizes differ
    from `arch.var_dims`.
  - `ArgumentError`: if ``\\lambda_{\\min}(L_0) \\le 0`` with `normalize_spectrum = true`.
"""
function sincquadrature_preconditioner(
    arch::TTArchitecture{N},
    sites::Vector{<:Index},
    A_list::Vector{<:AbstractMatrix},
    M::Int;
    array_type::Type{<:AbstractArray}=Array,
    normalize_spectrum::Bool=true,
) where {N}
    d = length(A_list)
    all(size(mat, 1) == size(mat, 2) for mat in A_list) ||
        throw(DimensionMismatch("Only square matrices/operators are currently supported"))
    N == d || throw(DimensionMismatch("Number of local operators does not match architecture dimensions"))
    all(arch.var_dims[i] == size(A_list[i], 1) for i in 1:d) ||
        throw(DimensionMismatch("Arch var dims do not match matrix dimensions"))
    h = pi / sqrt(M)

    # Eigenvalues of a Kronecker sum are sums of the eigenvalues of its terms
    λmin = normalize_spectrum ? sum(_spectral_lower_bound, A_list) : 1.0
    λmin > 0 || throw(ArgumentError("The Kronecker sum of A_list must be positive definite, got λ_min = $λmin"))

    precond_mpo = nothing
    total_sites = length(sites)

    for k in (-M):M
        t_k = exp(k * h) / λmin
        c_k = h * t_k

        links = [Index(1, "Link,l=$i") for i in 1:(total_sites-1)]
        mpo_tensors = Vector{ITensor}(undef, total_sites)

        for i in 1:total_sites
            l = arch.site_map[i]
            expA = exp(-t_k * A_list[l])
            s = sites[i]

            T = ITensor(array_type(expA), s', s)

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
        end

        term_mpo = c_k * MPO(mpo_tensors)

        precond_mpo = _nothing_sum_handler(precond_mpo, term_mpo; cutoff=0.0)
    end

    return Preconditioner(precond_mpo)
end

# Implementation following the QTT inverse Laplacian of TT-Toolbox (https://github.com/oseledets/TT-Toolbox, MIT
# license, Copyright (C) 2009-2012 Ivan Oseledets, Sergey Dolgov, Vladimir Kazeev, Thomas Mach, Olga Lebedeva, Dmitry
# Savostyanov, Pavel Zhlobich, Le Song) rather than the formulas of Kazeev & Khoromskij (2012); see
# THIRD_PARTY_NOTICES.md.
"""
    _dd_exact_inverse_laplacian(sites, arch::QTTArchitecture{1}, grid, bc, array_type=Array) -> MPO

Return the exact QTT inverse, of bond dimension 5, of the second-order 1D Laplacian with Dirichlet conditions on both
ends, scaled by ``-h^2``. Throws an `ArgumentError` for other boundary conditions.
"""
function _dd_exact_inverse_laplacian(
    sites::Vector{<:Index},
    arch::QTTArchitecture{1},
    grid::Grid,
    bc::BoundaryConditions,
    array_type::Type{<:AbstractArray}=Array,
)
    (isa(bc.conditions[1].left, DirichletBC) && isa(bc.conditions[1].right, DirichletBC)) ||
        throw(ArgumentError("Only Dirichlet boundary conditions are supported for the exact QTT inverse Laplacian."))

    L = abs(arch.var_nsites[1])

    I_mat = [1.0 0.0; 0.0 1.0]
    P_mat = [0.0 1.0; 1.0 0.0]
    E_mat = [1.0 1.0; 1.0 1.0]
    F_mat = [1.0 -1.0; -1.0 1.0]
    K_mat = [-1.0 0.0; 0.0 1.0]
    L_mat = [0.0 -1.0; 1.0 0.0]

    link_inds = [Index(5, "Link,l=$i") for i in 1:(L-1)]

    tensors = ITensor[]

    for kappa in 1:L
        s = sites[kappa]

        k = L - kappa + 1

        xi_k = (2.0^(k-1) + 1.0) / (2.0^k + 1.0)
        eta_k = (2.0^(k-2)) / (2.0^k + 1.0)
        zeta_k = ((2.0^(k-1) + 1.0) / 2.0^(k-1)) * xi_k

        if kappa == 1
            T = zeros(Float64, 2, 2, 5)

            T[:, :, 1] = I_mat
            T[:, :, 2] = 0.25 * xi_k * I_mat + 0.25 * zeta_k * P_mat
            T[:, :, 3] = xi_k * I_mat - zeta_k * P_mat
            T[:, :, 4] = -0.5 * xi_k * K_mat
            T[:, :, 5] = 0.5 * zeta_k * L_mat

            it = adapt(array_type, ITensor(T, s, s', link_inds[kappa]))
            push!(tensors, it)

        elseif kappa == L
            T = zeros(Float64, 2, 2, 5)

            T[:, :, 1] = (1.0 / 3.0) * (I_mat + E_mat)
            T[:, :, 2] = 2.0 * E_mat
            T[:, :, 3] = (1.0 / 18.0) * F_mat
            T[:, :, 4] = (2.0 / 3.0) * K_mat
            T[:, :, 5] = -(2.0 / 3.0) * L_mat

            it = adapt(array_type, ITensor(T, s, s', link_inds[kappa-1]))
            push!(tensors, it)

        else
            T = zeros(Float64, 2, 2, 5, 5)

            T[:, :, 1, 1] = I_mat
            T[:, :, 1, 2] = 0.25 * xi_k * I_mat + 0.25 * zeta_k * P_mat
            T[:, :, 1, 3] = xi_k * I_mat - zeta_k * P_mat
            T[:, :, 1, 4] = -0.5 * xi_k * K_mat
            T[:, :, 1, 5] = 0.5 * zeta_k * L_mat

            T[:, :, 2, 2] = 2.0 * E_mat

            T[:, :, 3, 2] = 2.0 * (eta_k^2) * F_mat
            T[:, :, 3, 3] = 2.0 * (xi_k^2) * E_mat
            T[:, :, 3, 4] = 2.0 * xi_k * eta_k * K_mat
            T[:, :, 3, 5] = 2.0 * xi_k * eta_k * L_mat

            T[:, :, 4, 2] = 4.0 * eta_k * K_mat
            T[:, :, 4, 4] = 2.0 * xi_k * E_mat

            T[:, :, 5, 2] = -4.0 * eta_k * L_mat
            T[:, :, 5, 5] = 2.0 * xi_k * E_mat

            it = adapt(array_type, ITensor(T, s, s', link_inds[kappa-1], link_inds[kappa]))
            push!(tensors, it)
        end
    end

    return (-grid.dx[1]^2) * MPO(tensors)
end

"""
    _dn_exact_inverse_laplacian(sites, arch::QTTArchitecture{1}, grid, bc, array_type=Array) -> MPO

Return the exact QTT inverse, of bond dimension 4, of the second-order 1D Laplacian with a Dirichlet condition on the
left and a Neumann condition on the right, scaled by ``-h^2``. Throws an `ArgumentError` for other boundary conditions.
"""
function _dn_exact_inverse_laplacian(
    sites::Vector{<:Index},
    arch::QTTArchitecture{1},
    grid::Grid,
    bc::BoundaryConditions,
    array_type::Type{<:AbstractArray}=Array,
)
    (isa(bc.conditions[1].left, DirichletBC) && isa(bc.conditions[1].right, NeumannBC)) || throw(
        ArgumentError("Only Dirichlet-Neumann boundary conditions are supported for this exact QTT inverse Laplacian."),
    )

    L = abs(arch.var_nsites[1])

    I_mat = [1.0 0.0; 0.0 1.0]
    I2_mat = [0.0 0.0; 0.0 1.0]
    J_mat = [0.0 1.0; 0.0 0.0]
    E_mat = [1.0 1.0; 1.0 1.0]

    link_inds = [Index(4, "Link,l=$i") for i in 1:(L-1)]

    tensors = ITensor[]

    for kappa in 1:L
        s = sites[kappa]

        if kappa == 1
            T = zeros(Float64, 2, 2, 4)
            T[:, :, 1] = I_mat
            T[:, :, 2] = I2_mat
            T[:, :, 3] = J_mat
            T[:, :, 4] = J_mat'

            it = adapt(array_type, ITensor(T, s, s', link_inds[kappa]))
            push!(tensors, it)

        elseif kappa == L
            T = zeros(Float64, 2, 2, 4)
            T[:, :, 1] = E_mat + I2_mat
            T[:, :, 2] = 2.0 * E_mat
            T[:, :, 3] = E_mat + I2_mat + J_mat'
            T[:, :, 4] = E_mat + I2_mat + J_mat

            it = adapt(array_type, ITensor(T, s, s', link_inds[kappa-1]))
            push!(tensors, it)

        else
            T = zeros(Float64, 2, 2, 4, 4)
            T[:, :, 1, 1] = I_mat
            T[:, :, 1, 2] = I2_mat
            T[:, :, 1, 3] = J_mat
            T[:, :, 1, 4] = J_mat'

            T[:, :, 2, 2] = 2.0 * E_mat

            T[:, :, 3, 2] = I2_mat + J_mat'
            T[:, :, 3, 3] = E_mat

            T[:, :, 4, 2] = I2_mat + J_mat
            T[:, :, 4, 4] = E_mat
            it = adapt(array_type, ITensor(T, s, s', link_inds[kappa-1], link_inds[kappa]))
            push!(tensors, it)
        end
    end
    return (-grid.dx[1]^2) * MPO(tensors)
end

"""
    inverse_laplacian_preconditioner(sites::Vector{<:Index}, arch::QTTArchitecture{1}, grid::Grid,
                                     bc::BoundaryConditions, array_type::Type{<:AbstractArray}=Array) -> Preconditioner

Build the exact QTT inverse of the second-order one-dimensional Laplacian on `grid` with the boundary conditions `bc`.

The inverse of the 1D Laplacian has an explicit QTT representation of bond dimension 5 (Dirichlet–Dirichlet) or 4
(Dirichlet–Neumann) (Kazeev & Khoromskij, 2012); we follow the implementation of TT-Toolbox. The result is the
inverse of the operator that [`symmetrized_laplacian`](@ref) builds with `stencil_order = 2` for the same grid and
boundary conditions, which is negative definite: to precondition the SPD ``-\\Delta_h``, use ``-1`` times its
operator. It is exact for one dimension only, and serves as a reference.

# Arguments

  - `sites::Vector{<:Index}`: site indices of the operator.
  - `arch::QTTArchitecture{1}`: one-dimensional QTT architecture.
  - `grid::Grid`: the grid, whose spacing ``h`` scales the result by ``h^2``.
  - `bc::BoundaryConditions`: Dirichlet on the left, and Dirichlet or Neumann on the right.
  - `array_type::Type{<:AbstractArray}=Array`: array backend of the result.

# Throws

  - `ArgumentError`: for other boundary conditions.
"""
function inverse_laplacian_preconditioner(
    sites::Vector{<:Index},
    arch::QTTArchitecture{1},
    grid::Grid,
    bc::BoundaryConditions,
    array_type::Type{<:AbstractArray}=Array,
)
    if isa(bc.conditions[1].left, DirichletBC) && isa(bc.conditions[1].right, DirichletBC)
        mpo = _dd_exact_inverse_laplacian(sites, arch, grid, bc, array_type)
    elseif isa(bc.conditions[1].left, DirichletBC) && isa(bc.conditions[1].right, NeumannBC)
        mpo = _dn_exact_inverse_laplacian(sites, arch, grid, bc, array_type)
    else
        throw(
            ArgumentError(
                "inverse_laplacian_preconditioner supports only Dirichlet-Dirichlet or Dirichlet-Neumann " *
                "boundary conditions",
            ),
        )
    end
    return Preconditioner(mpo)
end
