"""
    _regularized_inv(mat; rtol=eps(Float64)) -> Matrix

Invert `mat` with Tikhonov regularization: `(mat + ε I)⁻¹` where
`ε = rtol * max(‖mat‖₂, 1)`. Avoids `SingularException` / huge entries when
TT truncation leaves a nearly singular local tensor.
"""
function _regularized_inv(mat::AbstractMatrix; rtol::Float64=eps(Float64))
    n = size(mat, 1)
    @assert size(mat, 1) == size(mat, 2) "Local rank-1 tensor must be square"
    ε = rtol * max(opnorm(mat), 1.0)
    return inv(mat + ε * I(n))
end

"""
    rank1_preconditioner(operator::MPO)

Build a rank-1 preconditioner by truncating the operator to bond dimension 1 and inverting each local n×n matrix.

# Returns

  - `Preconditioner`: The preconditioner with all bond dimensions equal to 1.
"""
function rank1_preconditioner(operator::MPO)
    rank1_operator = deepcopy(operator)
    truncate!(rank1_operator; maxdim=1)

    preconditioner_tensors = ITensor[]
    # TODO Ensure that this method of getting rid of the dummy legs is robust
    for tensor in rank1_operator
        mat = order(tensor) == 3 ? dropdims(array(tensor); dims=3) : dropdims(array(tensor); dims=(3, 4))
        invmat = _regularized_inv(mat)
        push!(preconditioner_tensors, ITensor(reshape(invmat, dims(tensor)), inds(tensor)...))
    end
    return Preconditioner(MPO(preconditioner_tensors))
end

"""
    newtoniter_preconditioner(operator::MPO, init_preconditioner::MPO, niter::Int)

Refine a preconditioner via Newton iteration: X_{k+1} = X_k (2I - A X_k). Not yet implemented.
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

"""
    newtoniter_preconditioner(operator::MPO, niter::Int)

Build a preconditioner via Newton iteration starting from a rank-1 initial guess.
"""
function newtoniter_preconditioner(operator::MPO, niter::Int; cutoff::Float64=1e-15, maxdim::Int=typemax(Int64))
    init_preconditioner = 0.01 * rank1_preconditioner(operator).operator
    return newtoniter_preconditioner(operator, init_preconditioner, niter; cutoff=cutoff, maxdim=maxdim)
end

"""
    linsolve_preconditioner(operator::MPO, solver::AbstractLinearSolver; kwargs...)

Build a preconditioner M⁻¹ by solving the Sylvester-like equation `A*X + X*A = 2I` using a linear solver in an extended Hilbert space.

# Returns

  - `Preconditioner`: The approximate inverse of the operator.
"""
function linsolve_preconditioner(
    operator::MPO,
    solver::AbstractLinearSolver,
    symmetrize::Bool=true;
    linsolve_x0=nothing,
    preconditioning::Bool=false,
    residual_tol::Union{Float64,Nothing}=nothing,
    kwargs...,
)
    op1 = deepcopy(operator) # To construct AX
    sites = only.(siteinds(operator; plev=0))

    helper_sites = sim(sites) # Helper sites to extend the space

    if symmetrize
        op2 = transpose(op1) # To construct XA
        replace_mpo_siteinds!(op2, helper_sites) # In this case A, for XA, lives in the helper space
    else
        op2 = nothing
    end

    linsolve_b = identity_mpo(sites, backend_type(operator)) # Construct the right hand side
    linsolve_b = symmetrize ? 2 * linsolve_b : linsolve_b # Construct the right hand side

    N = length(op1)
    combiners = [combiner(sites[i], helper_sites[i]; tags="csite=$i") for i in 1:N] # Combiners that join a pair of sites coming from the two spaces into a single index

    for i in 1:N
        op1[i] = op1[i] * delta(helper_sites[i], helper_sites[i]') # Id living on the helper space for op1
        op1[i] = op1[i] * combiners[i] * combiners[i]' # Join op and identity into a single leg for both operators

        if symmetrize
            op2[i] = op2[i] * delta(sites[i], sites[i]') # Id living on the original space for op1
            op2[i] = op2[i] * combiners[i] * combiners[i]'
        end

        replaceind!(linsolve_b[i], sites[i]' => helper_sites[i])
        linsolve_b[i] = linsolve_b[i] * combiners[i]
    end

    linsolve_A = symmetrize ? +(op1, op2; alg="directsum") : op1  # Construct the operator AX + XA
    linsolve_b = MPS([linsolve_b...]) # Convert the still MPO object into an MPS

    if isnothing(linsolve_x0)
        _linsolve_x0 = MPS([1e-12 * adapt(backend_type(operator), random_mpo(sites, 1))...])
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

    @info "Linsolve preconditioner with preconditioning set to $preconditioning"

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

# TODO Test also densitymatrix summing and zipup contraction -> Pass also maxdim and cutoff if needed
function fai_dg(
    operator::MPO,
    μ::Real,
    g_solver::AbstractLinearSolver,
    D_kwargs;
    preconditioning::Bool=false,
    residual_tol::Union{Float64,Nothing}=nothing,
    kwargs...,
)
    A = transpose(operator) # To construct XA
    array_type = backend_type(operator)

    sites = only.(siteinds(operator; plev=0))
    n = length(sites)

    helper_sites = sim(sites) # Helper sites to expand the space and vectorize the linear system
    replace_mpo_siteinds!(A, helper_sites) # In this case A, for XA, lives in the helper space

    combiners = [combiner(sites[i], helper_sites[i]; tags="csite=$i") for i in 1:n] # Combiners that join a pair of sites coming from the two spaces into a single index
    combiner_inds = [combinedind(combiner) for combiner in combiners]

    projector = triang_projector_mpo(sites, combiners, :lower; array_type=array_type)
    projector_complement = +(identity_mpo(combiner_inds, array_type), -1.0 * projector; alg="directsum")

    linsolve_b = identity_mpo(sites, backend_type(operator))
    linsolve_x0 = 1e-3 * lower_triang_mpo(sites, true; array_type=array_type)

    for i in 1:n
        A[i] = A[i] * delta(sites[i], sites[i]') # Id living on the original space for op1
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
                μ * projector_complement;
                alg="directsum",
            ),
        )
        unscaledG = linearsolve(
            g_solver, projected_A, projected_b, projected_preconditioner, projected_x0; ls_kwargs...
        )
    else
        unscaledG = linearsolve(g_solver, projected_A, projected_b, projected_x0; ls_kwargs...)
    end

    term1 = ITensorMPS.inner(projected_A, unscaledG, projected_A, unscaledG)
    term2 = 2 * real(ITensorMPS.inner(projected_b', projected_A, unscaledG))
    term3 = ITensorMPS.inner(projected_b, projected_b)
    rel_error = sqrt(max(0.0, term1 - term2 + term3)) / norm(projected_b)

    @info "Preconditioner found with rel residual = $rel_error"

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
        f, unscaledGdiag_mps, true; cutoff=cutoff, nsite=2, maxdim=maxdim, oversample=oversample, outputlevel=0
    )

    placeholder_sites = sim.(sites)
    ITensorMPS.replace_siteinds!(D_mps, placeholder_sites)
    for i in 1:n
        D_mps[i] = D_mps[i] * delta(placeholder_sites[i], sites[i], sites[i]')
    end
    return unscaledG, MPO([D_mps...])
end

function fai_preconditioner(
    operator::MPO,
    μ::Real,
    g_solver::AbstractLinearSolver,
    D_kwargs;
    totalmaxdim::Int=5,
    preconditioning::Bool=false,
    residual_tol::Union{Float64,Nothing}=nothing,
    kwargs...,
)
    unscaledG, D = fai_dg(
        operator, μ, g_solver, D_kwargs; preconditioning=preconditioning, residual_tol=residual_tol, kwargs...
    )
    Gl = apply(D, unscaledG; maxdim=totalmaxdim)
    return FactorizedPreconditioner(Gl)
end

"""
    _spectral_lower_bound(mat::AbstractMatrix) -> Float64

Smallest eigenvalue of a symmetric `mat`, or the smallest real part of its eigenvalues otherwise.
"""
function _spectral_lower_bound(mat::AbstractMatrix)
    dense = Matrix(mat)
    return ishermitian(dense) ? eigmin(Hermitian(dense)) : minimum(real, eigvals(dense))
end

"""
    sincquadrature_preconditioner(arch::TTArchitecture{N}, sites::Vector{<:Index}, A_list::Vector{<:AbstractMatrix}, M::Int; array_type::Type{<:AbstractArray}=Array, normalize_spectrum::Bool=true) where N

Build a Sinc-quadrature (sum of exponentials) preconditioner `B_M ≈ L0⁻¹` for the Kronecker sum
`L0 = Σ_ℓ A_ℓ` of the one-dimensional SPD operators in `A_list` (Khoromskij 2009). The result is
a sum of `2M + 1` rank-one terms, so its bond dimension is at most `2M + 1`.

The quadrature of `1/x` is accurate on spectra normalized to `[1, R]`. With `normalize_spectrum=true`,
`L0` is therefore rescaled by `λ_min(L0) = Σ_ℓ λ_min(A_ℓ)` before applying the quadrature, i.e.
`B_M = (1/λ_min) Σ_k c_k ⊗_ℓ exp(-t_k A_ℓ / λ_min)`, which keeps `B_M L0` well conditioned
regardless of the scaling of `A_list` (e.g. a `1/h²` factor). `normalize_spectrum=false` applies
the quadrature to the unscaled operators.
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
    @assert all(size(mat, 1) == size(mat, 2) for mat in A_list) "Only square matrices/operators are currently supported"
    @assert N == d "Number of local operators does not match architecture dimensions"
    @assert all(arch.var_dims[i] == size(A_list[i], 1) for i in 1:d) "Arch var dims fo not match matrix dimensions"
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

# Implementation following the tt_toolbox, not the paper
function _dd_exact_inverse_laplacian(
    sites::Vector{<:Index},
    arch::QTTArchitecture{1},
    grid::Grid,
    bc::BoundaryConditions,
    array_type::Type{<:AbstractArray}=Array,
)
    @assert isa(bc.conditions[1].left, DirichletBC) && isa(bc.conditions[1].right, DirichletBC) "Only Dirichlet boundary conditions are supported for the exact QTT inverse Laplacian."

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

function _dn_exact_inverse_laplacian(
    sites::Vector{<:Index},
    arch::QTTArchitecture{1},
    grid::Grid,
    bc::BoundaryConditions,
    array_type::Type{<:AbstractArray}=Array,
)
    @assert isa(bc.conditions[1].left, DirichletBC) && isa(bc.conditions[1].right, NeumannBC) "Only Dirichlet-Neumann boundary conditions are supported for this exact QTT inverse Laplacian."

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
    inverse_laplacian_preconditioner(arch::QTTArchitecture{1}, bc::BoundaryConditions, array_type::Type{<:AbstractArray}=Array)

Construct the exact explicit QTT representation of the -1.0 * 1D inverse Laplacian following
Kazeev & Khoromskij 2012 "Low-Rank Explicit QTT Representation of the Laplace Operator and Its Inverse".
Currently supports Dirichlet-Dirichlet and Dirichlet-Neumann boundary conditions.
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
        error("Currently only Dirichlet-Dirichlet or Dirichlet-Neumann boundary conditions are implemented")
    end
    return Preconditioner(mpo)
end
