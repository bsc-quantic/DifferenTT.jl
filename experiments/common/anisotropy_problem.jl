"""
    Rotated (equicorrelated) anisotropy benchmark: shared problem definition

Included by `tt_anisotropy/worker.jl` and `tt_anisotropy_global/worker.jl` so both benchmarks solve
the identical problem

    -∇⋅(K∇u) = f  in (0,1)^N,   u = 0 on the boundary,   K = (1-ρ) I + ρ 𝟏𝟏ᵀ.

Instead of ρ, the benchmarks sweep the modeling floor F = κ(L0⁻¹ L) ≤ (1+(N-1)ρ)/(1-ρ) of any
preconditioner built on the separable L0 = -Δ (e.g. sinc quadrature), so every N is compared at
equal difficulty. See Sec. 5.1 and Appendix A of the paper.
"""

"""
ρ such that the analytic floor (1+(N-1)ρ)/(1-ρ) equals F.
"""
rho_from_floor(F::Real, N::Int) = (F - 1) / (F + N - 1)

"""
Analytic floor κ(L0⁻¹ L) bound for a given ρ ≥ 0.
"""
floor_from_rho(ρ::Real, N::Int) = (1 + (N - 1) * ρ) / (1 - ρ)

"""
    get_1d_A(i, grid, arch)

1D factor of L0 = -Δ along dimension `i`, used by the sinc preconditioner. It does not depend on ρ:
sinc is always built on the separable Laplacian.
"""
function get_1d_A(i::Int, grid, arch)
    bc = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    # onedim_differential_operator with an arch::TTArchitecture{N} returns the matrix we need
    mat = DifferenTT.onedim_differential_operator(2, arch, i, bc, Array; stencil_order=2)
    # the exact Laplacian has a minus sign to be positive definite
    # also scale by 1 / dx^2
    mat = -1.0 * mat .* (1.0 / grid.dx[i]^2)
    return mat
end

function get_coulomb_sol(N::Int, ε::Float64=0.05)
    c = fill(0.5, N)
    function sol(x)
        r2 = sum((x[j] - c[j])^2 for j in 1:N)
        env = prod(sin(pi * x[j]) for j in 1:N)
        return (1.0 / sqrt(r2 + ε^2)) * env
    end
    return sol
end

"""
    build_rhs(kind, A, sites, grid, arch; sol_maxdim)

Right-hand side of the benchmark. Returns `(rhs, sol)`, where `sol` is `nothing` unless the exact
discrete solution is known.

  - `:ones` (default): f ≡ 1. u = L⁻¹𝟏 is (twice) the mean exit time from the unit box of a Brownian
    motion with covariance K, e.g. the expected life of a multi-asset corridor contract on N
    equicorrelated assets. Rank 1 and deterministic.
  - `:random`: normalized random MPS of bond dimension 5, which excites the transverse modes that set
    λ_min(K) and that the smooth, permutation-symmetric f ≡ 1 barely excites.
  - `:manufactured`: Coulomb-like TCI solution of `tt_poisson`, with the discrete rhs A * sol, so the
    solve error can be measured.
"""
function build_rhs(kind::Symbol, A::MPO, sites, grid, arch; sol_maxdim::Int)
    N = length(sites)
    if kind == :ones
        return ones_mps(sites), nothing

    elseif kind == :random
        Random.seed!(N)
        rhs = random_mps(sites; linkdims=5)
        return rhs / norm(rhs), nothing

    elseif kind == :manufactured
        sol_func = get_coulomb_sol(N, 0.2)
        sol_initialpivot = optfirstpivot(Float64, sol_func, grid, arch)
        # Cap ttcross by sol_maxdim to avoid OOM
        sol_tci, _ = crossinterpolate2(
            Float64, sol_func, grid, arch; initialpivots=[sol_initialpivot], maxbonddim=sol_maxdim, tolerance=1e-12
        )
        sol_mps = MPS(sol_tci)
        replace_siteinds!(sol_mps, sites)
        # Discrete RHS A * sol_mps so FD truncation error does not dominate the solve error.
        return apply(A, sol_mps; alg="naive"), sol_mps

    else
        error("Unknown rhs kind: $kind (expected :ones, :random or :manufactured)")
    end
end

"""
    build_problem(N, F; nodes_per_dim, rhs_kind, sol_maxdim)

Build the benchmark problem for dimension `N` and floor `F`. Returns a NamedTuple with

  - `A`: -∇⋅(K∇) as a rank-3 MPO (rank 2 for F = 1),
  - `A0`: -Δ, the separable operator sinc approximates (used by the FAI-on-L0 control),
  - `rhs`, `sol` (see `build_rhs`), `A_list` (1D factors of A0 for sinc), `grid`, `arch`, `ρ`.
"""
function build_problem(N::Int, F::Real; nodes_per_dim::Int, rhs_kind::Symbol, sol_maxdim::Int)
    ρ = rho_from_floor(F, N)

    box_size = ntuple(i -> 1.0, N)
    nodes = ntuple(i -> nodes_per_dim, N)

    arch = TTArchitecture(nodes)
    grid = Grid(arch, box_size)

    # Homogeneous Dirichlet: `scalar_dirichlet_correction` and `symmetrize_rhs` are no-ops here,
    # and the former would not cover the mixed-derivative term anyway, so both are skipped.
    bcs = ntuple(i -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), N)
    bc = BoundaryConditions(bcs...)

    sites = siteinds(arch)
    field = ScalarField(sites, arch, bc)

    A = -1.0 * equicorrelated_laplacian(field, grid; correlation=ρ, stencil_order=2)
    A0 = -1.0 * symmetrized_laplacian(field, grid; stencil_order=2)
    @assert maxlinkdim(A) <= 3

    rhs, sol = build_rhs(rhs_kind, A, sites, grid, arch; sol_maxdim=sol_maxdim)

    A_list = [get_1d_A(i, grid, arch) for i in 1:N]

    return (; A, A0, rhs, sol, A_list, grid, arch, ρ)
end

"""
Relative 2-norm distance between two MPS, computed with inner products to save memory.
"""
function relative_distance(x::MPS, y::MPS)
    d2 = inner(x, x) - 2 * real(inner(y, x)) + inner(y, y)
    return sqrt(max(0.0, d2)) / norm(y)
end
