"""
    Check of the Gershgorin rule for the FAI penalty μ (`common/mu_rules.jl`)

The rule bounds ‖A‖_∞ from the 1D factors of A. We check here that the bound is an upper bound of λmax(A) and that it
is not loose, λmax(A) ≤ μ ≤ 1.5 λmax(A), on small cases where λmax(A) can be computed exactly:

  - anisotropy (Sec. 5.1): 15 nodes per dimension, N = 2, 3, 4 and F ∈ {1, 10, 100}. λmax from the sparse Kronecker
    sum of the same 1D matrices (Lanczos, KrylovKit), checked against the dense MPO of `equicorrelated_laplacian` for
    N = 2, 3, and the closed form N (4 + ρ (N - 1)) / h² of the bound;
  - Chorin (Sec. 5.2): the pressure operator -symmetrized_laplacian of `chorin_pipeline_stats/worker.jl` (stencils of
    order 8, Neumann/Dirichlet in x, periodic in y) on 2⁵ × 2⁵ and 2⁶ × 2⁶ QTT grids, contracted to a dense matrix.
    We also check the QTT 1D matrices against the TT ones, and the bound against the closed form 6.5016 (h_x⁻² + h_y⁻²),
    within 1 %, on 2⁸ × 2⁸ and 2¹⁰ × 2¹⁰.

It prints a table of the ratios μ / λmax and stops with an error if a check fails. Run by `experiments/smoke.jl`:

    julia --project=experiments experiments/common/mu_rules_check.jl
"""

using LinearAlgebra
using SparseArrays
using KrylovKit
using Printf
using ITensors
using ITensorMPS
using DifferenTT

include(joinpath(@__DIR__, "mu_rules.jl"))
include(joinpath(@__DIR__, "anisotropy_problem.jl"))

failures = String[]
check(ok, msg) = ok || (push!(failures, msg); @error msg)

# Dense matrix of an MPO (rows on the primed indices), site 1 the slowest index
function mpo_matrix(A::MPO)
    sites = siteinds(first, A; plev=0)
    # The contraction has up to 2n + 1 indices: bound the order warning just past it
    previous = ITensors.set_warn_order(2 * length(A) + 2)
    T = try
        contract(collect(A))
    finally
        ITensors.set_warn_order(previous)
    end
    n = prod(dim.(sites))
    return reshape(Array(T, reverse(prime.(sites))..., reverse(sites)...), n, n)
end

# Kronecker product with `X` on axis `ℓ` (axis 1 the slowest, as in the MPO) and the identity elsewhere
function kron_axis(ops::Dict{Int,<:AbstractMatrix}, dims)
    return reduce(kron, [haskey(ops, ℓ) ? sparse(ops[ℓ]) : sparse(1.0I, dims[ℓ], dims[ℓ]) for ℓ in eachindex(dims)])
end

function lanczos_max(A)
    vals, _, info = eigsolve(A, ones(size(A, 1)), 1, :LR; issymmetric=true, tol=1e-10, krylovdim=40)
    info.converged >= 1 || error("Lanczos did not converge")
    return first(vals)
end

# ---- Anisotropy ----
println("Anisotropy (15 nodes per dimension, stencil order 2)")
@printf(
    "%3s %5s %8s %12s %12s %12s %12s %8s\n", "N", "F", "ρ", "λmax", "‖A_mpo‖_∞", "μ (rule)", "closed form", "μ/λmax"
)
for N in 2:4, F in (1.0, 10.0, 100.0)
    ρ = rho_from_floor(F, N)
    n = 15
    arch = TTArchitecture(ntuple(_ -> n, N))
    grid = Grid(arch, ntuple(_ -> 1.0, N))
    bcs = ntuple(_ -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), N)
    μ = anisotropy_mu_gershgorin(arch, grid, bcs, ρ)
    h = grid.dx[1]
    closed = N * (4 + ρ * (N - 1)) / h^2

    d2 = [onedim_matrix(2, arch, i, bcs[i]; stencil_order=2) ./ grid.dx[i]^2 for i in 1:N]
    d1 = [onedim_matrix(1, arch, i, bcs[i]; stencil_order=2) ./ grid.dx[i] for i in 1:N]
    dims = ntuple(_ -> n, N)
    A = -sum(kron_axis(Dict(i => d2[i]), dims) for i in 1:N)
    for l in 1:N, m in (l+1):N
        A -= 2ρ * kron_axis(Dict(l => d1[l], m => d1[m]), dims)
    end
    check(norm(A - A', Inf) < 1e-8 * norm(A, Inf), "anisotropy N=$N F=$F: sparse A not symmetric")
    λ = lanczos_max(A)

    norm_mpo = NaN
    if N <= 3
        # The sparse operator must be the MPO of the benchmark
        field = ScalarField(siteinds(arch), arch, BoundaryConditions(bcs...))
        Ampo = -1.0 * equicorrelated_laplacian(field, grid; correlation=ρ, stencil_order=2)
        Ad = mpo_matrix(Ampo)
        check(norm(Ad - A, Inf) < 1e-8 * norm(A, Inf), "anisotropy N=$N F=$F: sparse A differs from the MPO")
        norm_mpo = opnorm_inf(Ad)
        check(norm_mpo <= μ * (1 + 1e-12), "anisotropy N=$N F=$F: ‖A_mpo‖_∞ above the bound")
    end

    check(λ <= μ * (1 + 1e-10), "anisotropy N=$N F=$F: λmax = $λ above μ = $μ")
    check(μ <= 1.5λ, "anisotropy N=$N F=$F: μ/λmax = $(μ/λ) > 1.5")
    check(isapprox(μ, closed; rtol=1e-12), "anisotropy N=$N F=$F: μ = $μ differs from the closed form $closed")
    @printf("%3d %5g %8.4f %12.5g %12.5g %12.5g %12.5g %8.4f\n", N, F, ρ, λ, norm_mpo, μ, closed, μ / λ)
end

# Values of the paper runs (N = 10, 20; F = 1, 100): only the bound, against the closed form
println("\nAnisotropy, paper sizes (bound only)")
@printf("%3s %5s %12s %12s\n", "N", "F", "μ (rule)", "closed form")
for N in (10, 20), F in (1.0, 100.0)
    ρ = rho_from_floor(F, N)
    arch = TTArchitecture(ntuple(_ -> 15, N))
    grid = Grid(arch, ntuple(_ -> 1.0, N))
    bcs = ntuple(_ -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), N)
    μ = anisotropy_mu_gershgorin(arch, grid, bcs, ρ)
    closed = N * (4 + ρ * (N - 1)) * 14^2
    check(isapprox(μ, closed; rtol=1e-12), "anisotropy N=$N F=$F: μ = $μ differs from the closed form $closed")
    @printf("%3d %5g %12.5g %12.5g\n", N, F, μ, closed)
end

# ---- Chorin pressure operator ----
# The boundary conditions and grid of chorin_pipeline_stats/worker.jl (build_grid)
const BCSP = BoundaryConditions(
    BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0)), BoundaryCond1d(PeriodicBC(), PeriodicBC())
)
function chorin_field(nodes)
    arch = QTTArchitecture(nodes; interleaved=false)
    sites = siteinds(2, sum(nodes))
    grid = Grid(arch, (2.0, 1.0); periodic=DifferenTT.periodic_axes(BCSP))
    return ScalarField(zeros_mps(sites), arch, BCSP), grid, sites
end
const C8 = 205 / 72 + 2 * (8 / 5 + 1 / 5 + 8 / 315 + 1 / 560)

println("\nChorin pressure operator (QTT, stencil order 8)")
@printf("%9s %12s %12s %12s %12s %8s\n", "grid", "λmax", "‖A‖_∞", "μ (rule)", "closed form", "μ/λmax")
for k in (5, 6)
    nodes = (k, k)
    field, grid, sites = chorin_field(nodes)
    μ = laplacian_mu_gershgorin(field, grid; stencil_order=8)
    closed = C8 * sum(inv.(grid.dx .^ 2))

    # The QTT 1D matrices must be the TT ones (same closures, natural node order)
    tarch = TTArchitecture((2^k, 2^k))
    for i in 1:2
        q = onedim_matrix(2, field.arch, i, field.bcu[i]; stencil_order=8)
        t = onedim_matrix(2, tarch, i, field.bcu[i]; stencil_order=8)
        check(norm(q - t, Inf) < 1e-10, "chorin 2^$k axis $i: QTT 1D matrix differs from the TT one")
    end

    A = -1.0 * symmetrized_laplacian(field, grid; stencil_order=8)
    # MPO sites in order: x bits (most significant first) then y bits
    Ad = mpo_matrix(A)
    check(norm(Ad - Ad', Inf) < 1e-8 * norm(Ad, Inf), "chorin 2^$k: A not symmetric")
    λ = eigmax(Symmetric((Ad + Ad') / 2))
    check(opnorm_inf(Ad) <= μ * (1 + 1e-10), "chorin 2^$k: ‖A‖_∞ = $(opnorm_inf(Ad)) above the bound $μ")
    check(λ <= μ * (1 + 1e-10), "chorin 2^$k: λmax = $λ above μ = $μ")
    check(μ <= 1.5λ, "chorin 2^$k: μ/λmax = $(μ/λ) > 1.5")
    @printf("%4d×%-4d %12.5g %12.5g %12.5g %12.5g %8.4f\n", 2^k, 2^k, λ, opnorm_inf(Ad), μ, closed, μ / λ)
end

println("\nChorin, paper grids (bound only, closed form 6.5016 (h_x⁻² + h_y⁻²))")
@printf("%9s %12s %12s %10s\n", "grid", "μ (rule)", "closed form", "rel. dev.")
for k in (8, 10)
    field, grid, _ = chorin_field((k, k))
    μ = laplacian_mu_gershgorin(field, grid; stencil_order=8)
    closed = C8 * sum(inv.(grid.dx .^ 2))
    check(abs(μ / closed - 1) < 0.01, "chorin 2^$k: μ = $μ deviates by more than 1 % from the closed form $closed")
    @printf("%4d×%-4d %12.5g %12.5g %10.2e\n", 2^k, 2^k, μ, closed, μ / closed - 1)
end

if isempty(failures)
    println("\nAll checks passed")
else
    error("$(length(failures)) check(s) failed:\n" * join(failures, "\n"))
end
