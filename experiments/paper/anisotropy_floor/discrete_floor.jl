"""
    Rotated anisotropy benchmark: discrete floor and exact solution ranks (full format)

For small N, compares the analytic floor F = (1+(N-1)ρ)/(1-ρ) with the discrete floor κ(L0⁻¹ L)
on the benchmark grid, and measures the TT ranks of the exact solution of L u = 𝟏 (the default
rhs), which set the `maxdims_map` of the workers.

Everything is done in full format with 1D operators taken from DifferenTT, so the conventions
(grid spacing, stencils) are those of the benchmarks. L0⁻¹ is applied exactly through the
eigendecomposition of the 1D Laplacian, and the extreme eigenvalues of L0⁻¹ L come from the
Lanczos tridiagonal built from the L0-preconditioned CG coefficients (Saad 2003, §6.7).

Usage: julia --project=. experiments/paper/anisotropy_floor/discrete_floor.jl [N...]
"""

using LinearAlgebra
using Random
using Printf
using JLD2
using ITensors
using ITensorMPS
using TensorCrossInterpolation
using DifferenTT

include(joinpath(@__DIR__, "..", "..", "common", "anisotropy_problem.jl"))

# Keep in sync with the workers
const N_NODES_PER_DIM = 15
const floor_values = [1.0, 2.0, 4.0, 10.0, 30.0, 100.0]
const floor_dims = isempty(ARGS) ? [3, 4, 5] : parse.(Int, ARGS)
const rank_tols = [1e-4, 1e-6, 1e-8]

const out_folder = @__DIR__

"""
1D factors in the benchmark conventions: A = -d²/dx² and C = d/dx (centered).
"""
function onedim_factors(n::Int)
    arch = TTArchitecture((n,))
    grid = Grid(arch, (1.0,))
    bc = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    A = get_1d_A(1, grid, arch)
    C = DifferenTT.onedim_differential_operator(1, arch, 1, bc, Array; stencil_order=2) .* (1.0 / grid.dx[1])
    return Matrix(A), Matrix(C)
end

"""
Apply the 1D matrix `M` along mode `k` of the N-dimensional array `X`.
"""
function modemul(X::Array{Float64}, M::Matrix{Float64}, k::Int)
    N = ndims(X)
    perm = [k; setdiff(1:N, k)]
    Xp = permutedims(X, perm)
    sz = size(Xp)
    Y = reshape(M * reshape(Xp, sz[1], :), sz)
    return permutedims(Y, invperm(perm))
end

struct FullOperators
    A::Matrix{Float64}
    C::Matrix{Float64}
    Q::Matrix{Float64}      # eigenvectors of A
    invD::Array{Float64}    # 1 / (λ_i1 + ... + λ_iN)
end

function FullOperators(n::Int, N::Int)
    A, C = onedim_factors(n)
    E = eigen(Symmetric(A))
    invD = zeros(ntuple(_ -> n, N))
    for I in CartesianIndices(invD)
        invD[I] = 1 / sum(E.values[I[l]] for l in 1:N)
    end
    return FullOperators(A, C, E.vectors, invD)
end

"""
L = Σ A_ℓ - 2ρ Σ_{ℓ<m} C_ℓ C_m = Σ (A_ℓ + ρ C_ℓ²) - ρ (Σ C_ℓ)².
"""
function applyL(ops::FullOperators, X::Array{Float64}, ρ::Float64)
    N = ndims(X)
    Y = zero(X)
    for l in 1:N
        Y .+= modemul(X, ops.A, l)
    end
    iszero(ρ) && return Y
    GX = zero(X)
    for l in 1:N
        CX = modemul(X, ops.C, l)
        GX .+= CX
        Y .+= ρ .* modemul(CX, ops.C, l)
    end
    for l in 1:N
        Y .-= ρ .* modemul(GX, ops.C, l)
    end
    return Y
end

function applyL0inv(ops::FullOperators, X::Array{Float64})
    N = ndims(X)
    Y = X
    Qt = Matrix(ops.Q')
    for l in 1:N
        Y = modemul(Y, Qt, l)
    end
    Y = Y .* ops.invD
    for l in 1:N
        Y = modemul(Y, ops.Q, l)
    end
    return Y
end

"""
L0-preconditioned CG. Returns the solution, iteration count and the CG coefficients.
"""
function pcg(ops::FullOperators, b::Array{Float64}, ρ::Float64; tol::Float64, maxiter::Int=1000)
    x = zero(b);
    r = copy(b);
    z = applyL0inv(ops, r);
    p = copy(z);
    rz = dot(r, z)
    αs = Float64[];
    βs = Float64[]
    for it in 1:maxiter
        Ap = applyL(ops, p, ρ)
        α = rz / dot(p, Ap)
        x .+= α .* p
        r .-= α .* Ap
        z = applyL0inv(ops, r)
        rzn = dot(r, z)
        push!(αs, α);
        push!(βs, rzn / rz)
        norm(r) / norm(b) < tol && return x, it, αs, βs
        p .= z .+ (rzn / rz) .* p
        rz = rzn
    end
    return x, maxiter, αs, βs
end

"""
Ritz values of L0⁻¹ L from the Lanczos tridiagonal associated with the CG coefficients.
"""
function ritz_extremes(αs::Vector{Float64}, βs::Vector{Float64})
    m = length(αs)
    d = [1 / αs[1]; [1 / αs[j] + βs[j-1] / αs[j-1] for j in 2:m]]
    e = [sqrt(βs[j]) / αs[j] for j in 1:(m-1)]
    ev = eigvals(SymTridiagonal(d, e))
    return minimum(ev), maximum(ev)
end

"""
TT-SVD ranks of X with relative Frobenius accuracy `tol`.
"""
function tt_ranks(X::Array{Float64}, tol::Float64)
    N = ndims(X);
    sz = size(X)
    δ = tol / sqrt(N - 1) * norm(X)
    ranks = Int[]
    C = reshape(X, sz[1], :)
    r = 1
    for k in 1:(N-1)
        C = reshape(C, r * sz[k], :)
        F = svd(C)
        tail = sqrt.(reverse(cumsum(reverse(F.S .^ 2))))
        rk = max(1, something(findfirst(t -> t <= δ, tail), length(F.S) + 1) - 1)
        push!(ranks, rk)
        C = Diagonal(F.S[1:rk]) * F.Vt[1:rk, :]
        r = rk
    end
    return ranks
end

rows = Dict{String,Any}[]
for N in floor_dims
    ops = FullOperators(N_NODES_PER_DIM, N)
    for F in floor_values
        ρ = rho_from_floor(F, N)
        t = @elapsed begin
            # Discrete floor from a random start vector (excites the whole spectrum)
            Random.seed!(N)
            _, it_random, αs, βs = pcg(ops, randn(size(ops.invD)), ρ; tol=1e-10)
            λmin, λmax = ritz_extremes(αs, βs)
            # Exact solution for the default rhs f ≡ 1
            sol, it_ones, _, _ = pcg(ops, ones(size(ops.invD)), ρ; tol=1e-12)
            ranks = Dict(tol => tt_ranks(sol, tol) for tol in rank_tols)
        end
        row = Dict{String,Any}(
            "N" => N,
            "n" => N_NODES_PER_DIM,
            "F" => F,
            "rho" => ρ,
            "floor_analytic" => floor_from_rho(ρ, N),
            "lambda_min" => λmin,
            "lambda_max" => λmax,
            "floor_discrete" => λmax / λmin,
            "exact_L0_pcg_iters_random" => it_random,
            "exact_L0_pcg_iters_ones" => it_ones,
            "solution_ranks" => ranks,
        )
        push!(rows, row)
        @printf(
            "N=%d F=%7.1f ρ=%.4f  λ∈[%.4f, %.4f]  discrete floor %8.3f  iters(rand/ones) %3d/%3d  max rank %s  (%.1fs)\n",
            N,
            F,
            ρ,
            λmin,
            λmax,
            λmax / λmin,
            it_random,
            it_ones,
            join(["$(tol)=>$(maximum(ranks[tol]))" for tol in rank_tols], ", "),
            t
        )
        flush(stdout)
    end
end

# Merge with previous runs (e.g. a separate, slower N = 6 run)
out_file = joinpath(out_folder, "discrete_floor.jld2")
if isfile(out_file)
    previous = load(out_file, "rows")
    new_keys = Set((r["N"], r["F"]) for r in rows)
    rows = vcat(filter(r -> (r["N"], r["F"]) ∉ new_keys, previous), rows)
end
sort!(rows; by=r -> (r["N"], r["F"]))
save(out_file, "rows", rows)

open(joinpath(out_folder, "discrete_floor.md"), "w") do io
    println(io, "# Analytic vs discrete floor (n = $(N_NODES_PER_DIM)), exact solution ranks for f ≡ 1\n")
    println(
        io,
        "| N | F (analytic) | ρ | λ_min(L0⁻¹L) | λ_max(L0⁻¹L) | discrete floor | exact-L0 PCG iters (random / ones) | max TT rank @1e-4 / 1e-6 / 1e-8 |",
    )
    println(io, "|---|---|---|---|---|---|---|---|")
    for r in rows
        rk = join([maximum(r["solution_ranks"][tol]) for tol in rank_tols], " / ")
        @printf(
            io,
            "| %d | %g | %.4f | %.4f | %.4f | %.2f | %d / %d | %s |\n",
            r["N"],
            r["F"],
            r["rho"],
            r["lambda_min"],
            r["lambda_max"],
            r["floor_discrete"],
            r["exact_L0_pcg_iters_random"],
            r["exact_L0_pcg_iters_ones"],
            rk
        )
    end
end
println("Saved $out_file")
