using Test
using ITensors
using ITensorMPS
using KrylovKit
using DifferenTT
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "testutils.jl"))
using .TestUtils

@testset "TTCG directly" begin
    s = siteinds("S=1/2", 4)
    A = random_mpo(rng(1), s)
    A = replaceprime(DifferenTT.transpose_mpo(A)' * A, 2 => 1) # SPD
    b = random_mps(rng(2), s)
    x0 = random_mps(rng(3), s)
    Ad, bd = todense(A), todense(b)

    alg = DifferenTT.TTCG(maxiter=2, maxdim=10, cutoff=1e-8)
    x, info = KrylovKit.linsolve(A, b, x0, alg)

    # Two iterations do not reach tol = 1e-8
    @test info.numiter == 2
    @test info.converged == 0
    @test length(x) == 4

    # Test TTCG with a0 and a1: (I - A)x = b
    x2, info2 = KrylovKit.linsolve(A, b, x0, alg, 1.0, -1.0)
    @test info2.numiter == 2

    # Without truncation, the residual of the recurrence is the true one. (With cutoff = 1e-8 they drift apart by
    # 1e-5 to 3e-4 relative on this problem.)
    exact = DifferenTT.TTCG(maxiter=2, cutoff=1e-15, orth_cutoff=1e-15)
    x, info = KrylovKit.linsolve(A, b, x0, exact)
    @test info.normres ≈ norm(bd - Ad * todense(x)) rtol = 1e-6
    x2, info2 = KrylovKit.linsolve(A, b, x0, exact, 1.0, -1.0)
    @test info2.normres ≈ norm(bd - (I - Ad) * todense(x2)) rtol = 1e-6
end

# Regression test for B1: the step denominator of the shifted system (a₀I + a₁A)x = b used a₁⟨p,p⟩ instead of
# a₀⟨p,p⟩, so TT-CG diverged for a₀ ≠ 0 (relative residual 1.3e4 after 200 iterations on this problem).
# The tolerances are 1e-7, not 1e-10: the solver applies A with ITensorMPS.apply, whose default density-matrix
# algorithm is accurate only to about √eps ≈ 1.5e-8 relative to the dense product, which sets the floor of the true
# residual even when the recurrence residual is at 1e-14.
@testset "TTCG on a shifted SPD system (B1)" begin
    A, sites, _, _ = tt_laplacian((8, 8))
    b = random_mps(rng(1), sites; linkdims=2)
    x0 = random_mps(rng(2), sites; linkdims=2)
    a₀, a₁ = 1e4, 1.0
    Ad = todense(A)
    Ashift = a₀ * I + a₁ * Ad
    x_ref = Ashift \ todense(b)
    shifted_relres(x) = norm(Ashift * todense(x) - todense(b)) / norm(todense(b))

    alg = DifferenTT.TTCG(; maxiter=200, tol=1e-12, cutoff=1e-15, orth_cutoff=1e-15)
    x, info = KrylovKit.linsolve(A, b, x0, alg, a₀, a₁)
    @test info.converged == 1
    @test shifted_relres(x) < 1e-7
    @test norm(todense(x) - x_ref) <= 1e-7 * norm(x_ref)

    @testset "preconditioned, $name" for (name, M⁻¹) in (
        ("Preconditioner(I)", Preconditioner(identity_mpo(sites))),
        ("FactorizedPreconditioner(I)", FactorizedPreconditioner(identity_mpo(sites))),
    )
        x, info = KrylovKit.linsolve(A, b, x0, M⁻¹, alg, a₀, a₁)
        @test info.converged == 1
        @test shifted_relres(x) < 1e-7
        @test norm(todense(x) - x_ref) <= 1e-7 * norm(x_ref)
    end
end

@testset "TTCG: relative tolerance, early return and non-convergence" begin
    A, sites, _, _ = tt_laplacian((8, 8))
    b = random_mps(rng(1), sites; linkdims=2)
    x_ref = todense(A) \ todense(b)
    x0 = zeros_mps(sites)
    alg = DifferenTT.TTCG(; maxiter=200, tol=1e-6, cutoff=1e-15, orth_cutoff=1e-15)

    x, info = KrylovKit.linsolve(A, b, x0, alg)
    @test info.converged == 1
    @test info.normres <= 1e-6 * norm(b)
    @test norm(todense(x) - x_ref) <= 1e-5 * norm(x_ref)

    # tol is relative to |b|: scaling b scales the solution and leaves the iterations unchanged
    x_scaled, info_scaled = KrylovKit.linsolve(A, 1e3 * b, x0, alg)
    @test info_scaled.numiter == info.numiter
    @test todense(x_scaled) ≈ 1e3 * todense(x) rtol = 1e-8

    # Starting from the solution, it returns without iterating
    x_exact = vec2mps(x_ref, sites)
    _, info_exact = KrylovKit.linsolve(A, b, x_exact, alg)
    @test info_exact.converged == 1
    @test info_exact.numiter == 0

    # Without convergence it returns converged = 0, and warns only with verbosity ≥ 1
    short = DifferenTT.TTCG(; maxiter=2, tol=1e-12)
    _, info_short = @test_logs KrylovKit.linsolve(A, b, x0, short)
    @test info_short.converged == 0
    @test info_short.numiter == 2
    @test_logs (:warn, r"TTCG linsolve stopped without converging") KrylovKit.linsolve(
        A, b, x0, DifferenTT.TTCG(; maxiter=2, tol=1e-12, verbosity=1)
    )
    @test_logs (:info, r"TTCG linsolve starts") (:info, r"TTCG linsolve converged") match_mode = :any KrylovKit.linsolve(
        A, b, x0, DifferenTT.TTCG(; maxiter=200, tol=1e-6, verbosity=2)
    )
end

@testset "Preconditioned TTCG: early return and verbosity" begin
    A, sites, _, _ = tt_laplacian((4, 4))
    b = random_mps(rng(1), sites; linkdims=2)
    M⁻¹ = Preconditioner(identity_mpo(sites))
    x_exact = vec2mps(todense(A) \ todense(b), sites)
    alg(; kwargs...) = DifferenTT.TTCG(; maxiter=100, tol=1e-6, cutoff=1e-15, orth_cutoff=1e-15, kwargs...)
    _, info = KrylovKit.linsolve(A, b, x_exact, M⁻¹, alg())
    @test info.converged == 1 && info.numiter == 0
    @test_logs (:info, r"without any iterations") KrylovKit.linsolve(A, b, x_exact, M⁻¹, alg(; verbosity=2))
    @test_logs (:info, r"without any iterations") KrylovKit.linsolve(A, b, x_exact, alg(; verbosity=2))
    for M in (nothing, M⁻¹)
        args = isnothing(M) ? (A, b, zeros_mps(sites)) : (A, b, zeros_mps(sites), M)
        @test_logs (:info, r"starts") (:info, r"in iteration") (:info, r"converged") match_mode = :any KrylovKit.linsolve(
            args..., alg(; verbosity=3)
        )
        @test_logs (:warn, r"stopped without converging") match_mode = :any KrylovKit.linsolve(
            args..., alg(; maxiter=2, tol=1e-14, verbosity=1)
        )
    end
end
