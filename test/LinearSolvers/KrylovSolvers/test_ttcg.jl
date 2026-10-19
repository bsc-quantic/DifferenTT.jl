using Test
using ITensors
using ITensorMPS
using KrylovKit
using DifferenTT

@testset "TTCG directly" begin
    s = siteinds("S=1/2", 4)
    A = random_mpo(s)
    A = replaceprime(DifferenTT.transpose_mpo(A)' * A, 2 => 1) # SPD
    b = random_mps(s)
    x0 = random_mps(s)

    alg = DifferenTT.TTCG(maxiter=2, maxdim=10, cutoff=1e-8)
    x, info = KrylovKit.linsolve(A, b, x0, alg)

    @test info.numiter >= 0
    @test length(x) == 4

    # Test TTCG with a0 and a1
    x2, info2 = KrylovKit.linsolve(A, b, x0, alg, 1.0, -1.0)
    @test info2.numiter >= 0
end

include(joinpath(@__DIR__, "..", "..", "testutils.jl"))
using .TestUtils
using LinearAlgebra

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
