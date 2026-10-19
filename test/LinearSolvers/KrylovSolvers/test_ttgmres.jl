using Test
using ITensors
using ITensorMPS
using KrylovKit
using DifferenTT
using LinearAlgebra

include(joinpath(@__DIR__, "..", "..", "testutils.jl"))
using .TestUtils

@testset "TTGMRES directly" begin
    s = siteinds("S=1/2", 4)
    A = random_mpo(rng(1), s)
    b = random_mps(rng(2), s)
    x0 = random_mps(rng(3), s)
    Ad, bd = todense(A), todense(b)

    alg = DifferenTT.TTGMRES(maxiter=2, krylovdim=2, maxdim=10, cutoff=1e-8)
    x, info = KrylovKit.linsolve(A, b, x0, alg)

    # Two cycles of two Arnoldi steps do not solve a random 16 × 16 system
    @test info.numiter == 2
    @test info.converged == 0
    @test length(x) == 4

    # Test TTGMRES with a0 and a1: (I - A)x = b
    x2, info2 = KrylovKit.linsolve(A, b, x0, alg, 1.0, -1.0)
    @test info2.numiter == 2

    # The residual is recomputed at the end of every cycle, so the returned one is the true residual, and GMRES does
    # not increase it
    exact = DifferenTT.TTGMRES(maxiter=2, krylovdim=2, cutoff=1e-15, orth_cutoff=1e-15)
    x, info = KrylovKit.linsolve(A, b, x0, exact)
    @test info.normres ≈ norm(bd - Ad * todense(x)) rtol = 1e-6
    @test info.normres <= norm(bd - Ad * todense(x0))
    x2, info2 = KrylovKit.linsolve(A, b, x0, exact, 1.0, -1.0)
    @test info2.normres ≈ norm(bd - (I - Ad) * todense(x2)) rtol = 1e-6
    @test info2.normres <= norm(bd - (I - Ad) * todense(x0))
end

# Regression test for B2: the preconditioned TT-GMRES passed V[i]' to inner(v, ::Preconditioner, w), which primes v
# itself, so the Arnoldi step threw a site-index mismatch.
# The tolerances are 1e-7, not 1e-10: the solver applies A with ITensorMPS.apply, whose default density-matrix
# algorithm is accurate only to about √eps ≈ 1.5e-8 relative to the dense product, which sets the floor of the true
# residual even when the recurrence residual is at 1e-14.
@testset "Preconditioned GlobalGMRES matches the dense solution (B2)" begin
    A, sites, _, _ = tt_laplacian((8, 8))
    b = random_mps(rng(1), sites; linkdims=2)
    x_ref = todense(A) \ todense(b)
    solver = GlobalGMRES(;
        updater_kwargs=GlobalUpdaterArgs(; krylovdim=30, maxiter=20, rtol=1e-12, cutoff=1e-15, orth_cutoff=1e-15)
    )

    # A diagonal, non-trivial factor G (positive entries), so that GᵀG ≠ I
    Gdiag = MPO(sites)
    for (n, s) in enumerate(sites)
        Gdiag[n] = ITensor(Diagonal(collect(range(1.0, 2.0; length=dim(s)))), s', s)
    end

    @testset "$name" for (name, M⁻¹) in (
        ("Preconditioner(I)", Preconditioner(identity_mpo(sites))),
        ("FactorizedPreconditioner(I)", FactorizedPreconditioner(identity_mpo(sites))),
        ("FactorizedPreconditioner(diagonal G)", FactorizedPreconditioner(Gdiag)),
    )
        x = linearsolve(solver, A, b, M⁻¹, nothing)
        @test relres(A, x, b) < 1e-7
        @test norm(todense(x) - x_ref) <= 1e-7 * norm(x_ref)
    end
end

@testset "TTGMRES: nonsymmetric system, relative tolerance, early return and non-convergence" begin
    A, sites, _, _ = tt_laplacian((8, 8))
    # A nonsymmetric perturbation of the Laplacian, of bond dimension 1
    R = random_mpo(rng(4), sites)
    Ans = A + (0.05 * norm(todense(A)) / norm(todense(R))) * R
    Ad = todense(Ans)
    @test !issymmetric(Ad)
    b = random_mps(rng(1), sites; linkdims=2)
    x_ref = Ad \ todense(b)
    x0 = zeros_mps(sites)
    alg = DifferenTT.TTGMRES(; krylovdim=30, maxiter=20, tol=1e-6, cutoff=1e-15, orth_cutoff=1e-15)

    x, info = KrylovKit.linsolve(Ans, b, x0, alg)
    @test info.converged == 1
    @test norm(Ad * todense(x) - todense(b)) <= 2e-6 * norm(todense(b))
    # The error is bounded by κ(A) times the relative residual
    @test norm(todense(x) - x_ref) <= cond(Ad) * 2e-6 * norm(x_ref)

    # tol is relative to |b|
    x_scaled, info_scaled = KrylovKit.linsolve(Ans, 1e3 * b, x0, alg)
    @test info_scaled.numiter == info.numiter
    @test todense(x_scaled) ≈ 1e3 * todense(x) rtol = 1e-8

    _, info_exact = KrylovKit.linsolve(Ans, b, vec2mps(x_ref, sites), alg)
    @test info_exact.converged == 1
    @test info_exact.numiter == 0

    short = DifferenTT.TTGMRES(; krylovdim=2, maxiter=1, tol=1e-12)
    _, info_short = @test_logs KrylovKit.linsolve(Ans, b, x0, short)
    @test info_short.converged == 0
    @test info_short.numiter == 1
    @test_logs (:warn, r"TTGMRES linsolve stopped without converging") KrylovKit.linsolve(
        Ans, b, x0, DifferenTT.TTGMRES(; krylovdim=2, maxiter=1, tol=1e-12, verbosity=1)
    )
end

@testset "Non-converged preconditioned TTGMRES reports the unpreconditioned residual" begin
    A, sites, _, _ = tt_laplacian((8, 8))
    b = random_mps(rng(1), sites; linkdims=2)
    # M⁻¹ = 0.01 I, so that |M⁻¹r| and |r| differ by a factor 100
    M⁻¹ = Preconditioner(0.01 * identity_mpo(sites))
    short = DifferenTT.TTGMRES(; krylovdim=2, maxiter=1, tol=1e-12, cutoff=1e-15, orth_cutoff=1e-15)
    x, info = KrylovKit.linsolve(A, b, zeros_mps(sites), M⁻¹, short)
    @test info.converged == 0
    @test info.normres ≈ norm(todense(b) - todense(A) * todense(x)) rtol = 1e-6
    @test_logs (:warn, r"TTGMRES linsolve stopped without converging") match_mode = :any KrylovKit.linsolve(
        A, b, zeros_mps(sites), M⁻¹, DifferenTT.TTGMRES(; krylovdim=2, maxiter=1, tol=1e-12, verbosity=1)
    )
end
