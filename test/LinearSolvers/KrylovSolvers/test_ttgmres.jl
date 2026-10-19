using Test
using ITensors
using ITensorMPS
using KrylovKit
using DifferenTT

@testset "TTGMRES directly" begin
    s = siteinds("S=1/2", 4)
    A = random_mpo(s)
    b = random_mps(s)
    x0 = random_mps(s)

    alg = DifferenTT.TTGMRES(maxiter=2, krylovdim=2, maxdim=10, cutoff=1e-8)
    x, info = KrylovKit.linsolve(A, b, x0, alg)

    @test info.numiter >= 0
    @test length(x) == 4

    # Test TTGMRES with a0 and a1
    x2, info2 = KrylovKit.linsolve(A, b, x0, alg, 1.0, -1.0)
    @test info2.numiter >= 0
end

include(joinpath(@__DIR__, "..", "..", "testutils.jl"))
using .TestUtils
using LinearAlgebra

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
