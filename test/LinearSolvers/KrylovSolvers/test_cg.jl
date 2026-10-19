using Test
using LinearAlgebra
using KrylovKit
using DifferenTT

@testset "Preconditioned CG" begin
    # Setup symmetric positive definite matrix A
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]
    # Simple preconditioner M
    M = [4.0 0.0; 0.0 3.0]
    Minv = inv(M)

    # Preconditioned CG
    alg = CG()
    x, info = linsolve(A, b, x0, Minv, alg)

    @test info.converged > 0
    @test A * x ≈ b

    # Test a0 and a1
    x2, info2 = linsolve(A, b, x0, Minv, alg, 1.0, -1.0)
    # solves (1.0 * I - (-1.0) * A) x = b => (I + A)x = b
    @test info2.converged > 0
    @test (I - A) * x2 ≈ b
end
