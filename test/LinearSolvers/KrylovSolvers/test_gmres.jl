using Test
using LinearAlgebra
using KrylovKit
using DifferenTT

@testset "Preconditioned GMRES" begin
    # Setup general matrix A
    A = [4.0 1.0; 2.0 3.0]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]
    # Simple preconditioner M
    M = [4.0 0.0; 0.0 3.0]
    Minv = inv(M)
    
    # Preconditioned GMRES
    alg = GMRES()
    x, info = linsolve(A, b, x0, Minv, alg)
    
    @test info.converged > 0
    @test A * x ≈ b
    
    # Test a0 and a1
    x2, info2 = linsolve(A, b, x0, Minv, alg, 1.0, -1.0)
    # solves (I - A)x = b
    @test info2.converged > 0
    @test (I - A) * x2 ≈ b
end
