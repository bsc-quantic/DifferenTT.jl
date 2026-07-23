using Test
using LinearAlgebra
using KrylovKit
using DifferenTT

@testset "preconditioned linsolve selector" begin
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    x0 = [0.0, 0.0]
    Minv = inv([4.0 0.0; 0.0 3.0])
    
    # Test default selector with Minv
    x, info = linsolve(A, b, x0, Minv)
    @test info.converged > 0
    @test A * x ≈ b
    
    A_asym = [4.0 1.0; 2.0 3.0]
    x_asym, info_asym = linsolve(A_asym, b, x0, Minv)
    @test info_asym.converged > 0
    @test A_asym * x_asym ≈ b
end
