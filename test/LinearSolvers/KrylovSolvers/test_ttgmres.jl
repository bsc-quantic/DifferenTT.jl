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
