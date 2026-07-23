using DifferenTT
using Test
using ITensors, ITensorMPS
using DifferenTT: AMEn

@testset "AMEn Solver" begin
    n = 4
    s = siteinds("S=1/2", n)
    
    A = random_mpo(s)
    b = random_mps(s)
    
    solver = AMEn()
    @test_throws ErrorException("AMEn solver not yet implemented") linearsolve(solver, A, b, nothing)
    
    M⁻¹ = DifferenTT.rank1_preconditioner(A)
    @test_throws ErrorException("AMEn solver not yet implemented") linearsolve(solver, A, b, M⁻¹, nothing)
end
