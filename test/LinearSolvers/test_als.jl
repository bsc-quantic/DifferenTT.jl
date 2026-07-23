using DifferenTT
using Test
using ITensors, ITensorMPS
using DifferenTT: ALS

@testset "ALS Solver" begin
    n = 4
    s = siteinds("S=1/2", n)
    
    # Small random MPS/MPO
    A = random_mpo(s)
    b = random_mps(s)
    
    solver = ALS()
    x = linearsolve(solver, A, b, nothing; nsweeps=1)
    
    @test x isa MPS
    @test length(x) == n

    # With preconditioning
    M⁻¹ = DifferenTT.rank1_preconditioner(A) 
    x_precond = linearsolve(solver, A, b, M⁻¹, nothing; nsweeps=1)
    @test x_precond isa MPS
    @test length(x_precond) == n
end
