using DifferenTT
using Test
using ITensors, ITensorMPS
using DifferenTT: ReducedPreconditioner, ReducedFactorizedPreconditioner

@testset "Projected Problem" begin
    n = 4
    s = siteinds("S=1/2", n)
    
    A = random_mpo(s)
    P = ProjMPO(A)
    
    # Test contract for AbstractProjMPO
    b = random_mps(s)
    position!(P, b, 2)
    H = ITensors.contract(P)
    @test H isa ITensor

    # Test contract for ReducedPreconditioner
    precond = DifferenTT.rank1_preconditioner(A)
    P_precond = ProjMPO(precond.operator)
    position!(P_precond, b, 2)
    red_precond = ReducedPreconditioner(P_precond)
    H_red_precond = ITensors.contract(red_precond)
    @test H_red_precond isa ITensor

    # Test contract for ReducedFactorizedPreconditioner
    red_fact_precond = ReducedFactorizedPreconditioner(P_precond)
    H_red_fact_precond = ITensors.contract(red_fact_precond)
    @test H_red_fact_precond isa ITensor
end
