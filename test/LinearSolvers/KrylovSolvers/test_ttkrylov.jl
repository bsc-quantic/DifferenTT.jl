using Test
using ITensors
using ITensorMPS
using KrylovKit
using DifferenTT

@testset "TTCG and TTGMRES types and GlobalSolvers" begin
    ttcg = DifferenTT.TTCG(maxiter=50)
    @test ttcg.maxiter == 50
    @test ttcg isa KrylovKit.KrylovAlgorithm

    ttgmres = DifferenTT.TTGMRES(maxiter=40)
    @test ttgmres.maxiter == 40
    @test ttgmres isa KrylovKit.KrylovAlgorithm

    global_cg = DifferenTT.GlobalCG(DifferenTT.GlobalUpdaterArgs(isposdef=true, ishermitian=true))
    @test global_cg isa DifferenTT.AbstractGlobalSolver

    global_gmres = DifferenTT.GlobalGMRES(DifferenTT.GlobalUpdaterArgs())
    @test global_gmres isa DifferenTT.AbstractGlobalSolver
end

@testset "TTCG and TTGMRES solvers with minimal MPO" begin
    s = siteinds("S=1/2", 4)
    A = random_mpo(s)
    A = replaceprime(transpose(A)' * A, 2 => 1) # make it positive definite and hermitian
    b = random_mps(s)
    x0 = random_mps(s)

    solver = DifferenTT.GlobalCG(DifferenTT.GlobalUpdaterArgs(isposdef=true, ishermitian=true, maxiter=2))
    x = DifferenTT.linearsolve(solver, A, b, x0)

    solver_gmres = DifferenTT.GlobalGMRES(DifferenTT.GlobalUpdaterArgs(maxiter=2, krylovdim=2))
    x_g = DifferenTT.linearsolve(solver_gmres, A, b, x0)
end
