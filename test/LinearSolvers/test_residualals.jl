using DifferenTT
using Test
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "ResidualALS Solver" begin
    n = 4
    sites = siteinds("S=1/2", n)
    A = MPO(sites, "Id")
    b = random_mps(sites; linkdims=2)

    @testset "ResidualALS instantiation" begin
        solver = ResidualALS()
        @test solver.residual_tol == 1e-12
        @test solver.residual_damp == 1.1
        @test typeof(solver.updater) == DifferenTT.KrylovUpdater
    end

    @testset "Standard linearsolve" begin
        solver = ResidualALS(sweep_kwargs=DifferenTT.SweepingArgs(nsweeps=2, maxdim=4, cutoff=1e-8), residual_tol=1e-8)
        x = linearsolve(solver, A, b, nothing)
        @test x isa MPS
        @test norm(apply(A, x) - b) < 1e-6
    end

    @testset "Preconditioned linearsolve" begin
        solver = ResidualALS(sweep_kwargs=DifferenTT.SweepingArgs(nsweeps=2, maxdim=4, cutoff=1e-8), residual_tol=1e-8)
        M_prec = rank1_preconditioner(A)
        x = linearsolve(solver, A, b, M_prec, nothing)
        @test x isa MPS
        @test norm(apply(A, x) - b) < 1e-6
    end

    @testset "Local Preconditioned linearsolve" begin
        solver = ResidualALS(sweep_kwargs=DifferenTT.SweepingArgs(nsweeps=2, maxdim=4, cutoff=1e-8), residual_tol=1e-8)
        M_prec = rank1_preconditioner(A)
        x = linearsolve(solver, A, b, M_prec, nothing; preconditioner_sites=2:3)
        @test x isa MPS
        @test norm(apply(A, x) - b) < 1e-6
    end
end
