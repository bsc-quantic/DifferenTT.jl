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
    A = replaceprime(DifferenTT.transpose_mpo(A)' * A, 2 => 1) # make it positive definite and hermitian
    b = random_mps(s)
    x0 = random_mps(s)

    solver = DifferenTT.GlobalCG(DifferenTT.GlobalUpdaterArgs(isposdef=true, ishermitian=true, maxiter=2))
    x = DifferenTT.linearsolve(solver, A, b, x0)

    solver_gmres = DifferenTT.GlobalGMRES(DifferenTT.GlobalUpdaterArgs(maxiter=2, krylovdim=2))
    x_g = DifferenTT.linearsolve(solver_gmres, A, b, x0)
end

include(joinpath(@__DIR__, "..", "..", "testutils.jl"))
using .TestUtils

# The global solvers record the iteration count and the convergence flag in the optional stores, and warn when they
# stop without converging
@testset "Global solvers record their Krylov info" begin
    A, sites, _, _ = tt_laplacian((4, 4))
    b = random_mps(rng(1), sites; linkdims=2)
    @testset "$name" for (name, make, M⁻¹) in (
        ("GlobalCG", args -> GlobalCG(; updater_kwargs=args), nothing),
        ("GlobalCG, preconditioned", args -> GlobalCG(; updater_kwargs=args), Preconditioner(identity_mpo(sites))),
        ("GlobalGMRES", args -> GlobalGMRES(; updater_kwargs=args), nothing),
        (
            "GlobalGMRES, preconditioned",
            args -> GlobalGMRES(; updater_kwargs=args),
            Preconditioner(identity_mpo(sites)),
        ),
    )
        solve(solver) = isnothing(M⁻¹) ? linearsolve(solver, A, b, nothing) : linearsolve(solver, A, b, M⁻¹, nothing)
        numiter, converged = Ref(-1), Ref(false)
        args = GlobalUpdaterArgs(;
            ishermitian=true, isposdef=true, rtol=1e-10, cutoff=1e-14, numiter_store=numiter, converged_store=converged
        )
        x = solve(make(args))
        @test converged[]
        @test numiter[] > 0
        @test relres(A, x, b) < 1e-7

        stalled = GlobalUpdaterArgs(;
            ishermitian=true, isposdef=true, rtol=1e-14, maxiter=1, krylovdim=2, converged_store=converged
        )
        @test_logs (:warn, r"without converging") match_mode=:any solve(make(stalled))
        @test !converged[]

        @test_throws ArgumentError solve(
            make(GlobalUpdaterArgs(; ishermitian=true, isposdef=true, numiter_store=1, maxiter=2))
        )
    end
end
