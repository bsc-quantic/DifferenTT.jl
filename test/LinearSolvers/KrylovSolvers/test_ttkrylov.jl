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

include(joinpath(@__DIR__, "..", "..", "testutils.jl"))
using .TestUtils
using LinearAlgebra

@testset "TTCG and TTGMRES solvers with minimal MPO" begin
    s = siteinds("S=1/2", 4)
    A = random_mpo(rng(1), s)
    A = replaceprime(DifferenTT.transpose_mpo(A)' * A, 2 => 1) # make it positive definite and hermitian
    b = random_mps(rng(2), s)
    x0 = random_mps(rng(3), s)
    Ad, bd = todense(A), todense(b)
    x_ref = Ad \ bd
    energy_error(x) = (e=todense(x) - x_ref; sqrt(dot(e, Ad * e)))

    # Two CG iterations do not solve the system, but they reduce the A-norm of the error, which CG minimizes. The
    # warnings for not converging are expected.
    solver = DifferenTT.GlobalCG(DifferenTT.GlobalUpdaterArgs(isposdef=true, ishermitian=true, maxiter=2))
    x = quietly(() -> DifferenTT.linearsolve(solver, A, b, x0))
    @test x isa MPS
    @test energy_error(x) < energy_error(x0)

    # GMRES does not increase the residual
    solver_gmres = DifferenTT.GlobalGMRES(DifferenTT.GlobalUpdaterArgs(maxiter=2, krylovdim=2))
    x_g = quietly(() -> DifferenTT.linearsolve(solver_gmres, A, b, x0))
    @test x_g isa MPS
    @test norm(bd - Ad * todense(x_g)) < norm(bd - Ad * todense(x0))
end

@testset "GlobalCG requires an SPD operator" begin
    @test_throws ArgumentError GlobalCG(GlobalUpdaterArgs())
    @test_throws ArgumentError GlobalCG(; updater_kwargs=GlobalUpdaterArgs(; ishermitian=true))
    @test GlobalCG().updater_kwargs.isposdef && GlobalCG().updater_kwargs.ishermitian
    @test GlobalGMRES(; updater_kwargs=GlobalUpdaterArgs(; krylovdim=7)).updater_kwargs.krylovdim == 7
end

# M⁻¹v and ⟨v, M⁻¹w⟩ of both preconditioner types, against dense products. FactorizedPreconditioner(G) stands for GᵀG.
@testset "apply and inner of the preconditioners" begin
    sites = siteinds(2, 4)
    M = random_mpo(rng(1), sites) + random_mpo(rng(2), sites)
    Md = todense(M)
    v = random_mps(rng(3), sites; linkdims=2)
    w = random_mps(rng(4), sites; linkdims=2)
    vd, wd = todense(v), todense(w)
    @testset "$name" for (name, P, Pd) in (
        ("Preconditioner", Preconditioner(M), Md), ("FactorizedPreconditioner", FactorizedPreconditioner(M), Md' * Md)
    )
        @test todense(DifferenTT.apply(P, v)) ≈ Pd * vd rtol = 1e-10
        @test inner(v, P, w) ≈ dot(vd, Pd * wd) rtol = 1e-10
        @test KrylovKit.inner(v, P, w) ≈ dot(vd, Pd * wd) rtol = 1e-10
    end
    # GᵀG is positive semidefinite: ⟨v, GᵀG v⟩ = |Gv|² ≥ 0
    @test inner(v, FactorizedPreconditioner(M), v) ≈ norm(Md * vd)^2 rtol = 1e-10
end

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
