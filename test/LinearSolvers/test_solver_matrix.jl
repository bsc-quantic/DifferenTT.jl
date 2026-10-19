using Test
using DifferenTT
using ITensors
using ITensorMPS
using LinearAlgebra
using DifferenTT: ResidualExpansion, positions_sum

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# Every solver × {no preconditioner, Preconditioner, FactorizedPreconditioner} on the SPD Dirichlet Laplacian of a
# 2⁴ × 2⁴ QTT grid (8 sites, 256 unknowns, κ ≈ 116), against the dense solution.
#
# The preconditioners are exact and dense-built, so that this file does not depend on the FAI construction (tested in
# test_preconditioners.jl): Preconditioner(A⁻¹), and FactorizedPreconditioner(G) with G = L⁻¹ for A = L Lᵀ, so that
# GᵀG = A⁻¹. With them, each solve must reach the same solution with fewer Krylov iterations.

A, sites, _, _ = qtt_laplacian((4, 4))
Ad = todense(A)
b = random_mps(rng(1), sites; linkdims=2)
x_ref = Ad \ todense(b)
L = cholesky(Symmetric(Ad)).L
preconditioners = (
    ("none", nothing),
    ("Preconditioner", Preconditioner(mat2mpo(Matrix(Symmetric(inv(Ad))), sites; cutoff=1e-14))),
    ("FactorizedPreconditioner", FactorizedPreconditioner(mat2mpo(Matrix(inv(L)), sites; cutoff=1e-14))),
)
relerr(x) = norm(todense(x) - x_ref) / norm(x_ref)
solve(solver, M⁻¹) = isnothing(M⁻¹) ? linearsolve(solver, A, b, nothing) : linearsolve(solver, A, b, M⁻¹, nothing)

@testset "DMRG-type solvers" begin
    rtol = 1e-7
    local_args(store) = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-10, numiter_store=store)
    sweeps(; kwargs...) = SweepingArgs(;
        nsweeps=8, maxdim=32, cutoff=1e-14, (observer!)=ResidualObserver(; rtol), kwargs...
    )
    makers = (
        ("ALS", store -> ALS(; sweep_kwargs=sweeps(), updater_kwargs=local_args(store))),
        (
            "ResidualALS",
            store ->
                ResidualALS(; sweep_kwargs=sweeps(; cutoff=0.0), updater_kwargs=local_args(store), residual_tol=1e-10),
        ),
        ("AMEn", store -> AMEn(; sweep_kwargs=sweeps(), updater_kwargs=local_args(store), residual_tol=1e-9)),
        (
            "ExpandedALS",
            store -> ExpandedALS(;
                sweep_kwargs=sweeps(; nsite=1),
                updater_kwargs=local_args(store),
                expander_func=ResidualExpansion(A, b; maxdim=4),
            ),
        ),
    )
    @testset "$name" for (name, make) in makers
        solutions = Dict{String,MPS}()
        iterations = Dict{String,Float64}()
        for (pname, M⁻¹) in preconditioners
            store = KrylovSweepStats()
            x = solve(make(store), M⁻¹)
            solutions[pname] = x
            iterations[pname] = sum(positions_sum(store))
            # The observer stops at rtol; its residual is evaluated by an expansion whose rounding error is about
            # 1.5e-8 relative to |b| (see ResidualObserver), hence the factor 2
            @test relres(A, x, b) < 2rtol
            @test relerr(x) < 1e-6
        end
        @testset "$pname agrees and needs fewer iterations" for pname in ("Preconditioner", "FactorizedPreconditioner")
            @test norm(todense(solutions[pname]) - todense(solutions["none"])) < 1e-6 * norm(x_ref)
            # Measured: about 2.2 (ALS, ResidualALS) to 2.8 (ExpandedALS) times fewer local iterations
            @test iterations[pname] < iterations["none"]
        end
    end
end

# The global solvers apply A with ITensorMPS.apply, whose density-matrix algorithm is accurate only to about
# √eps ≈ 1.5e-8, so with κ ≈ 116 their true residual stalls near 3e-7 (measured); rtol = 1e-6 stays above that floor.
@testset "Global solvers" begin
    makers = (
        ("GlobalCG", args -> GlobalCG(; updater_kwargs=args)),
        ("GlobalGMRES", args -> GlobalGMRES(; updater_kwargs=args)),
    )
    @testset "$name" for (name, make) in makers
        solutions = Dict{String,MPS}()
        iterations = Dict{String,Int}()
        for (pname, M⁻¹) in preconditioners
            numiter, converged = Ref(0), Ref(false)
            args = GlobalUpdaterArgs(;
                ishermitian=true,
                isposdef=true,
                rtol=1e-6,
                krylovdim=40,
                maxiter=100,
                cutoff=1e-14,
                orth_cutoff=1e-14,
                numiter_store=numiter,
                converged_store=converged,
            )
            x = solve(make(args), M⁻¹)
            solutions[pname] = x
            iterations[pname] = numiter[]
            @test converged[]
            # The stopping test uses the truncated residual of the recurrence; the dense one may exceed it slightly
            @test relres(A, x, b) < 2e-6
            @test relerr(x) < 1e-5
        end
        @testset "$pname agrees and needs fewer iterations" for pname in ("Preconditioner", "FactorizedPreconditioner")
            @test norm(todense(solutions[pname]) - todense(solutions["none"])) < 1e-5 * norm(x_ref)
            # An exact preconditioner converges in one iteration (one restart cycle for GMRES), up to truncation
            @test iterations[pname] <= iterations["none"]
            @test iterations[pname] <= 2
        end
    end
end
