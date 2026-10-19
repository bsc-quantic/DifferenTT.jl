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

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# The Frobenius floor of the residual-based truncation follows the ITensors convention: with a residual tolerance so
# loose that the residual alone would keep rank 1, the rank chosen is the one that ITensors' svd keeps for the same
# cutoff (discarded relative weight Σσ² at most cutoff)
@testset "The cutoff floor follows the ITensors convention" begin
    A, sites, _, _ = qtt_laplacian((6,))
    b = random_mps(rng(1), sites; linkdims=4)
    state = orthogonalize(random_mps(rng(2), sites; linkdims=8), 3)
    problem = ITensorMPS.ReducedLinearProblem(A, b)
    ITensorMPS.set_nsite!(problem, 2)
    ITensorMPS.position!(problem, state, 3)

    # A two-site tensor with singular values 10⁰, 10⁻¹, …, so that the discarded weights are well separated
    two_site = state[3] * state[4]
    left = commoninds(two_site, state[3])
    U, S, V = svd(two_site, left)
    u, v = inds(S)
    σ = [10.0^(1 - k) for k in 1:dim(u)]
    phi = U * ITensor(LinearAlgebra.Diagonal(σ), u, v) * V

    @testset "cutoff = $cutoff" for cutoff in (1e-3, 1e-6, 1e-9)
        result = DifferenTT.residual_replacebond!(
            copy(state), 3, phi, problem; cutoff, resid_tol=1e6, resid_damp=1.0, ortho="left"
        )
        _, S_cut, _ = svd(phi, left; cutoff)
        itensors_rank = dim(first(inds(S_cut)))
        @test result.chosen_rank == itensors_rank
        @test result.chosen_rank < length(σ)
    end
end
