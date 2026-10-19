using DifferenTT
using Test
using ITensors
using ITensorMPS
using StableRNGs

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# Every randomized routine takes an `rng` keyword: two calls with equally seeded generators give identical results.
@testset "Reproducibility with an explicit rng" begin
    A, sites, _, _ = tt_laplacian((4, 4))
    als = ALS(; sweep_kwargs=SweepingArgs(; nsweeps=2, maxdim=4, cutoff=1e-12))

    fai(seed) = fai_preconditioner(A; μ=1e7, solver=als, totalmaxdim=4, rng=StableRNG(seed))
    @test todense(fai(1).operator) == todense(fai(1).operator)

    lsp(seed) = linsolve_preconditioner(A, als; rng=StableRNG(seed))
    @test todense(lsp(1).operator) == todense(lsp(1).operator)

    state = random_mps(rng(1), sites; linkdims=2)
    for E in (DifferenTT.RandomSPSDexpansion, DifferenTT.RandomLTexpansion)
        @test todense(E(; rng=StableRNG(1))(state)) == todense(E(; rng=StableRNG(1))(state))
        @test todense(E(; rng=StableRNG(1))(state)) != todense(E(; rng=StableRNG(2))(state))
    end
end
