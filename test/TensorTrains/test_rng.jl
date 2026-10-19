using DifferenTT
using Test
using ITensors
using ITensorMPS
using StableRNGs
using TensorCrossInterpolation

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# Every randomized routine takes an `rng` keyword: two calls with equally seeded generators give identical results,
# and generators with different seeds draw different sketches.
@testset "Reproducibility with an explicit rng" begin
    sites = siteinds("Qubit", 5)
    a = random_mps(rng(1), sites; linkdims=3)
    b = random_mps(rng(2), sites; linkdims=2)
    k = Index(4, "sketch")

    @test todense(tt_mul(a, b; rng=StableRNG(1))) == todense(tt_mul(a, b; rng=StableRNG(1)))
    @test todense(tt_mul(a, b; rng=StableRNG(1))) ≈ todense(a) .* todense(b) rtol = 1e-8

    S1 = DifferenTT.suffix_sketches(a, k; rng=StableRNG(1))
    S2 = DifferenTT.suffix_sketches(a, k; rng=StableRNG(1))
    @test all(S1[v] == S2[v] for v in eachindex(S1))
    @test S1[1] != DifferenTT.suffix_sketches(a, k; rng=StableRNG(2))[1]
    @test DifferenTT.sketch(a, 2:5, k; rng=StableRNG(1)) == DifferenTT.sketch(a, 2:5, k; rng=StableRNG(1))
    @test DifferenTT.positive_sketch(a, 2:5, k; rng=StableRNG(1)) ==
        DifferenTT.positive_sketch(a, 2:5, k; rng=StableRNG(1))

    # An entrywise-positive MPS for the positive-domain sketches of elementwise_function
    p = copy(a)
    for i in eachindex(p)
        p[i] = ITensor(abs.(array(p[i])) .+ 0.1, inds(p[i]))
    end
    for positive_domain in (true, false)
        f1 = elementwise_function(exp, p, positive_domain; rng=StableRNG(1))
        f2 = elementwise_function(exp, p, positive_domain; rng=StableRNG(1))
        @test todense(f1) == todense(f2)
        @test todense(f1) ≈ exp.(todense(p)) rtol = 1e-6
    end

    arch = QTTArchitecture((3, 3))
    grid = Grid(arch, (1.0, 1.0))
    g(x) = exp(-((x[1] - 0.3)^2 + (x[2] - 0.6)^2))
    @test TensorCrossInterpolation.optfirstpivot(Float64, g, grid, arch; iters=20, rng=StableRNG(1)) ==
        TensorCrossInterpolation.optfirstpivot(Float64, g, grid, arch; iters=20, rng=StableRNG(1))
    @test DifferenTT._tci_random_pivot([2, 4, 8]; rng=StableRNG(3)) ==
        DifferenTT._tci_random_pivot([2, 4, 8]; rng=StableRNG(3))
end
