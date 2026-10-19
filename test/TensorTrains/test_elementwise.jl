using DifferenTT
using Test
using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using StableRNGs

@testset "elementwise.jl" begin
    n = 4
    sites = siteinds("Qubit", n)

    @testset "tt_mul MPS-MPS" begin
        Random.seed!(1234)
        mps1 = random_mps(sites; linkdims=3)
        mps2 = random_mps(sites; linkdims=4)

        mps_prod = tt_mul(mps1, mps2)

        A1 = contract_mps(mps1)
        A2 = contract_mps(mps2)
        A_prod = contract_mps(mps_prod)

        @test A_prod ≈ A1 .* A2

        mps_prod_direct = DifferenTT._direct_tt_mul(mps1, mps2)
        @test contract_mps(mps_prod_direct) ≈ A1 .* A2
    end

    @testset "tt_mul MPO-MPO" begin
        Random.seed!(1234)
        mpo1 = random_mpo(sites)
        mpo2 = random_mpo(sites)

        mpo_prod = tt_mul(mpo1, mpo2)

        A1 = contract_mpo(mpo1)
        A2 = contract_mpo(mpo2)
        A_prod = contract_mpo(mpo_prod)

        @test A_prod ≈ A1 .* A2
    end

    @testset "tt_mul MPS-MPO and MPO-MPS" begin
        Random.seed!(1234)
        mps = random_mps(sites; linkdims=2)
        mpo = random_mpo(sites)

        mpo_prod1 = tt_mul(mps, mpo)
        mpo_prod2 = tt_mul(mpo, mps)

        T_prod1 = contract_mpo(mpo_prod1)
        T_prod2 = contract_mpo(mpo_prod2)

        @test size(T_prod1) == size(contract_mpo(mpo))
        @test size(T_prod2) == size(contract_mpo(mpo))

        @test T_prod1 ≈ T_prod2

        @test mpo_prod1 isa MPO
        @test mpo_prod2 isa MPO
    end

    @testset "tt_mul MPS-MPS truncation and guards" begin
        # Smooth QTT operands, so that the exact product has a decaying spectrum and a rank cap
        # is a meaningful test of *how* the product truncates.
        nq = 10
        qsites = siteinds(2, nq)
        xs = range(0, 1; length=2^nq)
        m1 = vec2mps(collect(@. exp(-3xs) * sin(7xs) + 0.3), qsites; cutoff=1e-14)
        m2 = vec2mps(collect(@. cos(5xs) / (1 + xs^2)), qsites; cutoff=1e-14)

        exact = DifferenTT._direct_tt_mul(m1, m2; cutoff=0.0)
        A_exact = contract_mps(exact)
        relerr(m) = norm(contract_mps(m) - A_exact) / norm(A_exact)

        # The cross must not truncate internally: capping it afterwards by SVD has to land within
        # a small factor of the optimal rank-r truncation of the exact product.
        for r in (4, 6)
            capped = tt_mul(m1, m2; maxdim=r)
            optimal = truncate!(copy(exact); maxdim=r, cutoff=0.0)
            @test maxlinkdim(capped) <= r
            @test relerr(capped) <= 1.5 * relerr(optimal) + 1e-9
        end

        # The output is recompressed: finite, orthogonal, and no larger than the exact product.
        prod_mps = tt_mul(m1, m2)
        @test DifferenTT.allfinite(prod_mps)
        @test ITensorMPS.ortho_lims(prod_mps) == 1:1
        @test maxlinkdim(prod_mps) <= maxlinkdim(exact)
        @test relerr(prod_mps) < 1e-6

        # A relative pivot cutoff makes the product invariant under a rescaling of a factor.
        scaled = tt_mul(1e-8 * m1, m2; maxdim=6)
        reference = 1e-8 * contract_mps(tt_mul(m1, m2; maxdim=6))
        @test norm(contract_mps(scaled) - reference) / norm(reference) < 1e-6

        # A zero operand short-circuits instead of sketching a zero block.
        @test norm(tt_mul(0.0 * m1, m2)) == 0.0

        # A non-finite cross output triggers the fallback, which is counted.
        reset_tt_mul_fallbacks!()
        @test tt_mul_fallbacks() == 0
        nan_mps = copy(m1)
        nan_mps[3] = ITensor(fill(NaN, size(array(nan_mps[3]))), inds(nan_mps[3]))
        try
            tt_mul(nan_mps, m2)
        catch
            # `_direct_tt_mul` is free to throw on NaN input; only the fallback matters here.
        end
        @test tt_mul_fallbacks() == 1
        reset_tt_mul_fallbacks!()
    end

    @testset "pivot indices" begin
        pivs = [[1 => 2, 2 => 1], [1 => 1, 2 => 2], [1 => 2, 2 => 2]]
        i = DifferenTT.pivot_index(pivs; tags="Link,l=1")
        @test i isa Index{<:DifferenTT.PivotSpace}
        @test dim(i) == 3
        @test !ITensors.hasqns(i)
        @test DifferenTT.haspivots(i)
        @test !DifferenTT.haspivots(Index(3))
        @test DifferenTT.index_pivots(i) == pivs
        @test DifferenTT.index_pivots(copy(i)) == pivs
        @test DifferenTT.index_pivots(i') == pivs
        @test startswith(sprint(show, i), "(PI|dim=3|")
        @test occursin("Link,l=1", sprint(show, i))
        T = random_itensor(StableRNG(1), i, Index(2))
        @test norm(T) > 0
    end

    @testset "uniform_itensor" begin
        i, j = Index(3, "i"), Index(4, "j")
        T = DifferenTT.uniform_itensor(StableRNG(1), i, j)
        @test hassameinds(T, (i, j))
        A = array(T, i, j)
        @test all(0 .<= A .< 1)
        @test A == array(DifferenTT.uniform_itensor(StableRNG(1), i, j), i, j)
        @test A != array(DifferenTT.uniform_itensor(StableRNG(2), i, j), i, j)
        @test all(0 .<= array(DifferenTT.uniform_itensor(i, j), i, j) .< 1)
    end

    @testset "suffix_sketches" begin
        Random.seed!(1234)
        mps = random_mps(sites; linkdims=3)
        k = Index(5, "sketch")
        sketches = DifferenTT.suffix_sketches(mps, k)

        @test length(sketches) == n + 1
        @test sketches[n+1] == ITensor(1.0)
        # Every cached suffix carries the sketch index and the link index at its left edge, i.e.
        # exactly the indices `sketch` returns for the same region.
        for v in 2:n
            @test issetequal(inds(sketches[v]), (k, linkind(mps, v - 1)))
            @test issetequal(inds(sketches[v]), inds(DifferenTT.sketch(mps, v:n, k)))
        end
    end

    @testset "elementwise_function" begin
        Random.seed!(1234)
        mps = random_mps(sites; linkdims=2)

        for i in 1:n
            mps[i] = ITensor(abs.(array(mps[i])) .+ 0.1, inds(mps[i]))
        end

        func = x -> x^2
        mps_mapped = elementwise_function(func, mps, true; maxdim=20, oversample=10)

        A_mps = contract_mps(mps)
        A_mapped = contract_mps(mps_mapped)

        @test isapprox(A_mapped, func.(A_mps); rtol=0.5, atol=0.5)
    end
end
