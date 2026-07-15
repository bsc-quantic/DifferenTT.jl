using DifferenTT
using Test
using Adapt
using ITensors
using ITensorMPS

# TODO Expand these test to ensure the adapt pipelines work as intended inside the rest of the methods
@testset "utils.jl" begin
    @testset "replace_siteinds!" begin
        nsite = 4
        old_sites = siteinds(2, nsite)
        new_sites = siteinds(2, nsite)
        mpo = random_mpo(old_sites)

        Pdetn.replace_siteinds!(mpo, new_sites)

        # After replacement, old site indices should be gone
        for i in 1:nsite
            @test new_sites[i] ∈ inds(mpo[i])
            @test new_sites[i]' ∈ inds(mpo[i])
        end
    end

    @testset "transpose" begin
        nsite = 4
        sites = siteinds(2, nsite)
        mpo = random_mpo(sites)
        transposed_mpo = transpose(mpo)
        @test transposed_mpo isa MPO
        @test all(transposed_mpo[i] == swapinds(mpo[i], siteind(mpo, i), siteind(mpo, i)') for i in 1:nsite)
    end

    @testset "reverse" begin
        nsite = 4
        sites = siteinds(2, nsite)
        mpo = random_mpo(sites)
        reversed_mpos = reverse(mpo)
        @test reversed_mpos isa MPO
        @test length(reversed_mpos) == nsite
        # Check connectivity
        for i in 1:nsite-1
            @test !isnothing(commonind(reversed_mpos[i], reversed_mpos[i+1]))
        end
        # Check that original MPO is not mutated and objects are different
        @test reversed_mpos[1] !== mpo[nsite]
    end
    
    @testset "_nothing_sum_handler" begin
        sites = siteinds(2, 4)
        a = random_mps(sites)
        b = random_mps(sites)

        # All nothing → nothing
        @test isnothing(Pdetn._nothing_sum_handler(nothing, nothing))

        # One element → returns that element
        result = Pdetn._nothing_sum_handler(a, nothing)
        @test result isa MPS

        # Two elements → sum
        result2 = Pdetn._nothing_sum_handler(a, b)
        @test result2 isa MPS
    end

end