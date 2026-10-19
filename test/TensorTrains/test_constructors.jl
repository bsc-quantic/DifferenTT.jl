using DifferenTT
using ITensors
using ITensorMPS
using Test
using LinearAlgebra

@testset "constructors.jl" begin
    @testset "make_identity_site" begin
        # Edge site (one link)
        s = Index(2, "Site")
        l = Index(1, "Link")
        T_edge = DifferenTT.make_identity_site(s, l, Array{Float64})

        @test hasinds(T_edge, s, s', l)
        @test eltype(T_edge) == Float64
        @test T_edge[s=>1, s'=>1, l=>1] == 1.0
        @test T_edge[s=>2, s'=>2, l=>1] == 1.0
        @test T_edge[s=>1, s'=>2, l=>1] == 0.0
        @test T_edge[s=>2, s'=>1, l=>1] == 0.0

        # Bulk site (two links)
        s_bulk = Index(3, "Site")
        l1 = Index(1, "Link,l=1")
        l2 = Index(1, "Link,l=2")
        T_bulk = DifferenTT.make_identity_site(s_bulk, l1, l2, Array{Float64})

        @test hasinds(T_bulk, s_bulk, s_bulk', l1, l2)
        @test eltype(T_bulk) == Float64
        for i in 1:3
            for j in 1:3
                if i == j
                    @test T_bulk[s_bulk=>i, s_bulk'=>j, l1=>1, l2=>1] == 1.0
                else
                    @test T_bulk[s_bulk=>i, s_bulk'=>j, l1=>1, l2=>1] == 0.0
                end
            end
        end
    end

    @testset "identity_mpo" begin
        # With site indices
        sites = [Index(2, "Site,n=$i") for i in 1:4]
        mpo1 = identity_mpo(sites)
        @test mpo1 isa MPO
        @test length(mpo1) == 4
        @test maxlinkdim(mpo1) == 1

        # With site dimensions
        dims = [2, 3, 2, 4]
        mpo2 = identity_mpo(dims)
        @test mpo2 isa MPO
        @test length(mpo2) == 4
        @test maxlinkdim(mpo2) == 1
    end
    @testset "zeros_mpo" begin
        # With site indices
        sites = [Index(2, "Site,n=$i") for i in 1:4]
        mpo1 = zeros_mpo(sites)
        @test mpo1 isa MPO
        @test length(mpo1) == 4
        @test maxlinkdim(mpo1) == 1
        @test norm(mpo1) == 0.0

        # With site dimensions and custom linkdims
        dims = [2, 3, 2, 4]
        mpo2 = zeros_mpo(dims)
        @test mpo2 isa MPO
        @test length(mpo2) == 4
        @test maxlinkdim(mpo2) == 1
        @test norm(mpo2) == 0.0
    end

    @testset "zeros_mps" begin
        # With site indices
        sites = [Index(2, "Site,n=$i") for i in 1:4]
        mps1 = zeros_mps(sites)
        @test mps1 isa MPS
        @test length(mps1) == 4
        @test maxlinkdim(mps1) == 1
        @test norm(mps1) == 0.0

        # With site dimensions and custom linkdims
        dims = [2, 3, 2, 4]
        mps2 = zeros_mps(dims; linkdims=2)
        @test mps2 isa MPS
        @test length(mps2) == 4
        @test maxlinkdim(mps2) == 2
        @test norm(mps2) == 0.0
    end

    @testset "ones_mps" begin
        # With site indices and bond indices
        sites = [Index(2, "Site,n=$i") for i in 1:4]
        bonds = [Index(3, "Link,l=$i") for i in 1:3]
        mps1 = ones_mps(sites, bonds)
        @test mps1 isa MPS
        @test length(mps1) == 4
        @test maxlinkdim(mps1) == 3
        # Every entry should be 1.0, so the norm of the first tensor (2x3) should be sqrt(6)
        @test norm(mps1[1]) ≈ sqrt(2*3)
        # Second tensor (3x2x3) should be sqrt(18)
        @test norm(mps1[2]) ≈ sqrt(3*2*3)

        # With site indices and bond dim
        mps2 = ones_mps(sites, 2)
        @test mps2 isa MPS
        @test length(mps2) == 4
        @test maxlinkdim(mps2) == 2
        @test norm(mps2[1]) ≈ sqrt(2*2) # 2.0

        # With site dimensions and bond dim
        dims = [2, 3, 2, 4]
        mps3 = ones_mps(dims, 2)
        @test mps3 isa MPS
        @test length(mps3) == 4
        @test maxlinkdim(mps3) == 2
        @test norm(mps3[1]) ≈ sqrt(2*2)
        @test norm(mps3[2]) ≈ sqrt(2*3*2)
    end

    @testset "ones_mpo" begin
        # With site indices and bond indices
        sites = [Index(2, "Site,n=$i") for i in 1:4]
        bonds = [Index(3, "Link,l=$i") for i in 1:3]
        mpo1 = ones_mpo(sites, bonds)
        @test mpo1 isa MPO
        @test length(mpo1) == 4
        @test maxlinkdim(mpo1) == 3
        # Every entry is 1.0. For the first tensor: dim is (site, site', link) -> 2x2x3 = 12
        @test norm(mpo1[1]) ≈ sqrt(12)
        # Second tensor: link_prev x site x site' x link_next -> 3x2x2x3 = 36
        @test norm(mpo1[2]) ≈ sqrt(36)

        # With site indices and bond dim
        mpo2 = ones_mpo(sites, 2)
        @test mpo2 isa MPO
        @test length(mpo2) == 4
        @test maxlinkdim(mpo2) == 2
        @test norm(mpo2[1]) ≈ sqrt(2*2*2) # 8

        # With site dimensions and bond dim
        dims = [2, 3, 2, 4]
        mpo3 = ones_mpo(dims, 2)
        @test mpo3 isa MPO
        @test length(mpo3) == 4
        @test maxlinkdim(mpo3) == 2
        @test norm(mpo3[1]) ≈ sqrt(2*2*2) # dim 2 -> 2x2x2 = 8
        @test norm(mpo3[2]) ≈ sqrt(2*3*3*2) # dim 3 -> 2 x 3 x 3 x 2 = 36
    end

    @testset "arrays2mps" begin
        nsite = 4
        sites = siteinds(2, nsite)
        mps_arrays = [
            rand(2, 3), # Left end
            rand(2, 3, 5), # Bulk
            rand(2, 5, 4), # Bulk
            rand(2, 4), # Right end
        ]

        mps = DifferenTT.arrays2mps(sites, mps_arrays)
        @test mps isa MPS
        @test length(mps) == nsite
        @test siteind(mps, 1) == sites[1]
        @test linkdim(mps, 1) == 3
        @test linkdim(mps, 2) == 5
        @test linkdim(mps, 3) == 4

        # Test error conditions
        @test_throws DimensionMismatch DifferenTT.arrays2mps(siteinds(2, 3), mps_arrays) # Mismatched length
        bad_mps_arrays = [rand(2, 2), rand(2, 3)] # Mismatched bond dim
        @test_throws DimensionMismatch DifferenTT.arrays2mps(siteinds(2, 2), bad_mps_arrays)
    end

    @testset "arrays2mpo" begin
        nsite = 3
        sites = siteinds(2, nsite)
        mpo_arrays = [
            rand(2, 2, 4), # Left end
            rand(2, 2, 4, 5), # Bulk
            rand(2, 2, 5),  # Right end
        ]

        mpo = DifferenTT.arrays2mpo(sites, mpo_arrays)
        @test mpo isa MPO
        @test length(mpo) == nsite
        @test siteind(mpo, 2) == sites[2]
        @test linkdim(mpo, 1) == 4
        @test linkdim(mpo, 2) == 5

        # Test error conditions
        @test_throws DimensionMismatch DifferenTT.arrays2mpo(siteinds(2, 2), mpo_arrays) # Mismatched length
        bad_mpo_arrays = [rand(2, 2, 2), rand(2, 2, 3)] # Mismatched bond dim
        @test_throws DimensionMismatch DifferenTT.arrays2mpo(siteinds(2, 2), bad_mpo_arrays)
    end

    @testset "vec2tensor" begin
        # 4-element vector → 2×2 tensor
        v4 = [1.0, 2.0, 3.0, 4.0]
        t4 = DifferenTT.vec2tensor(v4, [2, 2])
        @test size(t4) == (2, 2)

        # 8-element vector → 2×2×2 tensor
        v8 = collect(1.0:8.0)
        t8 = DifferenTT.vec2tensor(v8, [2, 2, 2])
        @test size(t8) == (2, 2, 2)

        # 16-element vector → 2×2×2×2 tensor, or 2x8 tensor
        v16 = collect(1.0:16.0)
        t16 = DifferenTT.vec2tensor(v16, [2, 2, 2, 2])
        @test size(t16) == (2, 2, 2, 2)

        t16_v2 = DifferenTT.vec2tensor(v16, [2, 8])
        @test size(t16_v2) == (8, 2) # Leg order is reversed in order to account for end minor ordering
    end

    @testset "vec2mps" begin
        n = 4
        sites = siteinds(2, n)
        v = rand(2^n)

        mps = DifferenTT.vec2mps(v, sites)
        @test mps isa MPS
        @test length(mps) == n
        @test siteinds(mps) == sites

        @test DifferenTT.contract_mps(DifferenTT.vec2mps(v, sites)) ≈ v
    end

    @testset "mat2tensor" begin
        # 4x4-element matrix → 2×2x2x2 tensor
        m4 = randn(4, 4)
        t4 = DifferenTT.mat2tensor(m4, [2, 2])
        @test size(t4) == (2, 2, 2, 2)

        # 8x8-element matrix → 2×2×2x2x2x2 tensor
        m8 = randn(8, 8)
        t8 = DifferenTT.mat2tensor(m8, [2, 2, 2])
        @test size(t8) == (2, 2, 2, 2, 2, 2)

        # 16-element vector → 2×2×2×2 tensor, or 2x8 tensor
        t24 = DifferenTT.mat2tensor(m8, [2, 4])
        @test size(t24) == (4, 2, 4, 2) # Remember the reversed order!
    end

    @testset "mat2mpo" begin
        n = 4
        sites = siteinds(2, n)
        m = rand(2^n, 2^n)

        mpo = DifferenTT.mat2mpo(m, sites)
        @test mpo isa MPO
        @test length(mpo) == n
        @test all(only.(siteinds(mpo; plev=0)) .== sites)

        @test DifferenTT.contract_mpo(DifferenTT.mat2mpo(m, sites)) ≈ m
    end

    # Regression test: vec2mps and mat2mpo passed the site dimensions to vec2tensor and mat2tensor in site order, which
    # take the least significant leg first, so they scrambled the entries whenever the dimensions were not a palindrome
    @testset "vec2mps and mat2mpo with different site dimensions" begin
        for dims in ([2, 3], [2, 3, 4], [4, 2, 3])
            sites = DifferenTT.site_indices(dims)
            v = collect(1.0:prod(dims))
            m = reshape(collect(1.0:(prod(dims)^2)), prod(dims), prod(dims))
            @test DifferenTT.contract_mps(DifferenTT.vec2mps(v, sites)) ≈ v
            @test DifferenTT.contract_mpo(DifferenTT.mat2mpo(m, sites)) ≈ m
        end
    end

    @testset "arrays2mps and arrays2mpo on a single site" begin
        s1 = DifferenTT.site_indices([3])
        @test DifferenTT.contract_mps(DifferenTT.arrays2mps(s1, [[1.0, 2.0, 3.0]])) == [1.0, 2.0, 3.0]
        M = [1.0 2.0 0.0; 3.0 4.0 0.0; 0.0 0.0 5.0]
        @test DifferenTT.contract_mpo(DifferenTT.arrays2mpo(s1, [M])) == M
    end

    @testset "upper_triang_mpo & lower_triang_mpo" begin
        # Test with a mix of small site dimensions
        dims = [2, 3, 2]
        sites = DifferenTT.site_indices(dims)
        D = prod(dims)

        # Upper with diag
        u_mpo = DifferenTT.upper_triang_mpo(sites, true)
        u_mat = contract_mpo(u_mpo)
        @test u_mat ≈ triu(ones(D, D))

        # Upper without diag
        u_strict = DifferenTT.upper_triang_mpo(sites, false)
        u_strict_mat = contract_mpo(u_strict)
        @test u_strict_mat ≈ triu(ones(D, D), 1)

        # Lower with diag
        l_mpo = DifferenTT.lower_triang_mpo(sites, true)
        l_mat = contract_mpo(l_mpo)
        @test l_mat ≈ tril(ones(D, D))

        # Lower without diag
        l_strict = DifferenTT.lower_triang_mpo(sites, false)
        l_strict_mat = contract_mpo(l_strict)
        @test l_strict_mat ≈ tril(ones(D, D), -1)

        @test_throws DimensionMismatch DifferenTT.triang_projector_mpo(sites, [combiner(sites[1], sites[1]')], :lower)
        @test_throws ArgumentError DifferenTT.triang_projector_mpo(sites, :diagonal)

        # Regression test: a single site threw a BoundsError
        s1 = DifferenTT.site_indices([3])
        @test contract_mpo(DifferenTT.upper_triang_mpo(s1)) ≈ triu(ones(3, 3))
        @test contract_mpo(DifferenTT.upper_triang_mpo(s1, false)) ≈ triu(ones(3, 3), 1)
        @test contract_mpo(DifferenTT.lower_triang_mpo(s1)) ≈ tril(ones(3, 3))
        @test contract_mpo(DifferenTT.lower_triang_mpo(s1, false)) ≈ tril(ones(3, 3), -1)
    end
end
