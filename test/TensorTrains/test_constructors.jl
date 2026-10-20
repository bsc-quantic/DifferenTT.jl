using DifferenTT
using ITensors
using ITensorMPS
using Test
using LinearAlgebra
include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

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

        # leg_dims are given in site order, and the first axis is the most significant (row-major order)
        t16_v2 = DifferenTT.vec2tensor(v16, [2, 8])
        @test size(t16_v2) == (2, 8)
        @test t16_v2 == permutedims(reshape(v16, 8, 2))
        t234 = DifferenTT.vec2tensor(collect(1:24), [2, 3, 4])
        @test size(t234) == (2, 3, 4)
        @test all(t234[i, j, k] == 1 + (i - 1) * 12 + (j - 1) * 4 + (k - 1) for i in 1:2, j in 1:3, k in 1:4)
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
        @test size(t24) == (2, 4, 2, 4)
        @test all(t24[i, j, k, l] == m8[(i-1)*4+j, (k-1)*4+l] for i in 1:2, j in 1:4, k in 1:2, l in 1:4)
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

    # Regression test: vec2tensor and mat2tensor used to take the leg lengths least significant first, while vec2mps and
    # mat2mpo passed them in site order, which scrambled the entries whenever the dimensions were not a palindrome
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

    # The constant MPOs and MPSs, checked densely for each way of giving the sites: indices, dimensions (unequal), a
    # QTT or a TT architecture, and a single site
    @testset "identity, zeros and ones against dense references" begin
        cases = [
            DifferenTT.site_indices([2, 3, 2]),
            [2, 3, 4],
            QTTArchitecture((2, 1); interleaved=true),
            TTArchitecture((3, 2), Dict(1 => 2, 2 => 1)),
            DifferenTT.site_indices([3]),
            [5],
        ]
        sitedims(c::Vector{<:Index}) = dim.(c)
        sitedims(c::Vector{Int}) = c
        sitedims(c::DifferenTT.Architecture) = collect(site_dims(c))
        for c in cases
            d = sitedims(c)
            D = prod(d)
            N = length(d)

            Id = identity_mpo(c)
            @test dim.(only.(siteinds(Id; plev=0))) == d
            @test maxlinkdim(Id) == 1
            @test todense(Id) == I(D)

            Z = zeros_mpo(c)
            @test todense(Z) == zeros(D, D)
            Zs = zeros_mps(c)
            @test todense(Zs) == zeros(D)
            @test dim.(siteinds(Zs)) == d

            @test todense(ones_mps(c)) == ones(D)
            @test todense(ones_mpo(c)) == ones(D, D)
            # with bond dimension χ every entry is the number of paths through the bonds, χ^(N-1)
            @test todense(ones_mps(c, 3)) == fill(3.0^(N - 1), D)
            @test todense(ones_mpo(c, 2)) == fill(2.0^(N - 1), D, D)
            if N > 1
                @test linkdims(ones_mps(c)) == fill(1, N - 1)
                @test linkdims(ones_mpo(c, 2)) == fill(2, N - 1)
                @test maxlinkdim(zeros_mps(c; linkdims=2)) == 2
            end
        end

        # the given link indices are used as they are
        sites = DifferenTT.site_indices([2, 2, 2])
        links = DifferenTT.link_indices([2, 3])
        M = ones_mps(sites, links)
        @test linkinds(M) == links
        @test todense(M) == fill(6.0, 8)
        O = ones_mpo(sites, links)
        @test linkinds(O) == links
        @test todense(O) == fill(6.0, 8, 8)

        @test_throws DimensionMismatch ones_mps(sites, links[1:1])
        @test_throws DimensionMismatch ones_mpo(sites, links[1:1])
    end

    @testset "make_pass_identity_site" begin
        s = Index(3, "Site")
        l1, l2 = Index(2, "Link,l=1"), Index(2, "Link,l=2")
        T = DifferenTT.make_pass_identity_site(s, l1, l2, 2, Array)
        @test hasinds(T, s, s', l1, l2)
        @test all(T[s=>i, s'=>j, l1=>a, l2=>b] == (i == j && a == b) for i in 1:3, j in 1:3, a in 1:2, b in 1:2)
        # pass_dim smaller than the links leaves the remaining channels empty
        T1 = DifferenTT.make_pass_identity_site(s, l1, l2, 1, Array)
        @test norm(T1) ≈ sqrt(3)
        @test T1[s=>2, s'=>2, l1=>1, l2=>1] == 1.0
    end

    # P·vec(X) = vec(tril(X)) (or triu), in the vectorization of the docstring: site k of the vector merges the column
    # (sites[k]) and the row (sites[k]') of site k of X
    @testset "triang_projector_mpo is the projector onto the triangular entries" begin
        tri = Dict(
            (:lower, true) => tril,
            (:lower, false) => X -> tril(X, -1),
            (:upper, true) => triu,
            (:upper, false) => X -> triu(X, 1),
        )
        for d in ([2], [2, 3], [3, 2, 2])
            sites = DifferenTT.site_indices(d)
            D = prod(d)
            X = randn(rng(), D, D)
            Xm = DifferenTT.mat2mpo(X, sites)
            for which in (:lower, :upper), incl in (true, false)
                P = DifferenTT.triang_projector_mpo(sites, which, incl)
                @test maxlinkdim(P) <= 2
                # the combiners of the second method are those of the projector's indices: rebuild them to vectorize X
                combs = [combiner(sites[k], sites[k]'; tags="csite=$k") for k in eachindex(sites)]
                cinds = only.(siteinds(P; plev=0))
                xv = MPS([Xm[k] * combs[k] * delta(combinedind(combs[k]), cinds[k]) for k in eachindex(sites)])
                yv = noprime(contract(P, xv))
                Y = MPO([yv[k] * delta(cinds[k], combinedind(combs[k])) * combs[k] for k in eachindex(sites)])
                @test todense(Y) ≈ tri[(which, incl)](X)
                # it is a diagonal 0/1 operator, i.e. a projector
                Pd = todense(P)
                @test Pd ≈ Diagonal(diag(Pd))
                @test sort(unique(round.(diag(Pd); digits=12))) ⊆ [0.0, 1.0]
                @test sum(diag(Pd)) ≈ (incl ? D * (D + 1) ÷ 2 : D * (D - 1) ÷ 2)
            end
        end
        # one site of dimension 2: the diagonal (1, 0, 1, 1) of Eq. (20) of the FAI paper, for row >= column, with the
        # combiner merging (column, row) as fai_dg does
        s = DifferenTT.site_indices([2])
        helper = DifferenTT.site_indices([2])
        P = DifferenTT.triang_projector_mpo(s, [combiner(s[1], helper[1])], :lower)
        @test diag(todense(P)) == [1.0, 0.0, 1.0, 1.0]
    end

    @testset "arrays2mps and arrays2mpo round trips" begin
        r = rng()
        sites = DifferenTT.site_indices([2, 3, 2])
        A = [randn(r, 2, 2), randn(r, 3, 2, 4), randn(r, 2, 4)]
        M = DifferenTT.arrays2mps(sites, A)
        @test linkdims(M) == [2, 4]
        ref = [sum(A[1][i, a] * A[2][j, a, b] * A[3][k, b] for a in 1:2, b in 1:4) for i in 1:2, j in 1:3, k in 1:2]
        @test todense(M) ≈ vec(permutedims(ref, (3, 2, 1)))
        @test DifferenTT.vec2tensor(todense(M), [2, 3, 2]) ≈ ref

        # with given link indices, which the MPS then carries
        links = DifferenTT.link_indices([2, 4])
        M2 = DifferenTT.arrays2mps(sites, links, A)
        @test linkinds(M2) == links
        @test todense(M2) ≈ todense(M)
        @test_throws DimensionMismatch DifferenTT.arrays2mps(sites, DifferenTT.link_indices([2, 3]), A)

        B = [randn(r, 2, 2, 3), randn(r, 3, 3, 3, 2), randn(r, 2, 2, 2)]
        O = DifferenTT.arrays2mpo(sites, B)
        @test linkdims(O) == [3, 2]
        # entry (row (i,j,k), column (i',j',k')): the first axis of each array is the output (row) index
        refO = [
            sum(B[1][i, ip, a] * B[2][j, jp, a, b] * B[3][k, kp, b] for a in 1:3, b in 1:2) for
            i in 1:2, j in 1:3, k in 1:2, ip in 1:2, jp in 1:3, kp in 1:2
        ]
        @test DifferenTT.mat2tensor(todense(O), [2, 3, 2]) ≈ refO
        links = DifferenTT.link_indices([3, 2])
        O2 = DifferenTT.arrays2mpo(sites, links, B)
        @test linkinds(O2) == links
        @test todense(O2) ≈ todense(O)
        @test_throws DimensionMismatch DifferenTT.arrays2mpo(sites, DifferenTT.link_indices([3, 3]), B)

        # contract → vec2mps/mat2mpo → contract
        @test todense(DifferenTT.vec2mps(todense(M), sites)) ≈ todense(M)
        @test todense(DifferenTT.mat2mpo(todense(O), sites)) ≈ todense(O)
    end

    @testset "vec2mps and mat2mpo pass the truncation keywords" begin
        sites = DifferenTT.site_indices([2, 2, 2, 2])
        v = randn(rng(), 16)
        @test maxlinkdim(DifferenTT.vec2mps(v, sites; maxdim=2)) == 2
        u = kron([1.0, 2.0], [1.0, -1.0], [3.0, 1.0], [0.5, 2.0])
        @test maxlinkdim(DifferenTT.vec2mps(u, sites; cutoff=1e-14)) == 1
        m = randn(rng(), 16, 16)
        @test maxlinkdim(DifferenTT.mat2mpo(m, sites; maxdim=3)) == 3
        @test maxlinkdim(DifferenTT.mat2mpo(kron(m[1:2, 1:2], m[3:4, 3:4], I(4)), sites; cutoff=1e-14)) == 1
    end

    # Regression test: on a single site ones_mps and ones_mpo skipped the adapt to array_type. Array{Float32} stands in
    # for a GPU backend: adapt converts the element type, so a tensor left out of the pipeline stays Float64
    @testset "array_type reaches the tensors of ones_mps and ones_mpo" begin
        for n in (1, 3)
            sites = DifferenTT.site_indices(fill(2, n))
            @test all(eltype(t) == Float32 for t in ones_mps(sites, Array{Float32}))
            @test all(eltype(t) == Float32 for t in ones_mpo(sites, Array{Float32}))
            if n > 1  # one site: a uniform delta tensor, which stores no array
                @test all(eltype(t) == Float32 for t in identity_mpo(sites, Array{Float32}))
            end
        end
    end
end
